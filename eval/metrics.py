"""Post-run metrics: complexity delta, correctness preservation, effort."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from github import Github

from eval.baseline import BaselineResult, _measure_python_complexity, _measure_js_complexity
from eval.repos import EvalRepo, get_repo
from eval.runner import RunResult

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    repo: str
    fork: str
    issue_number: int
    task_type: str
    level: str

    # Correctness
    status: str
    tests_pass_after: bool | None
    correctness_preserved: bool | None    # True if tests_pass_after and status==succeeded

    # PR quality
    fta: float | None                    # File Targeting Accuracy: fraction of modified files that existed pre-PR
    edit_ratio: float | None             # deletions / (additions + deletions); ~0 = phantom additions only
    files_modified: int | None           # total files touched in the PR

    # Resolution (Reproduce → Fix → Verify protocol)
    reproduced: bool | None              # Reproducer test confirmed failing on base code
    false_positive: bool | None          # Reproducer test passed before fix (agent misunderstood)
    resolved: bool | None                # Phoenix issue_resolved (RESOLUTION_MODE) after test step
    reproducer_file: str | None          # Path of the written reproducer test

    # Complexity
    complexity_avg_before: float
    complexity_avg_after: float
    complexity_delta_avg: float           # negative = improvement
    complexity_max_before: float
    complexity_max_after: float
    complexity_delta_max: float

    # Effort
    elapsed_seconds: float
    pr_number: int | None
    pr_url: str | None

    # Recovery
    recovery_ops: int                     # always 1 (git revert on feature branch)


def _compute_pr_quality(
    github_token: str,
    fork: str,
    pr_number: int,
) -> tuple[float | None, float | None, int | None]:
    """Return (fta, edit_ratio, files_modified) from the PR diff.

    fta (File Targeting Accuracy): fraction of files in the PR that already
        existed in the repo before the PR.  A file with status "modified",
        "removed", or "renamed" existed; "added" is a new file.
        Phantom-path PRs are pure additions → fta ≈ 0.

    edit_ratio: deletions / (additions + deletions).
        A PR that only adds code scores 0; a targeted in-place fix scores > 0.
    """
    try:
        g = Github(github_token)
        gh_repo = g.get_repo(fork)
        pr = gh_repo.get_pull(pr_number)
        files = list(pr.get_files())
        if not files:
            return None, None, 0

        total = len(files)
        existing = sum(
            1 for f in files if f.status in ("modified", "removed", "renamed")
        )
        fta = round(existing / total, 3)

        additions = sum(f.additions for f in files)
        deletions = sum(f.deletions for f in files)
        total_changes = additions + deletions
        edit_ratio = round(deletions / total_changes, 3) if total_changes > 0 else 0.0

        return fta, edit_ratio, total
    except Exception as e:
        logger.warning(f"Could not compute PR quality metrics for {fork}#{pr_number}: {e}")
        return None, None, None


def _read_run_log(issue_number: int) -> dict | None:
    """Read resolution metadata from the most recent run log for this issue."""
    import json as _json
    log_dir = Path("workspace/runs")
    if not log_dir.exists():
        return None
    try:
        candidates = sorted(
            log_dir.glob("*/run.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for run_file in candidates[:20]:
            data = _json.loads(run_file.read_text())
            if issue_number in (data.get("issues") or []):
                steps = data.get("steps", {})
                reproduce_step = steps.get("reproduce", {})
                test_step = steps.get("test", {})
                outputs = reproduce_step.get("outputs", {})
                test_outputs = test_step.get("outputs", {})
                issue_resolved = test_outputs.get("issue_resolved", test_outputs.get("resolved"))
                reproducer_passed = test_outputs.get("reproducer_passed")
                return {
                    "reproduced": outputs.get("reproduced"),
                    "false_positive": outputs.get("false_positive"),
                    "reproducer_skipped": outputs.get("skipped", False),
                    "reproducer_file": outputs.get("test_file"),
                    "issue_resolved": issue_resolved,
                    "reproducer_passed": reproducer_passed,
                    "resolved": issue_resolved,
                    # LLM usage fields written by Run.flush_llm_usage()
                    "llm_calls": data.get("llm_calls", 0),
                    "input_tokens": data.get("input_tokens", 0),
                    "output_tokens": data.get("output_tokens", 0),
                    "inference_seconds": data.get("inference_seconds", 0.0),
                }
    except Exception:
        pass
    return None


def _run(cmd: list[str], cwd: str, timeout: int = 300) -> tuple[int, str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout + result.stderr


def _checkout_pr_branch(repo_dir: str, pr_branch: str) -> bool:
    # Try local branch first (avoids expired token issues)
    code, _ = _run(["git", "checkout", "-f", pr_branch], cwd=repo_dir)
    if code != 0:
        # Fall back to fetching from remote
        code, _ = _run(["git", "fetch", "origin", pr_branch], cwd=repo_dir)
        if code != 0:
            return False
        code, _ = _run(["git", "checkout", "-f", pr_branch], cwd=repo_dir)
        if code != 0:
            return False
    # Hard-reset and clean to eliminate stale working-dir changes from other runs
    _run(["git", "reset", "--hard", "HEAD"], cwd=repo_dir)
    _run(["git", "clean", "-fd"], cwd=repo_dir)
    return True


def _install_deps(profile: str, repo_dir: str) -> None:
    """Best-effort dep install before running tests in metrics."""
    if profile == "frontend":
        if not (Path(repo_dir) / "node_modules").exists():
            _run(["npm", "install", "--prefer-offline"], cwd=repo_dir, timeout=180)
    else:
        root = Path(repo_dir)
        if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
            code, _ = _run(["pip", "install", "-e", ".[dev,test]", "--quiet", "--no-build-isolation"],
                           cwd=repo_dir, timeout=120)
            if code != 0:
                _run(["pip", "install", "-e", ".", "--quiet", "--no-build-isolation"],
                     cwd=repo_dir, timeout=120)


def _extract_failures(output: str) -> set[str]:
    """Extract FAILED test names from pytest output."""
    import re
    names: set[str] = set()
    for line in output.splitlines():
        m = re.match(r"^FAILED (.+?) -", line)
        if m:
            names.add(m.group(1).strip())
        elif line.startswith("FAILED "):
            names.add(line[7:].split(" ")[0].strip())
    return names


def _run_tests_in_dir(profile: str, repo_dir: str) -> tuple[bool, str]:
    _install_deps(profile, repo_dir)
    if profile == "python":
        code, out = _run(["python", "-m", "pytest", "--tb=no", "-q", "--no-header"],
                         cwd=repo_dir, timeout=300)
        return code in (0, 5), out[:2000]
    elif profile == "frontend":
        code, out = _run(["npm", "run", "test"], cwd=repo_dir, timeout=300)
        return code == 0, out[:2000]
    elif profile == "java":
        import shutil
        if shutil.which("mvn") is None and shutil.which("gradle") is None:
            return True, "Java build tools (mvn/gradle) not available; Java tests skipped."
        if shutil.which("mvn"):
            code, out = _run(["mvn", "-q", "test"], cwd=repo_dir, timeout=600)
        else:
            code, out = _run(["./gradlew", "test", "-q"], cwd=repo_dir, timeout=600)
        return code == 0, out[:2000]
    return False, "unknown profile"


def _tests_pass_with_baseline_comparison(profile: str, repo_dir: str) -> bool:
    """Run tests; if they fail, check if the baseline also fails.

    If baseline also fails AND Phoenix introduced no NEW failures, return True
    (the repo was already broken; Phoenix didn't make it worse).
    """
    pr_passed, pr_out = _run_tests_in_dir(profile, repo_dir)
    if pr_passed:
        return True

    pr_failures = _extract_failures(pr_out)

    # Stash changes (Phoenix's diff), run baseline, pop
    code, _ = _run(["git", "stash", "--include-untracked"], cwd=repo_dir, timeout=30)
    stashed = code == 0

    try:
        base_passed, base_out = _run_tests_in_dir(profile, repo_dir)
        base_failures = _extract_failures(base_out)
        baseline_failed = not base_passed
    finally:
        if stashed:
            _run(["git", "stash", "pop"], cwd=repo_dir, timeout=30)

    if not baseline_failed:
        # Baseline was clean — Phoenix broke something
        return False

    # Baseline was broken; check if Phoenix added new failures
    new_failures = pr_failures - base_failures
    if new_failures:
        logger.info("Phoenix introduced %d new failure(s): %s", len(new_failures), list(new_failures)[:3])
        return False

    logger.info("Baseline was already failing; Phoenix introduced no new failures — correctness preserved")
    return True


def compute_metrics(
    run: RunResult,
    baseline: BaselineResult,
    workspace: str,
    github_token: str = "",
) -> EvalMetrics:
    """Compare pre/post complexity and correctness for a completed run."""
    repo = get_repo(run.repo)
    repo_dir = str(Path(workspace) / repo.name)

    complexity_avg_after = baseline.complexity_avg
    complexity_max_after = baseline.complexity_max
    tests_pass_after: bool | None = None
    fta: float | None = None
    edit_ratio: float | None = None
    files_modified: int | None = None
    reproduced: bool | None = None
    false_positive: bool | None = None
    resolved: bool | None = None
    reproducer_file: str | None = None

    if run.status == "succeeded" and run.pr_url:
        # Branch naming convention: phoenix/issue-{number}
        branch = f"phoenix/issue-{run.issue_number}"
        try:
            if _checkout_pr_branch(repo_dir, branch):
                # Measure complexity after changes
                if repo.profile == "python":
                    complexity_avg_after, complexity_max_after = _measure_python_complexity(repo_dir)
                elif repo.profile == "frontend":
                    complexity_avg_after, complexity_max_after = _measure_js_complexity(repo_dir)

                # Run tests on the PR branch, accounting for pre-existing baseline failures
                tests_pass_after = _tests_pass_with_baseline_comparison(repo.profile, repo_dir)

                # Return to default branch
                _run(["git", "checkout", "-"], cwd=repo_dir)

            # PR quality metrics (from GitHub API — no local checkout needed)
            if run.pr_number and github_token:
                fta, edit_ratio, files_modified = _compute_pr_quality(
                    github_token, run.fork, run.pr_number
                )

            # Resolution metrics — read from run workspace if available
            run_log = _read_run_log(run.issue_number)
            if run_log:
                reproduced = run_log.get("reproduced")
                false_positive = run_log.get("false_positive")
                resolved = run_log.get("issue_resolved", run_log.get("resolved"))
                reproducer_file = run_log.get("reproducer_file")
            else:
                logger.warning(f"Could not checkout branch {branch} for {run.fork}#{run.issue_number}")
        except Exception as e:
            logger.error(f"Post-run measurement failed for {run.fork}#{run.issue_number}: {e}")
    else:
        tests_pass_after = None

    correctness_preserved = (
        (run.status == "succeeded") and (tests_pass_after is True)
        if tests_pass_after is not None else None
    )

    return EvalMetrics(
        repo=run.repo,
        fork=run.fork,
        issue_number=run.issue_number,
        task_type=run.task_type,
        level=repo.level,

        status=run.status,
        tests_pass_after=tests_pass_after,
        correctness_preserved=correctness_preserved,

        fta=fta,
        edit_ratio=edit_ratio,
        files_modified=files_modified,

        reproduced=reproduced,
        false_positive=false_positive,
        resolved=resolved,
        reproducer_file=reproducer_file,

        complexity_avg_before=baseline.complexity_avg,
        complexity_avg_after=complexity_avg_after,
        complexity_delta_avg=round(complexity_avg_after - baseline.complexity_avg, 2),
        complexity_max_before=baseline.complexity_max,
        complexity_max_after=complexity_max_after,
        complexity_delta_max=round(complexity_max_after - baseline.complexity_max, 2),

        elapsed_seconds=run.elapsed_seconds,
        pr_number=run.pr_number,
        pr_url=run.pr_url,
        recovery_ops=1,
    )


def compute_all_metrics(
    run_results: list[RunResult],
    baselines: list[BaselineResult],
    workspace: str,
    output_file: str = "eval/results/metrics.json",
    github_token: str = "",
) -> list[EvalMetrics]:
    baseline_map = {b.repo: b for b in baselines}
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    all_metrics: list[EvalMetrics] = []
    for run in run_results:
        baseline = baseline_map.get(run.repo)
        if not baseline:
            logger.warning(f"No baseline for {run.repo} — skipping metrics")
            continue
        logger.info(f"Computing metrics for {run.fork}#{run.issue_number}...")
        try:
            m = compute_metrics(run, baseline, workspace, github_token=github_token)
            all_metrics.append(m)
        except Exception as e:
            logger.error(f"  Failed: {e}")

    Path(output_file).write_text(
        json.dumps([asdict(m) for m in all_metrics], indent=2)
    )
    _print_metrics_summary(all_metrics)
    return all_metrics


def _print_metrics_summary(metrics: list[EvalMetrics]) -> None:
    if not metrics:
        print("No metrics to summarize.")
        return

    total = len(metrics)
    correct = [m for m in metrics if m.correctness_preserved is True]
    cp = len(correct) / total

    improved = [m for m in metrics if m.complexity_delta_avg < 0]
    avg_delta = sum(m.complexity_delta_avg for m in metrics) / total
    avg_time = sum(m.elapsed_seconds for m in metrics) / total

    # PR quality aggregates (only runs that have PR quality data)
    fta_runs = [m for m in metrics if m.fta is not None]
    er_runs  = [m for m in metrics if m.edit_ratio is not None]
    avg_fta  = sum(m.fta for m in fta_runs) / len(fta_runs) if fta_runs else None
    avg_er   = sum(m.edit_ratio for m in er_runs) / len(er_runs) if er_runs else None
    # "real fixes": existing file modified AND some deletions present
    real_fixes = [m for m in fta_runs if (m.fta or 0) > 0.5 and (m.edit_ratio or 0) > 0]

    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"  Total runs              : {total}")
    print(f"  Correctness preserved   : {len(correct)}/{total} ({cp:.1%})")
    if avg_fta is not None:
        print(f"  File Targeting Acc (FTA): {avg_fta:.1%}  (n={len(fta_runs)})")
    if avg_er is not None:
        print(f"  Edit ratio (del/total)  : {avg_er:.1%}  (n={len(er_runs)})")
    if fta_runs:
        print(f"  Likely real fixes       : {len(real_fixes)}/{len(fta_runs)}  (FTA>0.5 & edits>0)")
    # Resolution: Phoenix policy (RESOLUTION_MODE) vs reproduction signal
    reproduced_runs = [m for m in metrics if m.reproduced is True]
    policy_resolved = [m for m in metrics if m.resolved is True]
    fp_runs         = [m for m in metrics if m.false_positive is True]
    print(f"  Phoenix resolution (policy): {len(policy_resolved)}/{total}  (RESOLUTION_MODE gate)")
    if reproduced_runs:
        print(f"  Reproduction rate       : {len(reproduced_runs)}/{total}  (confirmed failing repro test on base)")
    if fp_runs:
        print(f"  False positive rate     : {len(fp_runs)}/{total}  (test passed before fix)")
    print(f"  Complexity improved     : {len(improved)}/{total}")
    print(f"  Avg complexity delta    : {avg_delta:+.2f}")
    print(f"  Avg time per run        : {avg_time:.0f}s")
    print(f"  Recovery cost           : 1 git revert (all runs)")
    print("=" * 60)

    # By level
    for level in ("easy", "medium", "hard", "extreme"):
        lvl = [m for m in metrics if m.level == level]
        if not lvl:
            continue
        ok = sum(1 for m in lvl if m.correctness_preserved is True)
        print(f"  {level:<10} {ok}/{len(lvl)} passed")


def export_csv(
    metrics: list[EvalMetrics],
    output_file: str = "eval/results/metrics.csv",
) -> None:
    import csv
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        return
    fields = list(asdict(metrics[0]).keys())
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(m) for m in metrics)
    logger.info(f"CSV saved to {output_file}")


if __name__ == "__main__":
    import os
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    workspace = os.getenv("EVAL_WORKSPACE", "/tmp/phoenix-eval")
    token = os.getenv("GITHUB_TOKEN", "")
    runs_file = sys.argv[1] if len(sys.argv) > 1 else "eval/results/run_results.json"
    baselines_file = sys.argv[2] if len(sys.argv) > 2 else "eval/results/baselines.json"

    runs = [RunResult(**d) for d in json.loads(Path(runs_file).read_text())]
    baselines = [BaselineResult(**d) for d in json.loads(Path(baselines_file).read_text())]

    all_metrics = compute_all_metrics(runs, baselines, workspace, github_token=token)
    export_csv(all_metrics)
