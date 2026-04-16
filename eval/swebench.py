"""SWE-bench evaluation mode for Phoenix.

Loads instances from the HuggingFace SWE-bench dataset, forks repos at the
correct base commit, runs Phoenix via the normal webhook pipeline, then
evaluates resolution using the oracle FAIL_TO_PASS tests.

Usage:
    python -m eval.main_swebench --tier lite --max-per-repo 3

Requires:
    pip install datasets  (HuggingFace datasets library)
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from github import Github, GithubException

logger = logging.getLogger(__name__)

# SWE-bench dataset identifiers on HuggingFace
DATASET_IDS = {
    "lite":     "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full":     "princeton-nlp/SWE-bench",
}

# Python repos from SWE-bench Lite that Phoenix handles well
# (pure Python, pip-installable, pytest-compatible)
SUPPORTED_REPOS = {
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "psf/requests",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "pallets/flask",
    "sympy/sympy",
    "marshmallow-code/marshmallow",
    "pypa/pip",
}


@dataclass
class SWEBenchInstance:
    instance_id: str          # e.g. "django__django-15789"
    repo: str                 # e.g. "django/django"
    base_commit: str          # git SHA to evaluate against
    problem_statement: str    # the issue text shown to Phoenix
    test_patch: str           # oracle test patch (applied post-run to evaluate)
    fail_to_pass: list[str]   # test IDs that should go fail→pass
    pass_to_pass: list[str]   # test IDs that must stay passing
    hints_text: str = ""


@dataclass
class SWEBenchResult:
    instance_id: str
    repo: str
    fork: str
    issue_number: int
    issue_url: str
    pr_number: int | None
    pr_url: str | None
    status: str               # "succeeded" | "failed" | "timeout"
    # Oracle resolution (SWE-bench FAIL_TO_PASS tests)
    resolved_oracle: bool | None  # True if all FAIL_TO_PASS oracle tests pass
    cp: bool | None               # True if no PASS_TO_PASS tests broke
    fail_to_pass_passed: int      # how many oracle tests now pass
    fail_to_pass_total: int
    pass_to_pass_broken: int      # how many previously-passing tests broke
    # Phoenix resolution (test step; see RESOLUTION_MODE, default tests-only)
    reproduced: bool | None           # Reproducer confirmed issue exists (test failed on base)
    phoenix_resolved: bool | None     # issue_resolved per RESOLUTION_MODE (tests / reproducer / both)
    resolved_reproducer: bool | None  # Strict: synthetic reproducer test passes after fix (reproducer_passed)
    false_positive: bool | None       # Reproducer test passed before fix (misunderstood)
    reproducer_skipped: bool          # Could not write a failing test (non-blocking)
    reproducer_file: str | None       # Path of written reproducer test
    # PR quality
    fta: float | None
    edit_ratio: float | None
    elapsed_seconds: float
    # Observability (SWE-bench harness + Phoenix wait)
    wait_cap_seconds: int = 0
    wait_timed_out: bool = False
    phoenix_final_label: str | None = None
    oracle_ran: bool = False
    oracle_skip_reason: str | None = None


@dataclass
class OracleEvalOutcome:
    """Result of applying oracle test_patch and running FAIL_TO_PASS / PASS_TO_PASS."""

    resolved: bool | None
    cp: bool | None
    fail_to_pass_passed: int
    fail_to_pass_total: int
    pass_to_pass_broken: int
    skip_reason: str | None = None


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_swebench_instances(
    tier: Literal["lite", "verified", "full"] = "lite",
    repos: set[str] | None = None,
    max_per_repo: int = 3,
) -> list[SWEBenchInstance]:
    """Load SWE-bench instances, filtered to supported repos."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets not installed. Run: pip install datasets"
        )

    dataset_id = DATASET_IDS[tier]
    logger.info(f"Loading SWE-bench dataset: {dataset_id}")
    ds = load_dataset(dataset_id, split="test")

    filter_repos = repos or SUPPORTED_REPOS
    per_repo: dict[str, int] = {}
    instances: list[SWEBenchInstance] = []

    for row in ds:
        repo = row["repo"]
        if repo not in filter_repos:
            continue
        if per_repo.get(repo, 0) >= max_per_repo:
            continue

        fail_to_pass = json.loads(row["FAIL_TO_PASS"]) if isinstance(row["FAIL_TO_PASS"], str) else row["FAIL_TO_PASS"]
        pass_to_pass = json.loads(row["PASS_TO_PASS"]) if isinstance(row["PASS_TO_PASS"], str) else row["PASS_TO_PASS"]

        instances.append(SWEBenchInstance(
            instance_id=row["instance_id"],
            repo=repo,
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            test_patch=row.get("test_patch", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            hints_text=row.get("hints_text", ""),
        ))
        per_repo[repo] = per_repo.get(repo, 0) + 1

    logger.info(f"Loaded {len(instances)} instances from {len(per_repo)} repos")
    return instances


# ── Fork setup ────────────────────────────────────────────────────────────────

def _run(cmd: list[str], cwd: str | None = None, timeout: int = 300) -> tuple[int, str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout + result.stderr


def prepare_swebench_fork(
    instance: SWEBenchInstance,
    github_token: str,
    workspace: str,
    your_username: str,
) -> str:
    """Fork the repo (if needed) and reset the fork's default branch to base_commit.

    Returns the fork full name, e.g. "kkipngenokoech/django".
    """
    g = Github(github_token)
    repo_name = instance.repo.split("/")[1]
    fork_full = f"{your_username}/{repo_name}"

    # Fork if it doesn't exist
    try:
        fork = g.get_repo(fork_full)
        logger.info(f"  Fork exists: {fork_full}")
    except GithubException:
        upstream = g.get_repo(instance.repo)
        fork = g.get_user().create_fork(upstream)
        logger.info(f"  Created fork: {fork_full}")
        time.sleep(5)  # GitHub takes a moment to provision forks

    # Ensure required AI labels exist on the fork
    from eval.baseline import ensure_labels
    ensure_labels(fork_full, github_token)

    # Clone or update local copy
    repo_dir = str(Path(workspace) / repo_name)
    if not Path(repo_dir).exists():
        logger.info(f"  Cloning {fork_full}...")
        code, out = _run(
            ["git", "clone", f"https://x-access-token:{github_token}@github.com/{fork_full}.git", repo_dir],
            timeout=120,
        )
        if code != 0:
            raise RuntimeError(f"Clone failed: {out[:300]}")
    else:
        _run(["git", "fetch", "origin"], cwd=repo_dir, timeout=60)

    # Reset fork to base_commit
    logger.info(f"  Resetting {fork_full} to {instance.base_commit[:8]}...")
    _run(["git", "checkout", "main"], cwd=repo_dir)
    # Try main, then master
    code, _ = _run(["git", "checkout", "main"], cwd=repo_dir)
    if code != 0:
        _run(["git", "checkout", "master"], cwd=repo_dir)

    # Fetch the base commit from the upstream (it may not be in the fork yet)
    code, _ = _run(
        ["git", "fetch", f"https://github.com/{instance.repo}.git", instance.base_commit],
        cwd=repo_dir, timeout=60,
    )
    _run(["git", "reset", "--hard", instance.base_commit], cwd=repo_dir)
    code, _ = _run(
        ["git", "push", "origin", "HEAD", "--force"],
        cwd=repo_dir, timeout=60,
    )
    if code != 0:
        logger.warning(f"  Force-push to {fork_full} failed — fork may be out of sync")

    return fork_full


def mirror_swebench_issue(
    instance: SWEBenchInstance,
    fork_full: str,
    github_token: str,
) -> tuple[int, str]:
    """Create a GitHub issue on the fork with the SWE-bench problem statement.

    Returns (issue_number, issue_url).
    """
    g = Github(github_token)
    fork_repo = g.get_repo(fork_full)

    body = (
        f"{instance.problem_statement}\n\n"
        f"---\n"
        f"*SWE-bench instance: `{instance.instance_id}`  "
        f"Base commit: `{instance.base_commit[:8]}`*"
    )
    issue = fork_repo.create_issue(
        title=f"[SWE-bench] {instance.instance_id}",
        body=body,
    )
    logger.info(f"  Created issue #{issue.number}: {issue.html_url}")
    return issue.number, issue.html_url


# ── Oracle evaluation ─────────────────────────────────────────────────────────

def _apply_patch(patch_text: str, repo_dir: str) -> bool:
    """Apply a unified diff patch string to the repo. Returns True on success.

    Tries a plain apply first, then ``git apply --3way`` so oracle ``test_patch`` can
    merge when the agent already edited overlapping lines (common on pytest).
    """
    if not patch_text.strip():
        return True
    r1 = subprocess.run(
        ["git", "apply", "--whitespace=fix", "-"],
        input=patch_text,
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r1.returncode == 0:
        return True
    logger.warning("  Patch apply failed (will try 3-way): %s", (r1.stderr or "")[:300])
    r2 = subprocess.run(
        ["git", "apply", "--3way", "--whitespace=fix", "-"],
        input=patch_text,
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if r2.returncode == 0:
        logger.info("  Oracle test_patch applied via 3-way merge")
        return True
    logger.warning("  Patch apply failed after 3-way: %s", (r2.stderr or "")[:400])
    return False


def _run_specific_tests(
    test_ids: list[str],
    repo_dir: str,
    timeout: int = 300,
) -> tuple[set[str], set[str]]:
    """Run a specific list of pytest test IDs. Returns (passed, failed) sets."""
    if not test_ids:
        return set(), set()

    cmd = ["python", "-m", "pytest", "--tb=no", "-q", "--no-header"] + test_ids[:50]
    result = subprocess.run(
        cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
    )
    output = result.stdout + result.stderr

    passed: set[str] = set()
    failed: set[str] = set()
    for line in output.splitlines():
        if line.startswith("PASSED"):
            passed.add(line.split()[1] if len(line.split()) > 1 else line)
        elif line.startswith("FAILED"):
            tid = line.split("::")[0].replace("FAILED ", "").strip()
            failed.add(tid)
    # If pytest exit code 0, all ran tests passed
    if result.returncode == 0:
        passed = set(test_ids)
        failed = set()
    elif result.returncode in (1,):
        # Some tests failed — extract from output
        import re
        for line in output.splitlines():
            m = re.match(r"^FAILED (.+?) -", line)
            if m:
                failed.add(m.group(1).strip())
        passed = set(test_ids) - failed

    return passed, failed


def evaluate_resolution(
    instance: SWEBenchInstance,
    pr_branch: str,
    repo_dir: str,
) -> OracleEvalOutcome:
    """Check if Phoenix's PR resolves the SWE-bench instance.

    Protocol:
      1. Checkout Phoenix's PR branch.
      2. Apply the oracle test_patch (adds the issue-specific tests).
      3. Run FAIL_TO_PASS tests — they should now pass.
      4. Run PASS_TO_PASS tests — they should still pass.

    ``skip_reason`` is set when oracle tests were not run (checkout/patch).
    """
    ftp_total = len(instance.fail_to_pass)
    # Checkout PR branch
    code, _ = _run(["git", "checkout", "-f", pr_branch], cwd=repo_dir)
    if code != 0:
        code, _ = _run(["git", "fetch", "origin", pr_branch], cwd=repo_dir)
        if code != 0:
            logger.warning(f"  Cannot checkout {pr_branch}")
            return OracleEvalOutcome(
                None, None, 0, ftp_total, 0, skip_reason="oracle_pr_branch_checkout_failed",
            )
        _run(["git", "checkout", "-f", pr_branch], cwd=repo_dir)

    # Apply oracle test patch
    if instance.test_patch:
        ok = _apply_patch(instance.test_patch, repo_dir)
        if not ok:
            logger.warning("  Oracle test patch failed to apply — skipping oracle eval")
            _run(["git", "checkout", "-f", pr_branch], cwd=repo_dir)
            return OracleEvalOutcome(
                None, None, 0, ftp_total, 0, skip_reason="oracle_test_patch_apply_failed",
            )

    # Evaluate FAIL_TO_PASS
    ftp_passed, ftp_failed = _run_specific_tests(instance.fail_to_pass, repo_dir)
    ftp_pass_count = len(ftp_passed)
    resolved = (ftp_pass_count == ftp_total) if ftp_total > 0 else None

    # Evaluate PASS_TO_PASS
    ptp_passed, ptp_failed = _run_specific_tests(instance.pass_to_pass[:30], repo_dir)
    ptp_broken = len(ptp_failed)
    cp = ptp_broken == 0

    logger.info(
        f"  Oracle: FAIL_TO_PASS {ftp_pass_count}/{ftp_total} passed | "
        f"PASS_TO_PASS {ptp_broken} broken → resolved={resolved} cp={cp}"
    )

    # Restore branch without patch
    _run(["git", "checkout", "-f", pr_branch], cwd=repo_dir)

    return OracleEvalOutcome(resolved, cp, ftp_pass_count, ftp_total, ptp_broken, skip_reason=None)


# ── PR quality ────────────────────────────────────────────────────────────────

def _compute_pr_quality(
    github_token: str,
    fork: str,
    pr_number: int,
) -> tuple[float | None, float | None]:
    try:
        g = Github(github_token)
        gh_repo = g.get_repo(fork)
        pr = gh_repo.get_pull(pr_number)
        files = list(pr.get_files())
        if not files:
            return None, None
        total = len(files)
        existing = sum(1 for f in files if f.status in ("modified", "removed", "renamed"))
        fta = round(existing / total, 3)
        additions = sum(f.additions for f in files)
        deletions = sum(f.deletions for f in files)
        total_ch = additions + deletions
        edit_ratio = round(deletions / total_ch, 3) if total_ch > 0 else 0.0
        return fta, edit_ratio
    except Exception as e:
        logger.warning(f"PR quality fetch failed: {e}")
        return None, None


# ── Main orchestration ────────────────────────────────────────────────────────

def _clamp_swebench_wait_seconds(v: int) -> int:
    return max(120, min(int(v), 6 * 3600))


def run_swebench_eval(
    instances: list[SWEBenchInstance],
    github_token: str,
    workspace: str = "/tmp/phoenix-swebench",
    output_file: str = "eval/results/swebench_results.json",
    sse_url: str = "http://localhost:8000/eval/stream",
    max_wait: int | None = None,
) -> list[SWEBenchResult]:
    """Run Phoenix on SWE-bench instances and evaluate with oracle tests."""
    from eval.runner import EvalSSEListener, wait_for_completion, sw_eval_max_wait_seconds
    from eval.issues import CreatedIssue

    effective_wait = (
        _clamp_swebench_wait_seconds(max_wait)
        if max_wait is not None
        else sw_eval_max_wait_seconds()
    )

    Path(workspace).mkdir(parents=True, exist_ok=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    g = Github(github_token)
    your_username = g.get_user().login
    logger.info(f"Running as: {your_username}")
    logger.info(
        f"Phoenix wait cap per issue: {effective_wait}s "
        f"(SWEBENCH_MAX_WAIT env or max_wait=… / --max-wait)"
    )

    # Pre-register all issues with the SSE listener
    listener = EvalSSEListener(sse_url=sse_url)
    results: list[SWEBenchResult] = []
    total = len(instances)

    listener.start()

    for i, instance in enumerate(instances, 1):
        logger.info(
            f"\n[{i}/{total}] {instance.instance_id}  "
            f"(base: {instance.base_commit[:8]})"
        )
        try:
            # 1. Fork + reset to base commit
            fork_full = prepare_swebench_fork(
                instance, github_token, workspace, your_username
            )

            # 2. Mirror issue to fork
            issue_number, issue_url = mirror_swebench_issue(
                instance, fork_full, github_token
            )

            # 3. Register with SSE listener
            ev = listener.register(fork_full, issue_number)

            # 4. Trigger Phoenix
            gh_fork = g.get_repo(fork_full)
            gh_issue = gh_fork.get_issue(issue_number)
            gh_issue.add_to_labels("ai:ready")
            logger.info(f"  Applied ai:ready → waiting for Phoenix...")

            # 5. Wait for completion
            created = CreatedIssue(
                repo=instance.repo,
                fork=fork_full,
                issue_number=issue_number,
                issue_url=issue_url,
                task_type="swebench",
                title=f"[SWE-bench] {instance.instance_id}",
                label_applied="ai:ready",
            )
            run = wait_for_completion(created, github_token, ev, listener, effective_wait)

            # 6. Oracle evaluation
            oracle_resolved = cp_ok = None
            ftp_passed_n = 0
            ptp_broken_n = 0
            fta = edit_ratio = None
            oracle_ran = False
            oracle_skip_reason: str | None = None

            repo_name = instance.repo.split("/")[1]
            repo_dir = str(Path(workspace) / repo_name)

            if run.status == "succeeded" and run.pr_number:
                pr_branch = f"phoenix/issue-{issue_number}"
                outcome = evaluate_resolution(instance, pr_branch, repo_dir)
                oracle_resolved = outcome.resolved
                cp_ok = outcome.cp
                ftp_passed_n = outcome.fail_to_pass_passed
                ptp_broken_n = outcome.pass_to_pass_broken
                oracle_ran = outcome.skip_reason is None
                oracle_skip_reason = outcome.skip_reason
                fta, edit_ratio = _compute_pr_quality(github_token, fork_full, run.pr_number)
            else:
                if run.status == "timeout":
                    oracle_skip_reason = "phoenix_wait_timeout"
                elif run.status == "failed":
                    oracle_skip_reason = "phoenix_run_failed"
                elif not run.pr_number:
                    oracle_skip_reason = "phoenix_no_pr"
                else:
                    oracle_skip_reason = "phoenix_skipped_oracle"

            # Read Reproducer metrics from the run workspace
            from eval.metrics import _read_run_log
            run_log = _read_run_log(issue_number) or {}
            rp = run_log.get("reproducer_passed")
            if rp is None and "reproducer_passed" not in run_log:
                # Legacy run.json: repro outcome lived only on "resolved" alongside reproduced
                leg = run_log.get("resolved")
                rp = leg if run_log.get("reproduced") and leg is not None else None
            phoenix_ok = run_log.get("issue_resolved", run_log.get("resolved"))

            result = SWEBenchResult(
                instance_id=instance.instance_id,
                repo=instance.repo,
                fork=fork_full,
                issue_number=issue_number,
                issue_url=issue_url,
                pr_number=run.pr_number,
                pr_url=run.pr_url,
                status=run.status,
                resolved_oracle=oracle_resolved,
                cp=cp_ok,
                fail_to_pass_passed=ftp_passed_n,
                fail_to_pass_total=len(instance.fail_to_pass),
                pass_to_pass_broken=ptp_broken_n,
                reproduced=run_log.get("reproduced"),
                phoenix_resolved=phoenix_ok,
                resolved_reproducer=(True if rp is True else (False if rp is False else None)),
                false_positive=run_log.get("false_positive"),
                reproducer_skipped=run_log.get("reproducer_skipped", False),
                reproducer_file=run_log.get("reproducer_file"),
                fta=fta,
                edit_ratio=edit_ratio,
                elapsed_seconds=run.elapsed_seconds,
                wait_cap_seconds=effective_wait,
                wait_timed_out=run.status == "timeout",
                phoenix_final_label=run.final_label or None,
                oracle_ran=oracle_ran,
                oracle_skip_reason=oracle_skip_reason,
            )
            results.append(result)
            _print_result(result)

        except Exception as e:
            logger.error(f"  Exception for {instance.instance_id}: {e}")
            results.append(SWEBenchResult(
                instance_id=instance.instance_id,
                repo=instance.repo,
                fork="",
                issue_number=0,
                issue_url="",
                pr_number=None,
                pr_url=None,
                status="error",
                resolved_oracle=None,
                cp=None,
                fail_to_pass_passed=0,
                fail_to_pass_total=len(instance.fail_to_pass),
                pass_to_pass_broken=0,
                reproduced=None,
                phoenix_resolved=None,
                resolved_reproducer=None,
                false_positive=None,
                reproducer_skipped=False,
                reproducer_file=None,
                fta=None,
                edit_ratio=None,
                elapsed_seconds=0,
                wait_cap_seconds=effective_wait,
                wait_timed_out=False,
                phoenix_final_label=None,
                oracle_ran=False,
                oracle_skip_reason="harness_exception",
            ))

        # Save incrementally
        Path(output_file).write_text(
            json.dumps([asdict(r) for r in results], indent=2)
        )

        if i < total:
            time.sleep(10)

    listener.stop()
    _print_summary(results)
    logger.info(f"\nSWE-bench results saved to {output_file}")
    return results


def _print_result(r: SWEBenchResult) -> None:
    icon = "✅" if r.resolved_oracle is True else ("❌" if r.resolved_oracle is False else "⏱")
    ftp = f"{r.fail_to_pass_passed}/{r.fail_to_pass_total}"
    if r.reproduced is True:
        repro_s = "repro=✓"
    elif r.reproducer_skipped:
        repro_s = "repro=skip"
    elif r.reproduced is False:
        repro_s = "repro=✗"
    else:
        repro_s = "repro=?"
    pol_s = "pol=✓" if r.phoenix_resolved is True else ("pol=✗" if r.phoenix_resolved is False else "pol=?")
    repro_ok = "repro_ok=✓" if r.resolved_reproducer is True else (
        "repro_ok=✗" if r.resolved_reproducer is False else "repro_ok=?"
    )
    fta_s   = f"FTA={r.fta:.0%}" if r.fta is not None else "FTA=?"
    tmo = " TMO" if r.wait_timed_out else ""
    ora = ""
    if not r.oracle_ran and r.oracle_skip_reason:
        ora = f"  ora_skip={r.oracle_skip_reason}"
    print(
        f"  {icon} {r.instance_id:<42} oracle={ftp}  {repro_s}  {pol_s}  {repro_ok}  {fta_s}  "
        f"[{r.elapsed_seconds:.0f}s/{r.wait_cap_seconds}s]{tmo}{ora}"
    )


def _print_summary(results: list[SWEBenchResult]) -> None:
    total = len(results)
    if total == 0:
        print("\n(No SWE-bench results to summarize.)")
        return
    oracle_resolved    = [r for r in results if r.resolved_oracle is True]
    cp_ok              = [r for r in results if r.cp is True]
    reproduced_runs    = [r for r in results if r.reproduced is True]
    phoenix_ok_l       = [r for r in results if r.phoenix_resolved is True]
    repro_resolved     = [r for r in results if r.resolved_reproducer is True]
    false_pos_runs     = [r for r in results if r.false_positive is True]
    skipped_runs       = [r for r in results if r.reproducer_skipped]
    fta_runs           = [r for r in results if r.fta is not None]
    er_runs            = [r for r in results if r.edit_ratio is not None]

    print("\n" + "=" * 65)
    print("SWE-BENCH EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Total instances            : {total}")
    print()
    print(f"  ── Oracle (SWE-bench) ────────────────────────────────")
    print(f"  Resolved (FAIL_TO_PASS)    : {len(oracle_resolved)}/{total} ({len(oracle_resolved)/total:.1%})")
    print(f"  Correctness preserved (CP) : {len(cp_ok)}/{total} ({len(cp_ok)/total:.1%})")
    oracle_ran = [r for r in results if r.oracle_ran]
    oracle_skipped = [r for r in results if not r.oracle_ran]
    wait_timeouts = [r for r in results if r.wait_timed_out]
    print(f"  Oracle tests ran           : {len(oracle_ran)}/{total}")
    if wait_timeouts:
        print(f"  Phoenix wait timeouts      : {len(wait_timeouts)}/{total} (cap={wait_timeouts[0].wait_cap_seconds}s)")
    if oracle_skipped:
        reasons = Counter((r.oracle_skip_reason or "unknown") for r in oracle_skipped)
        print(f"  Oracle skipped (no run)    : {len(oracle_skipped)}/{total} — {dict(reasons)}")
    print()
    print(f"  ── Phoenix resolution (RESOLUTION_MODE) ─────────────")
    print(f"  Policy gate passed         : {len(phoenix_ok_l)}/{total} ({len(phoenix_ok_l)/total:.1%})")
    print(f"  ── Reproduce→Fix→Verify ──────────────────────────────")
    print(f"  Reproduced                 : {len(reproduced_runs)}/{total} ({len(reproduced_runs)/total:.1%})")
    print(f"  Skipped (no failing test)  : {len(skipped_runs)}/{total} ({len(skipped_runs)/total:.1%})")
    if reproduced_runs:
        print(f"  Repro test passes after fix : {len(repro_resolved)}/{len(reproduced_runs)} ({len(repro_resolved)/len(reproduced_runs):.1%})")
    if false_pos_runs:
        print(f"  False positive rate        : {len(false_pos_runs)}/{total} ({len(false_pos_runs)/total:.1%})")
    print()
    print(f"  ── PR Quality ────────────────────────────────────────")
    if fta_runs:
        avg_fta = sum(r.fta for r in fta_runs) / len(fta_runs)
        print(f"  Avg FTA                    : {avg_fta:.1%}  (n={len(fta_runs)})")
    if er_runs:
        avg_er = sum(r.edit_ratio for r in er_runs) / len(er_runs)
        print(f"  Avg edit ratio             : {avg_er:.1%}  (n={len(er_runs)})")
    print("=" * 65)

    # Per-repo breakdown
    repos: dict[str, list[SWEBenchResult]] = {}
    for r in results:
        repos.setdefault(r.repo, []).append(r)
    print("\nPer repo (oracle / Phoenix policy / repro test pass / total):")
    for repo, rs in sorted(repos.items()):
        ok_o = sum(1 for r in rs if r.resolved_oracle is True)
        ok_p = sum(1 for r in rs if r.phoenix_resolved is True)
        ok_r = sum(1 for r in rs if r.resolved_reproducer is True)
        print(f"  {repo:<40} {ok_o}/{len(rs)} oracle   {ok_p}/{len(rs)} policy   {ok_r}/{len(rs)} repro_ok")
