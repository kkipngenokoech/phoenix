"""Baseline measurement: fork repo, run tests, measure complexity."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from github import Github

from eval.repos import EvalRepo

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    repo: str
    fork: str
    tests_pass: bool
    test_output: str
    complexity_avg: float          # average cyclomatic complexity (radon / eslint)
    complexity_max: float          # max cyclomatic complexity
    file_count: int
    loc: int                       # lines of code
    measured_at: str


def _run(cmd: list[str], cwd: str, timeout: int = 300) -> tuple[int, str]:
    """Run a shell command and return (returncode, combined output)."""
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout + result.stderr


def _measure_python_complexity(repo_dir: str) -> tuple[float, float]:
    """Use radon to measure cyclomatic complexity. Returns (avg, max)."""
    code, out = _run(
        ["python", "-m", "radon", "cc", ".", "-s", "-j"],
        cwd=repo_dir,
        timeout=120,
    )
    if code != 0 or not out.strip():
        return 0.0, 0.0
    try:
        data = json.loads(out)
        scores: list[float] = []
        for blocks in data.values():
            if isinstance(blocks, list):
                for block in blocks:
                    if isinstance(block, dict) and "complexity" in block:
                        scores.append(block["complexity"])
        if not scores:
            return 0.0, 0.0
        return round(sum(scores) / len(scores), 2), round(max(scores), 2)
    except (json.JSONDecodeError, KeyError, TypeError):
        return 0.0, 0.0


def _measure_js_complexity(repo_dir: str) -> tuple[float, float]:
    """Use eslint complexity rule to measure JS/TS complexity."""
    has_ts = any(Path(repo_dir).rglob("*.ts"))
    eslint_config: dict = {
        "rules": {"complexity": ["warn", 1]},
        "env": {"browser": True, "es2021": True},
    }
    if has_ts:
        eslint_config["parser"] = "@typescript-eslint/parser"

    config_path = Path(repo_dir) / ".eslint_eval.json"
    config_path.write_text(json.dumps(eslint_config))

    code, out = _run(
        ["npx", "eslint", ".", "--ext", ".js,.ts,.jsx,.tsx",
         "-f", "json", "--no-eslintrc", "-c", ".eslint_eval.json"],
        cwd=repo_dir,
        timeout=180,
    )
    config_path.unlink(missing_ok=True)

    try:
        results = json.loads(out)
        scores: list[int] = []
        for file_result in results:
            for msg in file_result.get("messages", []):
                if msg.get("ruleId") == "complexity":
                    m = re.search(r"complexity (\d+)", msg.get("message", ""))
                    if m:
                        scores.append(int(m.group(1)))
        if not scores:
            return 0.0, 0.0
        return round(sum(scores) / len(scores), 2), round(max(scores), 2)
    except (json.JSONDecodeError, KeyError):
        return 0.0, 0.0


def _measure_java_complexity(repo_dir: str) -> tuple[float, float]:
    """Placeholder — use PMD or checkstyle for Java complexity."""
    code, out = _run(
        ["mvn", "-q", "pmd:pmd", "-Dpmd.failOnViolation=false"],
        cwd=repo_dir,
        timeout=300,
    )
    # TODO: parse PMD XML report from target/pmd.xml
    return 0.0, 0.0


def _count_loc(repo_dir: str) -> tuple[int, int]:
    """Count lines of code and files using cloc if available, else file count."""
    try:
        code, out = _run(["cloc", ".", "--json", "--quiet"], cwd=repo_dir, timeout=60)
    except FileNotFoundError:
        code, out = 1, ""
    if code == 0:
        try:
            data = json.loads(out)
            total = data.get("SUM", {})
            return total.get("code", 0), total.get("nFiles", 0)
        except json.JSONDecodeError:
            pass
    # cloc not installed — count source files directly
    root = Path(repo_dir)
    exts = {"*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java"}
    files = [f for ext in exts for f in root.rglob(ext)
             if ".git" not in f.parts and "node_modules" not in f.parts]
    loc = sum(len(f.read_text(errors="ignore").splitlines()) for f in files)
    return loc, len(files)


def _run_tests(repo: EvalRepo, repo_dir: str) -> tuple[bool, str]:
    """Run the test suite and return (passed, output)."""
    if repo.profile == "python":
        code, out = _run(
            ["python", "-m", "pytest", "--tb=no", "-q", "--no-header"],
            cwd=repo_dir,
            timeout=300,
        )
        passed = code in (0, 5)  # 5 = no tests collected
        return passed, out[:4000]

    elif repo.profile == "frontend":
        # Try npm test
        code, out = _run(["npm", "test", "--", "--watchAll=false", "--passWithNoTests"],
                         cwd=repo_dir, timeout=300)
        if code == 127:  # npm not found or no test script
            code, out = _run(["npm", "run", "build"], cwd=repo_dir, timeout=300)
        return code == 0, out[:4000]

    elif repo.profile == "java":
        code, out = _run(["mvn", "-q", "test", "-Dsurefire.failIfNoSpecifiedTests=false"],
                         cwd=repo_dir, timeout=600)
        return code == 0, out[:4000]

    return False, "Unknown profile"


def fork_repo(repo: EvalRepo, github_token: str, your_username: str) -> str:
    """Fork the repo to your account if not already forked. Returns fork full_name."""
    g = Github(github_token)
    user = g.get_user()
    try:
        fork = user.get_repo(repo.fork_name)
        logger.info(f"Fork already exists: {fork.full_name}")
        _enable_issues(fork)
        return fork.full_name
    except Exception:
        pass

    upstream = g.get_repo(repo.full_name)
    try:
        fork = user.create_fork(upstream)
    except Exception as e:
        if "403" in str(e):
            raise RuntimeError(
                f"Cannot fork {repo.full_name}: 403 Forbidden.\n"
                "Your GITHUB_TOKEN needs a classic PAT with 'repo' scope.\n"
                "Go to: GitHub → Settings → Developer settings → "
                "Personal access tokens → Tokens (classic) → Generate new token → check 'repo'."
            ) from e
        raise
    logger.info(f"Forked {repo.full_name} → {fork.full_name}")
    time.sleep(5)  # GitHub needs a moment to set up the fork
    _enable_issues(fork)
    return fork.full_name


def _enable_issues(fork) -> None:
    """Enable Issues on a fork (GitHub disables them by default)."""
    try:
        if not fork.has_issues:
            fork.edit(has_issues=True)
            logger.info(f"Enabled Issues on {fork.full_name}")
    except Exception as e:
        logger.warning(f"Could not enable Issues on {fork.full_name}: {e}")


def ensure_labels(fork_full_name: str, github_token: str) -> None:
    """Create Phoenix labels on the fork if missing."""
    g = Github(github_token)
    repo = g.get_repo(fork_full_name)
    label_specs = [
        ("ai:ready",       "0075ca", "Phoenix AI: ready to process"),
        ("ai:in-progress", "e4e669", "Phoenix AI: currently processing"),
        ("ai:review",      "0e8a16", "Phoenix AI: PR ready for review"),
        ("ai:revise",      "d93f0b", "Phoenix AI: needs revision"),
        ("ai:failed",      "b60205", "Phoenix AI: run failed"),
        ("ai:done",        "006b75", "Phoenix AI: completed"),
    ]
    existing = {l.name for l in repo.get_labels()}
    for name, color, description in label_specs:
        if name not in existing:
            repo.create_label(name=name, color=color, description=description)
            logger.info(f"Created label '{name}' on {fork_full_name}")


def measure_baseline(repo: EvalRepo, github_token: str, workspace: str) -> BaselineResult:
    """Clone the fork, run tests, measure complexity, return BaselineResult."""
    g = Github(github_token)
    your_username = g.get_user().login
    fork_name = fork_repo(repo, github_token, your_username)
    ensure_labels(fork_name, github_token)

    clone_url = f"https://{github_token}@github.com/{fork_name}.git"
    clone_dir = Path(workspace) / repo.name

    if not clone_dir.exists():
        logger.info(f"Cloning {fork_name}...")
        subprocess.run(["git", "clone", "--depth=1", clone_url, str(clone_dir)],
                       check=True, capture_output=True)
    else:
        logger.info(f"Using existing clone at {clone_dir}")

    repo_dir = str(clone_dir)

    # Install dependencies
    if repo.profile == "python":
        _run(["pip", "install", "-e", ".[dev,test]", "-q"], cwd=repo_dir, timeout=300)
        _run(["pip", "install", "radon", "-q"], cwd=repo_dir, timeout=60)
    elif repo.profile == "frontend":
        _run(["npm", "install", "--silent"], cwd=repo_dir, timeout=300)

    tests_pass, test_output = _run_tests(repo, repo_dir)
    logger.info(f"{repo.full_name}: tests={'PASS' if tests_pass else 'FAIL'}")

    if repo.profile == "python":
        avg_cc, max_cc = _measure_python_complexity(repo_dir)
    elif repo.profile == "frontend":
        avg_cc, max_cc = _measure_js_complexity(repo_dir)
    elif repo.profile == "java":
        avg_cc, max_cc = _measure_java_complexity(repo_dir)
    else:
        avg_cc, max_cc = 0.0, 0.0

    loc, file_count = _count_loc(repo_dir)

    return BaselineResult(
        repo=repo.full_name,
        fork=fork_name,
        tests_pass=tests_pass,
        test_output=test_output,
        complexity_avg=avg_cc,
        complexity_max=max_cc,
        file_count=file_count,
        loc=loc,
        measured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def run_baselines(repos: list[EvalRepo], github_token: str, workspace: str,
                  output_file: str = "eval/results/baselines.json") -> list[BaselineResult]:
    """Measure baselines for a list of repos and save to JSON."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    results: list[BaselineResult] = []

    for repo in repos:
        logger.info(f"\n{'='*60}\nBaseline: {repo.full_name} [{repo.level}]\n{'='*60}")
        try:
            result = measure_baseline(repo, github_token, workspace)
            results.append(result)
            logger.info(f"  CC avg={result.complexity_avg} max={result.complexity_max} "
                        f"LOC={result.loc} files={result.file_count}")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    Path(output_file).write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )
    logger.info(f"\nBaselines saved to {output_file}")
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    token = os.getenv("GITHUB_TOKEN", "")
    workspace = os.getenv("EVAL_WORKSPACE", "/tmp/phoenix-eval")
    tier = sys.argv[1] if len(sys.argv) > 1 else "pilot"

    from eval.repos import PILOT_REPOS, TIER1_REPOS, TIER2_REPOS
    repo_set = {"pilot": PILOT_REPOS, "tier1": TIER1_REPOS, "tier2": TIER2_REPOS}.get(tier, PILOT_REPOS)

    Path(workspace).mkdir(parents=True, exist_ok=True)
    run_baselines(repo_set, token, workspace)
