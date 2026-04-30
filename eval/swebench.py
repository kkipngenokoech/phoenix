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

# All repos present in SWE-bench Lite (300 instances).
# Covers the complete benchmark split — no exclusions.
SUPPORTED_REPOS = {
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pylint-dev/pylint",
    "pydata/xarray",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
    # Extended: additional repos present in Lite but not in original pilot
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
    # LLM usage (all agents combined for this issue)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    inference_seconds: float = 0.0


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

    # Clone or update local copy (large repos like sphinx need more time)
    repo_dir = str(Path(workspace) / repo_name)
    if not Path(repo_dir).exists():
        logger.info(f"  Cloning {fork_full}...")
        code, out = _run(
            ["git", "clone", "--depth=50", f"https://x-access-token:{github_token}@github.com/{fork_full}.git", repo_dir],
            timeout=300,
        )
        if code != 0:
            # Retry without --depth in case shallow clone fails
            logger.warning(f"  Shallow clone failed, retrying full clone...")
            code, out = _run(
                ["git", "clone", f"https://x-access-token:{github_token}@github.com/{fork_full}.git", repo_dir],
                timeout=600,
            )
        if code != 0:
            raise RuntimeError(f"Clone failed: {out[:300]}")
    else:
        _run(["git", "fetch", "origin"], cwd=repo_dir, timeout=120)

    # Reset fork to base_commit
    logger.info(f"  Resetting {fork_full} to {instance.base_commit[:8]}...")
    code, _ = _run(["git", "checkout", "main"], cwd=repo_dir)
    if code != 0:
        _run(["git", "checkout", "master"], cwd=repo_dir)

    # Fetch the base commit from upstream (may not be in shallow fork)
    code, out = _run(
        ["git", "fetch", "--unshallow", f"https://github.com/{instance.repo}.git", instance.base_commit],
        cwd=repo_dir, timeout=300,
    )
    if code != 0:
        # Already full-depth or different branch; try plain fetch
        _run(
            ["git", "fetch", f"https://github.com/{instance.repo}.git", instance.base_commit],
            cwd=repo_dir, timeout=120,
        )
    _run(["git", "reset", "--hard", instance.base_commit], cwd=repo_dir)
    # Remove untracked files left by previous runs (prevents oracle contamination)
    _run(["git", "clean", "-fd"], cwd=repo_dir)
    code, out = _run(
        ["git", "push", "origin", "HEAD", "--force"],
        cwd=repo_dir, timeout=120,
    )
    if code != 0:
        logger.warning(f"  Force-push to {fork_full} failed — fork may be out of sync: {out[:200]}")

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

    hints_section = (
        f"\n\n## Hints\n{instance.hints_text.strip()}\n"
        if instance.hints_text and instance.hints_text.strip()
        else ""
    )
    if instance.fail_to_pass:
        test_list = "\n".join(f"- `{t}`" for t in instance.fail_to_pass[:20])
        tests_section = (
            f"\n\n## Tests that must pass after this fix\n"
            f"The following tests are currently failing and must pass once the issue is resolved:\n"
            f"{test_list}\n"
        )
    else:
        tests_section = ""
    body = (
        f"{instance.problem_statement}"
        f"{hints_section}"
        f"{tests_section}\n\n"
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

def _extract_new_files_from_patch(patch_text: str, repo_dir: str) -> list[str]:
    """Parse a unified diff and directly write files missing from the working tree.

    Handles two cases that cause 'does not exist in index' failures:
      1. Patch adds a genuinely new file (--- /dev/null).
      2. Patch modifies a file that exists in the oracle's solution commit but not
         at the SWE-bench base commit (e.g. test_wcs.py was added in the same PR
         that fixed the bug — it doesn't exist at the pinned base commit).

    For case 2 the file path appears in the +++ line but is absent on disk.
    We reconstruct the file by taking all + lines across all hunks, which gives
    us the oracle's complete intended test content.

    Returns list of file paths that were successfully created/written.
    """
    import re
    created: list[str] = []
    root = Path(repo_dir)

    file_sections = re.split(r"(?=^diff --git )", patch_text, flags=re.MULTILINE)
    for section in file_sections:
        if not section.strip():
            continue

        # Extract destination path from +++ b/<path>
        dest_match = re.search(r"^\+\+\+ b/(.+)$", section, re.MULTILINE)
        if not dest_match:
            continue
        dest_path = dest_match.group(1).strip()
        if dest_path == "/dev/null":
            continue  # deletion — nothing to create
        target = root / dest_path

        # Determine if we should handle this section:
        #   (a) explicit new-file patch  OR
        #   (b) file simply doesn't exist on disk at this base commit
        is_new_file = ("--- /dev/null" in section or "new file mode" in section)
        is_missing  = not target.exists()

        if not (is_new_file or is_missing):
            continue

        # Reconstruct the file's after-state from all + lines in all hunks
        lines: list[str] = []
        in_hunk = False
        for line in section.splitlines():
            if line.startswith("@@"):
                in_hunk = True
                continue
            if not in_hunk:
                continue
            if line.startswith("+"):
                lines.append(line[1:])
            elif line.startswith("\\"):
                pass  # "No newline at end of file" marker
            # - lines are context from the old file; for missing files we skip them

        if not lines:
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(lines))
            created.append(dest_path)
            logger.info("  Oracle patch: created new file %s (%d lines)", dest_path, len(lines))
        except Exception as e:
            logger.warning("  Oracle patch: failed to create %s: %s", dest_path, e)

    return created


def _apply_patch(patch_text: str, repo_dir: str) -> bool:
    """Apply a unified diff patch string to the repo. Returns True on success.

    Strategy (in order):
    1. git apply — standard fast path
    2. git apply --3way — handles overlapping lines from Phoenix's changes
    3. Direct file creation — handles new files absent at the base commit
       (oracle test_patch adds test_table.py that doesn't exist in the index)
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

    # Fallback: directly create new files from the patch for hunks that fail
    # because the file doesn't exist in the git index at this base commit.
    created = _extract_new_files_from_patch(patch_text, repo_dir)
    if created:
        # Try git apply again for any remaining hunks (modified existing files)
        r3 = subprocess.run(
            ["git", "apply", "--whitespace=fix", "--ignore-missing-newline", "-"],
            input=patch_text,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r3.returncode == 0 or created:
            logger.info(
                "  Oracle patch: direct-created %d new file(s); remaining hunks %s",
                len(created),
                "applied" if r3.returncode == 0 else "skipped (already handled)",
            )
            return True

    return False


def _get_or_create_oracle_venv(workspace: str) -> tuple[str, str]:
    """Return (python_bin, pip_bin) for a dedicated oracle venv.

    This venv is isolated from the Phoenix execution environment, so processing
    pytest-dev/pytest or any other repo never clobbers the oracle's pytest.
    Created once per workspace; reused across instances.
    """
    import sys
    venv_path = Path(workspace) / ".oracle-venv"
    python_bin = str(venv_path / "bin" / "python")
    pip_bin = str(venv_path / "bin" / "pip")

    # Lightweight packages always needed by repo conftest.py files.
    # Installed separately so they don't get blocked by numpy/scipy version
    # conflicts when we later run pip install -e ".[dev,test]" for old repos.
    _CONFTEST_DEPS = [
        "pytest",
        "hypothesis",        # astropy conftest.py: import hypothesis
        "setuptools_scm",    # astropy _dev/scm_version.py via pytest warning config
        "pytest-mock",
        "pytest-xdist",
        "pytest-timeout",
        "legacy-cgi",        # cgi module removed in Python 3.13+; required by Django 3.x/4.x
    ]

    if not venv_path.exists():
        logger.info("  Oracle: creating isolated venv at %s", venv_path)
        # --system-site-packages: inherit compiled packages from the Phoenix venv
        # (astropy, numpy, scipy, etc. that can't compile from old source on Python 3.14).
        # Oracle installs its OWN pytest so it can never be clobbered by Phoenix's
        # pip install -e . operations (local venv packages shadow system-site-packages).
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv_path)],
            check=True, timeout=60,
        )
        subprocess.run(
            [pip_bin, "install", "--quiet"] + _CONFTEST_DEPS,
            check=True, timeout=240,
        )
        logger.info("  Oracle: venv ready (system-site-packages enabled)")
    else:
        # Verify pytest is healthy in the oracle venv (defensive check)
        ver = subprocess.run([python_bin, "-m", "pytest", "--version"],
                             capture_output=True, text=True, timeout=10)
        if ver.returncode != 0 or "dev" in ver.stdout.lower():
            logger.warning("  Oracle venv pytest broken (%s) — reinstalling", ver.stdout.strip())
            subprocess.run([pip_bin, "install", "--quiet", "--upgrade"] + _CONFTEST_DEPS,
                           capture_output=True, timeout=240)

    return python_bin, pip_bin


def _detect_test_id_format(test_ids: list[str]) -> str:
    """Return the test ID format used by this instance.

    Formats:
      'django'       — 'method_name (dotted.module.ClassName)'  → runtests.py
      'bare_name'    — 'test_function_name'                     → pytest -k
      'pytest_nodeid'— 'path/to/test.py::Class::method'        → pytest ids
    """
    if not test_ids:
        return "pytest_nodeid"
    first = test_ids[0]
    if "(" in first:
        return "django"
    if "::" in first or "/" in first:
        return "pytest_nodeid"
    return "bare_name"


def _is_django_test_format(test_ids: list[str]) -> bool:
    """Detect Django's native 'method_name (dotted.module.ClassName)' test ID format."""
    return _detect_test_id_format(test_ids) == "django"


def _convert_django_test_ids(test_ids: list[str]) -> list[str]:
    """Convert 'method (module.Class)' → deduplicated 'module.Class' selectors for runtests.py.

    Django's runtests.py only supports module-level or class-level selectors, not
    method-level 4-part paths. We deduplicate so each class only runs once.
    """
    import re as _re
    seen: set[str] = set()
    out = []
    for tid in test_ids:
        m = _re.match(r"^\w+ \(([\w.]+)\)$", tid)
        selector = m.group(1) if m else tid
        if selector not in seen:
            seen.add(selector)
            out.append(selector)
    return out


def _run_django_tests(
    test_ids: list[str],
    repo_dir: str,
    oracle_python: str,
    timeout: int = 300,
) -> tuple[set[str], set[str]]:
    """Run Django-format test IDs via tests/runtests.py. Returns (passed, failed)."""
    import re as _re
    selectors = _convert_django_test_ids(test_ids)
    tests_dir = str(Path(repo_dir) / "tests")
    cmd = [oracle_python, "runtests.py", "--settings=test_sqlite",
           "--verbosity=2", "--parallel=1"] + selectors
    result = subprocess.run(
        cmd, cwd=tests_dir, capture_output=True, text=True, timeout=timeout,
    )
    output = result.stdout + result.stderr

    if result.returncode == 0:
        return set(test_ids), set()

    # Parse per-test verdict from verbosity=2 output.
    # Python < 3.13 format: "test_method (module.Class) ... ok"
    # Python 3.14+ format:  "test_method (module.Class.test_method) ... ok"
    # Normalise both to "test_method (module.Class)" to match SWE-bench test IDs.
    test_ids_set = set(test_ids)
    failed: set[str] = set()
    verdict_pat = _re.compile(r"^(.+?) \.\.\. (ok|FAIL|ERROR|skip.*|expected failure)$", _re.IGNORECASE)
    for line in output.splitlines():
        m = verdict_pat.match(line.strip())
        if m:
            label_part = m.group(1).strip()
            verdict = m.group(2).lower()
            # Resolve label_part → SWE-bench test ID
            if label_part in test_ids_set:
                key = label_part
            else:
                # Python 3.14: "test_method (module.Class.test_method)" → "test_method (module.Class)"
                m2 = _re.match(r"^(\w+) \(([\w.]+)\.\w+\)$", label_part)
                if m2:
                    candidate = f"{m2.group(1)} ({m2.group(2)})"
                    key = candidate if candidate in test_ids_set else None
                else:
                    key = None
            if key and verdict in ("fail", "error"):
                failed.add(key)

    if failed:
        return test_ids_set - failed, failed

    # Exit non-0 but no parseable verdicts → treat all as failed
    logger.warning("  Oracle Django tests: exit %d, no parsed verdicts\n%s",
                   result.returncode, output[:4000])
    return set(), set(test_ids)


def _run_bare_name_tests(
    test_ids: list[str],
    repo_dir: str,
    oracle_python: str,
    timeout: int = 300,
) -> tuple[set[str], set[str]]:
    """Run bare function-name test IDs (sympy format) via pytest -k. Returns (passed, failed)."""
    import re as _re
    k_expr = " or ".join(test_ids[:50])
    cmd = [oracle_python, "-m", "pytest", "--tb=line", "-k", k_expr]
    result = subprocess.run(
        cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
    )
    output = result.stdout + result.stderr

    if result.returncode == 0:
        return set(test_ids), set()

    if result.returncode == 1:
        failed: set[str] = set()
        for line in output.splitlines():
            m = _re.match(r"^FAILED (.+?) -", line)
            if m:
                path_part = m.group(1).strip()  # e.g. "sympy/core/tests/test_expr.py::test_foo"
                for tid in test_ids:
                    if path_part.endswith(f"::{tid}"):
                        failed.add(tid)
        if failed:
            return set(test_ids) - failed, failed
        logger.warning("  Oracle bare-name tests: exit 1, no FAILED lines\n%s", output[:600])
        return set(), set(test_ids)

    logger.warning("  Oracle bare-name tests: exit %d\n%s", result.returncode, output[:800])
    return set(), set()


def _run_specific_tests(
    test_ids: list[str],
    repo_dir: str,
    timeout: int = 300,
    oracle_python: str = "python",
    oracle_pip: str = "pip",
) -> tuple[set[str], set[str]]:
    """Run a specific list of test IDs. Returns (passed, failed) sets.

    Handles both pytest node IDs (most repos) and Django's native
    'method (module.ClassName)' format automatically.

    oracle_python / oracle_pip: binaries from the isolated oracle venv so that
    Phoenix runs against other repos cannot clobber the oracle's pytest.
    """
    if not test_ids:
        return set(), set()

    root = Path(repo_dir)

    # Detect C-extension packages by presence of Cython source files.
    # For these repos (astropy, matplotlib, scikit-learn, …) we MUST NOT run
    # pip install -e . because:
    #   1. Old base-commit C extensions often fail to compile on newer Pythons.
    #   2. Even a failed editable install creates a .pth / direct_url.json
    #      registration that OVERRIDES system-site-packages, then Python finds
    #      the clone directory's astropy/__init__.py but can't load compiled
    #      extensions → ImportError → exit 4.
    # System-site-packages (Phoenix's venv, built via make install) already has
    # working compiled versions of these packages.
    has_cython = bool(list(root.rglob("*.pyx"))[:1])

    if has_cython:
        logger.info(
            "  Oracle: C-extension repo detected (%s) — skipping editable install, "
            "relying on system-site-packages for compiled extensions",
            root.name,
        )
        # Safety: remove any leftover editable registration from a previous run
        # that might shadow system-site-packages.
        subprocess.run(
            [oracle_pip, "uninstall", "-y", root.name],
            capture_output=True, timeout=30,
        )
    else:
        # Pure-Python repo: safe to install from source at the exact base commit.
        # Try progressively simpler extras so we always get at least a plain install.
        for spec in (".[dev,test]", ".[test]", ".[dev]", "."):
            r = subprocess.run(
                [oracle_pip, "install", "-e", spec, "--quiet", "--no-build-isolation"],
                cwd=repo_dir, capture_output=True, timeout=240,
            )
            if r.returncode == 0:
                break

        # Requirements files for test-only deps not captured by package extras.
        for req_file in ("requirements-dev.txt", "requirements-test.txt", "requirements-testing.txt"):
            req_path = root / req_file
            if req_path.exists():
                subprocess.run(
                    [oracle_pip, "install", "-r", str(req_path), "--quiet"],
                    cwd=repo_dir, capture_output=True, timeout=240,
                )
                break

    # Re-install lightweight conftest deps — old pip install -e ".[dev,test]"
    # constraints can downgrade hypothesis / setuptools_scm that we installed.
    subprocess.run(
        [oracle_pip, "install", "--quiet", "--upgrade",
         "hypothesis", "setuptools_scm", "pytest-mock", "pytest-xdist", "pytest-timeout"],
        capture_output=True, timeout=120,
    )

    # Guard: pip install of pytest-dev/pytest replaces oracle venv's pytest
    # with a dev build that fails minversion checks.
    try:
        ver = subprocess.run(
            [oracle_python, "-m", "pytest", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if "dev" in ver.stdout.lower() or ver.returncode != 0:
            logger.warning(
                "  Oracle venv: pip install clobbered pytest (%s) — restoring stable version",
                ver.stdout.strip(),
            )
            subprocess.run(
                [oracle_pip, "install", "--quiet", "--upgrade", "pytest"],
                capture_output=True, timeout=120,
            )
    except Exception:
        pass

    # Dispatch based on test ID format — each repo family uses a different convention.
    fmt = _detect_test_id_format(test_ids)
    if fmt == "django":
        return _run_django_tests(test_ids, repo_dir, oracle_python, timeout=timeout)
    if fmt == "bare_name":
        return _run_bare_name_tests(test_ids, repo_dir, oracle_python, timeout=timeout)

    # --tb=short is supported by pytest 5+ and produces parseable FAILED lines.
    cmd = [oracle_python, "-m", "pytest", "--tb=short"] + test_ids[:50]
    result = subprocess.run(
        cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout,
    )
    output = result.stdout + result.stderr

    import re as _re

    if result.returncode == 0:
        return set(test_ids), set()

    if result.returncode == 1:
        failed: set[str] = set()
        for line in output.splitlines():
            m = _re.match(r"^FAILED (.+?) -", line)
            if m:
                failed.add(m.group(1).strip())
        if failed:
            return set(test_ids) - failed, failed
        # Exit code 1 but no FAILED lines = collection/import error
        logger.warning("  Oracle tests: exit 1, no FAILED lines (collection error?)\n%s", output[:600])
        return set(), set(test_ids)

    # Exit codes 2–5: interrupted / internal error / usage error / no tests.
    # "Did not run" → neither passed nor failed (don't penalise cp).
    logger.warning("  Oracle tests: exit %d — tests did not run\n%s",
                   result.returncode, output[:1200])
    return set(), set()


def evaluate_resolution(
    instance: SWEBenchInstance,
    pr_branch: str,
    repo_dir: str,
    oracle_python: str = "python",
    oracle_pip: str = "pip",
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
    ftp_passed, ftp_failed = _run_specific_tests(
        instance.fail_to_pass, repo_dir,
        oracle_python=oracle_python, oracle_pip=oracle_pip,
    )
    ftp_pass_count = len(ftp_passed)
    resolved = (ftp_pass_count == ftp_total) if ftp_total > 0 else None

    # Evaluate PASS_TO_PASS
    ptp_passed, ptp_failed = _run_specific_tests(
        instance.pass_to_pass[:30], repo_dir,
        oracle_python=oracle_python, oracle_pip=oracle_pip,
    )
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


def _load_completed_instance_ids(output_file: str) -> set[str]:
    """Return instance_ids that already have a result in output_file (for resume)."""
    p = Path(output_file)
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        return {r["instance_id"] for r in data if isinstance(r, dict) and "instance_id" in r}
    except Exception:
        return set()


def _trigger_phoenix_direct(fork: str, issue_number: int, trigger_url: str) -> None:
    """POST to Phoenix /eval/trigger to dispatch a run without GitHub App webhooks."""
    import urllib.request as _urllib_req
    payload = json.dumps({"repo": fork, "issue_number": issue_number}).encode()
    req = _urllib_req.Request(
        trigger_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urllib_req.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            run_id = result.get("run_id", "N/A")
            logger.info(f"  Phoenix trigger: {result.get('status')} run={run_id[:8] if run_id != 'N/A' else 'N/A'}")
    except Exception as e:
        logger.warning(f"  Phoenix direct trigger failed: {e} — label-based fallback only")


def run_swebench_eval(
    instances: list[SWEBenchInstance],
    github_token: str,
    workspace: str = "/tmp/phoenix-swebench",
    output_file: str = "eval/results/swebench_results.json",
    sse_url: str = "http://localhost:8000/eval/stream",
    max_wait: int | None = None,
    workers: int = 1,
    resume: bool = True,
) -> list[SWEBenchResult]:
    """Run Phoenix on SWE-bench instances and evaluate with oracle tests.

    Args:
        workers: Number of parallel Phoenix instances to run. Each worker
                 needs its own clone workspace to avoid git conflicts.
                 Set to 1 for sequential (safe default). 2-4 works well
                 on a machine with enough RAM.
        resume:  Skip instances whose instance_id is already in output_file.
                 Enables safe restart after interruption.
    """
    from eval.runner import EvalSSEListener, wait_for_completion, sw_eval_max_wait_seconds
    from eval.issues import CreatedIssue

    effective_wait = (
        _clamp_swebench_wait_seconds(max_wait)
        if max_wait is not None
        else sw_eval_max_wait_seconds()
    )

    Path(workspace).mkdir(parents=True, exist_ok=True)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Create (or verify) the isolated oracle venv once for the whole eval run.
    # This venv is never touched by Phoenix's orchestrator, so pip install -e .
    # inside a pytest-dev/pytest repo cannot clobber the oracle's pytest binary.
    oracle_python, oracle_pip = _get_or_create_oracle_venv(workspace)

    # Resume: skip already-completed instances
    if resume:
        completed = _load_completed_instance_ids(output_file)
        if completed:
            before = len(instances)
            instances = [i for i in instances if i.instance_id not in completed]
            logger.info(f"Resume: skipping {before - len(instances)} completed instance(s), {len(instances)} remaining")

    if workers > 1:
        logger.warning(
            "workers=%d requested, but parallel execution is not yet implemented. "
            "Running sequentially (workers=1). To parallelize, run multiple Phoenix server "
            "instances on separate ports with separate --workspace dirs and split the instance "
            "list with --only or --repos per process.",
            workers,
        )

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
            logger.info(f"  Applied ai:ready → triggering Phoenix directly...")
            trigger_url = sse_url.replace("/eval/stream", "/eval/trigger")
            _trigger_phoenix_direct(fork_full, issue_number, trigger_url)

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
                outcome = evaluate_resolution(
                    instance, pr_branch, repo_dir,
                    oracle_python=oracle_python, oracle_pip=oracle_pip,
                )
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
                llm_calls=run_log.get("llm_calls", 0),
                input_tokens=run_log.get("input_tokens", 0),
                output_tokens=run_log.get("output_tokens", 0),
                inference_seconds=run_log.get("inference_seconds", 0.0),
            )
            results.append(result)
            _print_result(result)

        except Exception as e:
            import traceback as _tb
            err_detail = str(e)[:300]
            logger.error(f"  Exception for {instance.instance_id}: {err_detail}")
            logger.debug(_tb.format_exc())
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
                oracle_skip_reason=f"harness_exception: {err_detail}",
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
    tok_s = f"  tok={r.input_tokens}+{r.output_tokens}" if r.input_tokens or r.output_tokens else ""
    inf_s = f"  inf={r.inference_seconds:.0f}s" if r.inference_seconds else ""
    print(
        f"  {icon} {r.instance_id:<42} oracle={ftp}  {repro_s}  {pol_s}  {repro_ok}  {fta_s}  "
        f"[{r.elapsed_seconds:.0f}s/{r.wait_cap_seconds}s]{tmo}{tok_s}{inf_s}{ora}"
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
    print()
    print(f"  ── LLM Usage ─────────────────────────────────────────")
    tok_runs = [r for r in results if r.input_tokens or r.output_tokens]
    if tok_runs:
        total_in  = sum(r.input_tokens for r in tok_runs)
        total_out = sum(r.output_tokens for r in tok_runs)
        total_tok = total_in + total_out
        avg_in    = total_in  / len(tok_runs)
        avg_out   = total_out / len(tok_runs)
        avg_calls = sum(r.llm_calls for r in tok_runs) / len(tok_runs)
        avg_inf   = sum(r.inference_seconds for r in tok_runs) / len(tok_runs)
        total_inf = sum(r.inference_seconds for r in tok_runs)
        print(f"  Instances with usage data  : {len(tok_runs)}/{total}")
        print(f"  Total tokens               : {total_tok:,}  (in={total_in:,}  out={total_out:,})")
        print(f"  Avg tokens / issue         : {avg_in + avg_out:,.0f}  (in={avg_in:,.0f}  out={avg_out:,.0f})")
        print(f"  Avg LLM calls / issue      : {avg_calls:.1f}")
        print(f"  Avg inference time / issue : {avg_inf:.0f}s")
        print(f"  Total inference time       : {total_inf/60:.1f} min")
    else:
        print(f"  No usage data recorded (run with instrumented build)")
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
