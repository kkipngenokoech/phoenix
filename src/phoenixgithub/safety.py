"""Safety guards applied to Coder output before changes land on disk or in a PR.

Each guard is stateless and returns a SafetyViolation (or None if clean).
The Orchestrator calls run_all_guards() after the Coder runs; any violation
triggers a retry with targeted feedback rather than silently shipping bad code.

Guards implemented here:
  1. BlastRadiusGuard    — rejects changes touching too many files or lines
  2. SecurityScanGuard   — detects dangerous code patterns in generated content
  3. DependencyGuard     — flags new imports absent from the project's manifest
  4. SensitivePathGuard  — extends workflow blocking to other sensitive paths
  5. SecretsGuard        — detects hardcoded credentials / tokens in generated code
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class SafetyViolation:
    guard: str
    message: str
    details: list[str] = field(default_factory=list)
    severity: str = "error"  # "error" blocks the run; "warning" logs only

    def feedback(self) -> str:
        lines = [f"SAFETY VIOLATION [{self.guard}]: {self.message}"]
        lines.extend(f"  - {d}" for d in self.details[:5])
        return "\n".join(lines)


# ── Guard 1: Blast radius ─────────────────────────────────────────────────────

MAX_FILES_CHANGED = 10
MAX_LINES_ADDED = 600
MAX_LINES_TOTAL = 1000


def blast_radius_guard(changes: list[dict[str, Any]]) -> SafetyViolation | None:
    """Reject changes that touch too many files or add too many lines.

    A legitimate bug fix rarely needs to change 10+ files or add 600+ lines.
    Runaway generation that rewrites the whole codebase is caught here.
    """
    n_files = len(changes)
    n_added = sum(len((c.get("content") or c.get("replace") or "").splitlines()) for c in changes)
    n_total = n_added  # approximation; deletions aren't tracked in the Coder schema

    details: list[str] = []
    if n_files > MAX_FILES_CHANGED:
        details.append(f"Changed {n_files} files (limit {MAX_FILES_CHANGED}). Narrow your changes to the specific files listed in the plan.")
    if n_added > MAX_LINES_ADDED:
        details.append(f"Added ~{n_added} lines (limit {MAX_LINES_ADDED}). Make surgical edits to existing files instead of rewriting them.")

    if details:
        logger.warning("BlastRadiusGuard: %s", "; ".join(details))
        return SafetyViolation(
            guard="BlastRadiusGuard",
            message=f"Change set too large ({n_files} files, ~{n_added} lines added).",
            details=details,
        )
    return None


# ── Guard 2: Security pattern scanner ────────────────────────────────────────

_DANGEROUS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bos\.system\s*\(", re.I), "os.system() call — use subprocess with explicit args instead"),
    (re.compile(r"\beval\s*\(", re.I), "eval() call — never use eval() on untrusted or generated input"),
    (re.compile(r"\bexec\s*\(", re.I), "exec() call — exec() with dynamic content is a code injection risk"),
    (re.compile(r"\bpickle\.loads?\s*\(", re.I), "pickle.load() on untrusted data enables arbitrary code execution"),
    (re.compile(r"\b__import__\s*\(", re.I), "__import__() dynamic import — prefer explicit imports"),
    (re.compile(r"subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True", re.I), "subprocess with shell=True is a command injection risk"),
    (re.compile(r"\.format\s*\([^)]*request\.", re.I), "SQL/template injection risk: user request data used in .format()"),
    (re.compile(r"f['\"].*\{request\.", re.I), "f-string with request data — potential injection"),
]


def security_scan_guard(changes: list[dict[str, Any]]) -> SafetyViolation | None:
    """Scan generated code for dangerous patterns.

    Only fires on newly created or replaced content — not on the search
    (old) side of patch operations.
    """
    hits: list[str] = []
    for change in changes:
        content = change.get("content") or change.get("replace") or ""
        if not content:
            continue
        fp = change.get("file_path", "?")
        for pattern, desc in _DANGEROUS_PATTERNS:
            if pattern.search(content):
                hits.append(f"{fp}: {desc}")

    if hits:
        logger.warning("SecurityScanGuard: %d dangerous pattern(s) found", len(hits))
        return SafetyViolation(
            guard="SecurityScanGuard",
            message=f"Generated code contains {len(hits)} dangerous pattern(s).",
            details=hits,
            severity="warning",  # Log but don't block — reviewer will see it
        )
    return None


# ── Guard 3: Dependency guard ─────────────────────────────────────────────────

_IMPORT_RE = re.compile(r"^(?:import|from)\s+([a-zA-Z0-9_]+)", re.MULTILINE)

# stdlib + common test packages that are always safe
_STDLIB_AND_COMMON = {
    "os", "sys", "re", "json", "time", "math", "copy", "io", "abc",
    "collections", "itertools", "functools", "pathlib", "typing",
    "dataclasses", "contextlib", "threading", "logging", "warnings",
    "unittest", "pytest", "mock", "string", "struct", "hashlib",
    "datetime", "calendar", "random", "decimal", "fractions",
    "subprocess", "shutil", "tempfile", "glob", "fnmatch",
    "http", "urllib", "email", "html", "xml", "csv", "sqlite3",
    "ast", "inspect", "importlib", "pkgutil", "traceback", "pprint",
    "textwrap", "enum", "weakref", "gc", "operator", "heapq",
}


def _read_declared_deps(clone_path: str) -> set[str]:
    """Return package names declared in requirements.txt / pyproject.toml / setup.py."""
    root = Path(clone_path)
    names: set[str] = set()

    req_files = list(root.glob("requirements*.txt")) + list(root.glob("requirements/*.txt"))
    for rf in req_files:
        try:
            for line in rf.read_text(errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg = re.split(r"[>=<!;\[]", line)[0].strip().replace("-", "_").lower()
                    if pkg:
                        names.add(pkg)
        except Exception:
            pass

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            text = pyproject.read_text(errors="replace")
            for m in re.finditer(r'"([a-zA-Z0-9_\-]+)\s*[>=<!;]', text):
                names.add(m.group(1).replace("-", "_").lower())
        except Exception:
            pass

    return names


def dependency_guard(
    changes: list[dict[str, Any]],
    clone_path: str,
) -> SafetyViolation | None:
    """Flag new imports that aren't in the project's declared dependencies.

    This catches cases where the Coder hallucinates imports of packages
    that aren't installed in the test environment (causing import errors)
    or that are suspicious (unknown packages).
    """
    declared = _read_declared_deps(clone_path)
    unknown_imports: list[str] = []

    for change in changes:
        if change.get("action") not in ("create", "modify", None):
            continue
        content = change.get("content") or change.get("replace") or ""
        fp = change.get("file_path", "?")
        for m in _IMPORT_RE.finditer(content):
            pkg = m.group(1).lower().replace("-", "_")
            if pkg in _STDLIB_AND_COMMON:
                continue
            if pkg in declared:
                continue
            # Allow project-internal imports (same name as a local package)
            unknown_imports.append(f"{fp}: import {pkg!r} (not in project deps)")

    if unknown_imports:
        logger.info("DependencyGuard: %d unknown import(s) — may cause ImportError", len(unknown_imports))
        return SafetyViolation(
            guard="DependencyGuard",
            message=f"{len(unknown_imports)} import(s) not found in project dependencies.",
            details=unknown_imports[:8],
            severity="warning",
        )
    return None


# ── Guard 4: Sensitive path guard ─────────────────────────────────────────────

_SENSITIVE_PATH_PATTERNS = [
    (".github/", "GitHub Actions / workflow files — Phoenix cannot push to .github/"),
    ("settings.py", "Django settings file — changes here affect production config"),
    ("manage.py", "Django manage.py — should not need modification for a bug fix"),
    ("setup.py", "setup.py — package metadata; unlikely to need modification for a bug fix"),
    ("pyproject.toml", "pyproject.toml — package metadata"),
    ("Makefile", "Makefile — build system; unlikely needed for a bug fix"),
    (".env", ".env file — never modify environment variable files"),
    ("secrets", "file path contains 'secrets' — potential credential exposure"),
    ("credentials", "file path contains 'credentials' — potential credential exposure"),
]


def sensitive_path_guard(changes: list[dict[str, Any]]) -> SafetyViolation | None:
    """Block writes to sensitive paths beyond just .github/workflows/."""
    violations: list[str] = []
    for change in changes:
        fp = change.get("file_path", "")
        for pattern, reason in _SENSITIVE_PATH_PATTERNS:
            if pattern in fp:
                violations.append(f"{fp}: {reason}")
                break

    if violations:
        logger.warning("SensitivePathGuard: %d sensitive path(s) in change set", len(violations))
        return SafetyViolation(
            guard="SensitivePathGuard",
            message=f"Changes target {len(violations)} sensitive path(s).",
            details=violations,
        )
    return None


# ── Guard 5: Secrets / credential scanner ────────────────────────────────────

_SECRET_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(?i)(api[_-]?key|apikey)\s*=\s*["\'][A-Za-z0-9+/]{16,}["\']'), "Hardcoded API key"),
    (re.compile(r'(?i)(password|passwd|secret|token)\s*=\s*["\'][^"\']{8,}["\']'), "Hardcoded credential"),
    (re.compile(r'(?i)bearer\s+[A-Za-z0-9\-_=]{20,}'), "Hardcoded Bearer token"),
    (re.compile(r'ghp_[A-Za-z0-9]{36}'), "GitHub personal access token"),
    (re.compile(r'sk-[A-Za-z0-9]{32,}'), "OpenAI API key pattern"),
    (re.compile(r'AKIA[0-9A-Z]{16}'), "AWS access key ID pattern"),
]


def secrets_guard(changes: list[dict[str, Any]]) -> SafetyViolation | None:
    """Detect hardcoded credentials or tokens in generated code."""
    hits: list[str] = []
    for change in changes:
        content = change.get("content") or change.get("replace") or ""
        fp = change.get("file_path", "?")
        for pattern, desc in _SECRET_PATTERNS:
            if pattern.search(content):
                hits.append(f"{fp}: {desc}")

    if hits:
        logger.error("SecretsGuard: %d credential pattern(s) in generated code", len(hits))
        return SafetyViolation(
            guard="SecretsGuard",
            message=f"Generated code contains {len(hits)} potential secret(s).",
            details=hits,
            severity="error",
        )
    return None


# ── Composite runner ──────────────────────────────────────────────────────────

def run_all_guards(
    changes: list[dict[str, Any]],
    clone_path: str,
) -> tuple[list[SafetyViolation], list[SafetyViolation]]:
    """Run all guards. Returns (errors, warnings).

    Errors should trigger a retry with feedback.
    Warnings are logged and included in the PR description but don't block.
    """
    errors: list[SafetyViolation] = []
    warnings: list[SafetyViolation] = []

    guards = [
        blast_radius_guard(changes),
        security_scan_guard(changes),
        dependency_guard(changes, clone_path),
        sensitive_path_guard(changes),
        secrets_guard(changes),
    ]

    for v in guards:
        if v is None:
            continue
        if v.severity == "error":
            errors.append(v)
        else:
            warnings.append(v)

    return errors, warnings
