"""Git-related helper utilities used by runtime modules."""

from __future__ import annotations

from git import Repo


def get_default_branch(repo: Repo) -> str:
    """Return the default branch for a local repo clone.

    Priority:
    1. Symbolic ref of the remote HEAD (most accurate — what GitHub reports)
    2. Common default names present in the local branch list
    3. First local branch as final fallback
    """
    try:
        # e.g. "ref: refs/remotes/origin/main" → "main"
        ref = repo.git.symbolic_ref("refs/remotes/origin/HEAD")
        return ref.split("/")[-1]
    except Exception:
        pass
    branches = [b.name for b in repo.branches]
    for candidate in ("main", "master", "dev", "develop", "trunk"):
        if candidate in branches:
            return candidate
    return branches[0] if branches else "main"


def get_changed_paths(repo: Repo) -> set[str]:
    """Collect changed paths from git porcelain status."""
    try:
        porcelain = repo.git.status("--porcelain")
    except Exception:
        return set()

    paths: set[str] = set()
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        raw = line[3:].strip()
        if " -> " in raw:
            # For renames/copies, include destination path.
            raw = raw.split(" -> ", 1)[1].strip()
        if raw:
            paths.add(raw)
    return paths


def compute_uncovered_paths(changed_paths: set[str], requested_paths: set[str]) -> set[str]:
    """Return changed paths that are not covered by requested commit paths."""
    uncovered: set[str] = set()
    for path in changed_paths:
        normalized = path.rstrip("/")
        if normalized in requested_paths:
            continue
        # If git reports an untracked dir (e.g. "css/"), treat as covered
        # when at least one requested file is under that directory.
        if path.endswith("/") and any(req.startswith(normalized + "/") for req in requested_paths):
            continue
        uncovered.add(path)
    return uncovered

