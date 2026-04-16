#!/usr/bin/env python3
"""Enable GitHub Issues on your forks used by SWE-bench eval (PATCH has_issues)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from github import Github, GithubException  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enable Issues on owner/repo forks (default: your forks of all SWE-bench supported repos).",
    )
    parser.add_argument(
        "repos",
        nargs="*",
        metavar="NAME",
        help="Short repo name (django, requests) or full fork owner/repo. "
        "Omit to fix every fork matching eval.swebench.SUPPORTED_REPOS.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without calling the API",
    )
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        print("GITHUB_TOKEN required", file=sys.stderr)
        return 1

    from eval.swebench import SUPPORTED_REPOS  # noqa: E402

    g: Github | None = None
    login = os.getenv("GITHUB_LOGIN")
    if not login:
        g = Github(token)
        login = g.get_user().login

    if args.repos:
        forks: list[str] = []
        for name in args.repos:
            if "/" in name:
                forks.append(name)
            else:
                matched = [r for r in SUPPORTED_REPOS if r.split("/")[1] == name]
                if not matched:
                    print(f"Unknown repo short name: {name}", file=sys.stderr)
                    return 1
                for m in matched:
                    forks.append(f"{login}/{m.split('/')[1]}")
        forks = list(dict.fromkeys(forks))
    else:
        forks = [f"{login}/{r.split('/')[1]}" for r in sorted(SUPPORTED_REPOS)]

    for fork_full in forks:
        if args.dry_run:
            print(f"would enable issues: {fork_full}")
            continue
        if g is None:
            g = Github(token)
        try:
            r = g.get_repo(fork_full)
        except GithubException as e:
            msg = e.data.get("message", str(e)) if getattr(e, "data", None) else str(e)
            print(f"{fork_full}: cannot load ({e.status}): {msg}")
            continue
        if r.has_issues:
            print(f"{fork_full}: issues already enabled")
            continue
        try:
            r.edit(has_issues=True)
            print(f"{fork_full}: enabled issues")
        except GithubException as e:
            msg = e.data.get("message", str(e)) if getattr(e, "data", None) else str(e)
            print(f"{fork_full}: edit failed ({e.status}): {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
