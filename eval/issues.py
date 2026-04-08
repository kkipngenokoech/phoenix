"""Mirror real open issues from upstream repos onto evaluation forks."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import json

from github import Github, GithubException

from eval.repos import EvalRepo

logger = logging.getLogger(__name__)

# Labels that indicate an issue is not actionable code work
_SKIP_LABELS = {"question", "wontfix", "invalid", "duplicate", "won't fix",
                "need more info", "needs more info", "awaiting response"}

# Labels that signal the issue is a good candidate for code work
_PREFER_LABELS = {"bug", "enhancement", "refactor", "refactoring", "improvement",
                  "tech debt", "technical debt", "cleanup", "good first issue",
                  "help wanted", "performance", "breaking change"}


@dataclass
class CreatedIssue:
    repo: str           # upstream full_name, e.g. "psf/requests"
    fork: str           # fork full_name,     e.g. "kipng/requests"
    issue_number: int   # issue number on the fork
    issue_url: str
    task_type: str      # "real_issue" (mirrored) or a synthetic task type
    title: str
    label_applied: str
    upstream_issue_number: int | None = None   # original issue # on upstream
    upstream_issue_url: str | None = None


def _score_issue(issue) -> int:
    """Higher score = better candidate for Phoenix. Used for sorting."""
    label_names = {l.name.lower() for l in issue.labels}
    if label_names & _SKIP_LABELS:
        return -1
    score = 0
    if label_names & _PREFER_LABELS:
        score += 10
    # Prefer issues with some body text (enough context for the agent)
    body_len = len(issue.body or "")
    if body_len > 100:
        score += 3
    if body_len > 500:
        score += 2
    # Prefer less stale issues
    if issue.comments > 0:
        score += 1
    return score


def fetch_upstream_issues(
    repo: EvalRepo,
    github_token: str,
    max_issues: int = 3,
) -> list:
    """Return up to max_issues open issues from the upstream repo, best-first."""
    g = Github(github_token)
    upstream = g.get_repo(repo.full_name)

    candidates = []
    try:
        for issue in upstream.get_issues(state="open", sort="updated", direction="desc"):
            if issue.pull_request:   # GitHub API returns PRs in issues endpoint
                continue
            score = _score_issue(issue)
            if score >= 0:
                candidates.append((score, issue))
            if len(candidates) >= max_issues * 5:  # fetch a buffer to score from
                break
    except GithubException as e:
        logger.error(f"Could not fetch issues from {repo.full_name}: {e}")
        return []

    # Sort by score descending, take top N
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [issue for _, issue in candidates[:max_issues]]


def mirror_issues_to_fork(
    repo: EvalRepo,
    fork_full_name: str,
    upstream_issues: list,
    github_token: str,
    label: str | None = None,
) -> list[CreatedIssue]:
    """Recreate upstream issues on the fork. Label is applied only if provided."""
    g = Github(github_token)
    fork_repo = g.get_repo(fork_full_name)

    created: list[CreatedIssue] = []
    for upstream_issue in upstream_issues:
        body = (
            f"{upstream_issue.body or ''}\n\n"
            f"---\n"
            f"*Mirrored from [{repo.full_name}#{upstream_issue.number}]"
            f"({upstream_issue.html_url}) for Phoenix evaluation.*"
        )
        try:
            fork_issue = fork_repo.create_issue(
                title=upstream_issue.title,
                body=body,
                labels=[label] if label else [],
            )
            logger.info(
                f"  #{fork_issue.number} ← upstream #{upstream_issue.number}: "
                f"{upstream_issue.title[:60]}..."
            )
            created.append(CreatedIssue(
                repo=repo.full_name,
                fork=fork_full_name,
                issue_number=fork_issue.number,
                issue_url=fork_issue.html_url,
                task_type="real_issue",
                title=upstream_issue.title,
                label_applied=label,
                upstream_issue_number=upstream_issue.number,
                upstream_issue_url=upstream_issue.html_url,
            ))
            time.sleep(1)  # avoid secondary rate limits
        except GithubException as e:
            logger.error(f"  Failed to create issue '{upstream_issue.title[:50]}': {e}")

    return created


def generate_all_issues(
    repos: list[EvalRepo],
    github_token: str,
    label: str = "ai:ready",
    max_per_repo: int = 3,
    output_file: str = "eval/results/issues.json",
    apply_label: bool = False,
) -> list[CreatedIssue]:
    """Mirror real open issues from each upstream repo onto its fork.

    Forks the repo if it doesn't already exist, then mirrors issues.
    Safe to re-run (creates new issues each time — intended for fresh eval runs).
    """
    from eval.baseline import fork_repo, ensure_labels

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    g = Github(github_token)
    your_username = g.get_user().login

    all_issues: list[CreatedIssue] = []
    for repo in repos:
        logger.info(f"\nFetching issues from {repo.full_name}...")
        upstream_issues = fetch_upstream_issues(repo, github_token, max_issues=max_per_repo)
        if not upstream_issues:
            logger.warning(f"  No suitable open issues found in {repo.full_name} — skipping")
            continue

        logger.info(f"  Found {len(upstream_issues)} issue(s) to mirror")
        try:
            fork_name = fork_repo(repo, github_token, your_username)
            ensure_labels(fork_name, github_token)
            # Pass label only if apply_label=True; otherwise create unlabelled
            trigger = label if apply_label else None
            issues = mirror_issues_to_fork(repo, fork_name, upstream_issues, github_token, trigger)
            all_issues.extend(issues)
        except Exception as e:
            logger.error(f"  Failed for {repo.full_name}: {e}")

    Path(output_file).write_text(
        json.dumps([asdict(i) for i in all_issues], indent=2)
    )
    logger.info(f"\n{len(all_issues)} issues mirrored. Saved to {output_file}")
    return all_issues


if __name__ == "__main__":
    import os
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    token = os.getenv("GITHUB_TOKEN", "")
    tier = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    max_per_repo = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    from eval.repos import PILOT_REPOS, TIER1_REPOS
    repo_set = {"pilot": PILOT_REPOS, "tier1": TIER1_REPOS}.get(tier, PILOT_REPOS)

    generate_all_issues(repo_set, token, max_per_repo=max_per_repo)
