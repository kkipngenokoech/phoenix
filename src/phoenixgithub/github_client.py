"""GitHub client — wraps PyGithub + git CLI for all GitHub operations."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

import requests
from github import Github, GithubException
from github.Issue import Issue
from github.IssueComment import IssueComment
from github.PullRequest import PullRequest
from github.Repository import Repository
from git import Repo
from git.exc import GitCommandError

from phoenixgithub.config import Config, GitHubAppConfig, LabelConfig
from phoenixgithub.tools.git_utils import (
    compute_uncovered_paths,
    get_changed_paths,
    get_default_branch,
)
from phoenixgithub.tools.path_utils import extract_image_urls_from_texts, infer_image_extension

logger = logging.getLogger(__name__)


class GitHubClient:
    """All interactions with GitHub: issues, labels, clones, branches, PRs."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._gh = Github(config.github.token)
        self._repo: Repository = self._gh.get_repo(config.github.repo)
        self._labels = config.labels
        self._app_auth: object | None = None
        self._installation_id: int | None = None
        self._token_refreshed_at: float = 0.0

    def _ensure_fresh_token(self) -> None:
        """Refresh the installation token if it's older than 50 minutes."""
        import time
        if not self._app_auth or not self._installation_id:
            return
        if time.monotonic() - self._token_refreshed_at < 50 * 60:
            return
        self.refresh_token()
        self._token_refreshed_at = time.monotonic()

    @classmethod
    def from_app_auth(
        cls,
        config: Config,
        app_auth: object,
        installation_id: int,
        repo: str,
    ) -> GitHubClient:
        """Create a client authenticated via GitHub App installation token.

        Args:
            config: Application configuration.
            app_auth: GitHubAppAuth instance for token management.
            installation_id: The GitHub App installation ID.
            repo: Full repo name (owner/repo).
        """
        from phoenixgithub.github_app import GitHubAppAuth

        assert isinstance(app_auth, GitHubAppAuth)
        # Override repo in config for this client instance
        config = config.model_copy(
            update={"github": config.github.model_copy(update={"repo": repo})}
        )
        gh = app_auth.get_github_for_installation(installation_id)
        instance = cls.__new__(cls)
        instance.config = config
        instance._gh = gh
        instance._repo = gh.get_repo(repo)
        instance._labels = config.labels
        instance._app_auth = app_auth
        instance._installation_id = installation_id
        instance._token_refreshed_at = 0.0
        return instance

    # ------------------------------------------------------------------
    # Issues & Labels
    # ------------------------------------------------------------------

    def get_ready_issues(self) -> list[Issue]:
        """Find all open issues with the ai:ready label."""
        issues = self._get_issues_by_label(self._labels.ready)
        logger.info(f"Found {len(issues)} issues with label '{self._labels.ready}'")
        return issues

    def get_revise_issues(self) -> list[Issue]:
        """Find all open issues with the ai:revise label."""
        issues = self._get_issues_by_label(self._labels.revise)
        logger.info(f"Found {len(issues)} issues with label '{self._labels.revise}'")
        return issues

    def _get_issues_by_label(self, label_name: str) -> list[Issue]:
        """Find all open issues with a specific label, ensuring labels exist."""
        try:
            label = self._repo.get_label(label_name)
        except GithubException:
            logger.warning(f"Label '{label_name}' not found in repo — creating it")
            self._ensure_labels()
            label = self._repo.get_label(label_name)
        return list(self._repo.get_issues(state="open", labels=[label]))

    def transition_label(self, issue_number: int, from_label: str, to_label: str) -> None:
        """Set a single AI state label, clearing all other AI state labels."""
        self._ensure_fresh_token()
        issue = self._repo.get_issue(issue_number)
        ai_state_labels = {
            self._labels.ready,
            self._labels.in_progress,
            self._labels.review,
            self._labels.revise,
            self._labels.done,
            self._labels.failed,
        }
        existing_ai_labels = [label.name for label in issue.get_labels() if label.name in ai_state_labels]

        # Remove any conflicting AI state labels first.
        for label_name in existing_ai_labels:
            if label_name == to_label:
                continue
            try:
                issue.remove_from_labels(label_name)
            except GithubException:
                logger.debug(f"Issue #{issue_number}: could not remove label {label_name}")

        # Ensure target label is present.
        if to_label not in existing_ai_labels:
            issue.add_to_labels(to_label)

        logger.info(f"Issue #{issue_number}: {from_label} → {to_label}")

    def add_label(self, issue_number: int, label: str) -> None:
        self._repo.get_issue(issue_number).add_to_labels(label)

    def comment_on_issue(self, issue_number: int, body: str) -> None:
        self._ensure_fresh_token()
        self._repo.get_issue(issue_number).create_comment(body)

    def get_issue_comments(self, issue_number: int, limit: int = 30) -> list[dict[str, str]]:
        """Return recent issue comments with author metadata."""
        issue = self._repo.get_issue(issue_number)
        comments = list(issue.get_comments())
        selected = comments[-limit:] if limit > 0 else comments
        out: list[dict[str, str]] = []
        for c in selected:
            comment: IssueComment = c
            out.append(
                {
                    "author": getattr(comment.user, "login", "unknown") or "unknown",
                    "body": comment.body or "",
                }
            )
        return out

    def count_issue_comments_containing(self, issue_number: int, token: str) -> int:
        issue = self._repo.get_issue(issue_number)
        count = 0
        for comment in issue.get_comments():
            if token in (comment.body or ""):
                count += 1
        return count

    def get_issue(self, issue_number: int) -> Issue:
        return self._repo.get_issue(issue_number)

    def get_issue_image_urls(self, issue_number: int) -> list[str]:
        """Extract image URLs from the issue body and comments."""
        issue = self._repo.get_issue(issue_number)
        texts: list[str] = [issue.body or ""]
        for comment in issue.get_comments():
            texts.append(comment.body or "")
        return extract_image_urls_from_texts(texts)

    def download_issue_images(self, image_urls: list[str], target_dir: str, limit: int = 6) -> list[str]:
        """Download issue images locally for vision analysis."""
        if not image_urls:
            return []

        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        headers = {"Authorization": f"Bearer {self.config.github.token}"}

        for idx, url in enumerate(image_urls[:limit], start=1):
            try:
                resp = requests.get(url, headers=headers, timeout=20)
                if resp.status_code >= 400:
                    resp = requests.get(url, timeout=20)
                if resp.status_code >= 400:
                    logger.warning(f"Could not download image {url}: HTTP {resp.status_code}")
                    continue

                ext = infer_image_extension(url, resp.headers.get("content-type", ""))
                path = out_dir / f"issue_image_{idx}{ext}"
                path.write_bytes(resp.content)
                saved.append(str(path))
            except Exception as e:
                logger.warning(f"Failed to download image {url}: {e}")

        return saved

    def ensure_labels(self) -> None:
        """Public helper to ensure AI workflow labels exist."""
        self._ensure_labels()

    def _ensure_labels(self) -> None:
        """Create all AI labels if they don't exist."""
        existing = {l.name for l in self._repo.get_labels()}
        label_colors = {
            self._labels.ready: "0E8A16",
            self._labels.in_progress: "FBCA04",
            self._labels.review: "1D76DB",
            self._labels.revise: "E99695",
            self._labels.done: "0E8A16",
            self._labels.failed: "D93F0B",
        }
        for name, color in label_colors.items():
            if name not in existing:
                self._repo.create_label(name, color)
                logger.info(f"Created label: {name}")

    # ------------------------------------------------------------------
    # Clone & Branch
    # ------------------------------------------------------------------

    def ensure_clone(self, workspace_dir: str, *, full_reset: bool = True) -> str:
        """Clone the repo into workspace if not already cloned. Returns path to clone."""
        clone_path = Path(workspace_dir) / self.config.github.repo_name
        if clone_path.exists() and (clone_path / ".git").exists():
            repo = Repo(str(clone_path))
            if full_reset:
                # Refresh the remote URL with a fresh installation token so
                # pull doesn't fail with 401 when the previous token expired.
                token = self._get_clone_token()
                fresh_url = f"https://x-access-token:{token}@github.com/{self.config.github.repo}.git"
                repo.git.remote("set-url", "origin", fresh_url)

                logger.info(f"Pulling latest on {repo.active_branch.name}...")
                default_branch = get_default_branch(repo)
                repo.git.checkout("-f", default_branch)
                repo.git.fetch("origin", default_branch)
                # Use reset --hard instead of pull to handle force-pushed branches
                # (e.g. SWE-bench eval resets forks to specific base commits).
                repo.git.reset("--hard", f"origin/{default_branch}")
                repo.git.clean("-fd")
            else:
                logger.info("Reusing existing clone/worktree for incremental revise")
            return str(clone_path)

        clone_path.parent.mkdir(parents=True, exist_ok=True)
        token = self._get_clone_token()
        clone_url = f"https://x-access-token:{token}@github.com/{self.config.github.repo}.git"
        Repo.clone_from(clone_url, str(clone_path))
        logger.info(f"Cloned {self.config.github.repo} → {clone_path}")
        return str(clone_path)

    def create_branch(self, clone_path: str, branch_name: str, *, full_reset: bool = True) -> Repo:
        """Recreate and checkout a feature branch from the latest default branch."""
        repo = Repo(clone_path)
        default_branch = get_default_branch(repo)
        if full_reset:
            repo.git.checkout("-f", default_branch)
            repo.git.fetch("origin", default_branch)
            repo.git.reset("--hard", f"origin/{default_branch}")
            repo.git.clean("-fd")

            if branch_name in [b.name for b in repo.branches]:
                repo.git.branch("-D", branch_name)
                logger.info(f"Reset local branch: {branch_name}")

            # Ensure reruns start from a clean remote branch tip as well.
            try:
                repo.git.push("origin", f":{branch_name}")
                logger.info(f"Deleted remote branch (if existed): {branch_name}")
            except GitCommandError:
                logger.info(f"Remote branch did not exist (or could not be deleted): {branch_name}")

            repo.git.checkout("-b", branch_name)
            logger.info(f"Created branch: {branch_name}")
            return repo

        # Incremental revise mode: keep current issue branch history/worktree.
        if branch_name in [b.name for b in repo.branches]:
            repo.git.checkout(branch_name)
            logger.info(f"Reusing existing branch for revise: {branch_name}")
        else:
            repo.git.checkout(default_branch)
            repo.git.fetch("origin", default_branch)
            repo.git.reset("--hard", f"origin/{default_branch}")
            repo.git.checkout("-b", branch_name)
            logger.info(f"Created branch for revise: {branch_name}")
        return repo

    def commit_and_push(
        self, clone_path: str, branch_name: str, message: str, files: list[str] | None = None
    ) -> str:
        """Stage files, commit, push to remote. Returns commit SHA."""
        repo = Repo(clone_path)
        changed_paths = get_changed_paths(repo)
        if files:
            requested = {f.strip().rstrip("/") for f in files if f and f.strip()}
            omitted = sorted(compute_uncovered_paths(changed_paths, requested))
            if omitted:
                preview = ", ".join(omitted[:8])
                suffix = " ..." if len(omitted) > 8 else ""
                logger.warning(
                    "Some changed files were not in applied_files; auto-staging them: "
                    f"{preview}{suffix}"
                )
            for f in files:
                repo.git.add(f)
            # Auto-stage uncovered changed paths to avoid losing files that were
            # modified/created in earlier attempts but omitted from applied_files.
            for path in omitted:
                if path.endswith("/"):
                    continue
                repo.git.add(path)
        else:
            repo.git.add("-A")

        if not repo.is_dirty(untracked_files=False) and not repo.index.diff("HEAD"):
            logger.warning("Nothing to commit")
            return repo.head.commit.hexsha

        repo.index.commit(message)
        repo.git.push("--set-upstream", "origin", branch_name)
        sha = repo.head.commit.hexsha
        logger.info(f"Pushed {sha[:8]} to origin/{branch_name}")
        return sha

    # ------------------------------------------------------------------
    # Pull Requests
    # ------------------------------------------------------------------

    def create_pull_request(
        self,
        branch_name: str,
        title: str,
        body: str,
        issue_numbers: list[int],
        labels: list[str] | None = None,
    ) -> PullRequest:
        """Create a PR from the feature branch to the default branch."""
        self._ensure_fresh_token()
        closes_refs = " ".join(f"Closes #{n}" for n in issue_numbers)
        full_body = f"{body}\n\n---\n{closes_refs}"

        default = self._repo.default_branch
        try:
            pr = self._repo.create_pull(
                title=title,
                body=full_body,
                head=branch_name,
                base=default,
            )
        except GithubException as exc:
            # In revise mode we may push to an existing issue branch that already
            # has an open PR. Reuse it instead of failing the run.
            message = str(exc)
            if exc.status == 422 and "already exists" in message.lower():
                owner_head = f"{self.config.github.owner}:{branch_name}"
                existing = list(self._repo.get_pulls(state="open", head=owner_head, base=default))
                if existing:
                    pr = existing[0]
                    logger.info(f"Reusing existing PR #{pr.number}: {pr.html_url}")
                else:
                    raise
            else:
                raise

        if labels:
            pr.add_to_labels(*labels)

        logger.info(f"Created PR #{pr.number}: {pr.html_url}")
        return pr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_clone_token(self) -> str:
        """Return the best available token for git clone/push operations."""
        if self._app_auth and self._installation_id:
            from phoenixgithub.github_app import GitHubAppAuth

            assert isinstance(self._app_auth, GitHubAppAuth)
            return self._app_auth.get_access_token(self._installation_id)
        return self.config.github.token

    def refresh_token(self) -> None:
        """Refresh the installation token and PyGithub client (app mode only).

        Call this before long-running operations that may exceed the
        1-hour installation token lifetime.
        """
        if not self._app_auth or not self._installation_id:
            return
        from phoenixgithub.github_app import GitHubAppAuth

        assert isinstance(self._app_auth, GitHubAppAuth)
        self._gh = self._app_auth.get_github_for_installation(self._installation_id)
        self._repo = self._gh.get_repo(self.config.github.repo)
        logger.info("Refreshed GitHub App installation token")


