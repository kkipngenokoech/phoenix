"""Eval runner: listen for Phoenix completion events via SSE, collect results."""

from __future__ import annotations

import json
import logging
import time
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from github import Github, GithubException

from eval.issues import CreatedIssue

logger = logging.getLogger(__name__)

Status = Literal["pending", "in_progress", "succeeded", "failed", "timeout", "skipped"]

# SSE endpoint on the Phoenix server (same machine)
PHOENIX_SSE_URL = "http://localhost:8000/eval/stream"
MAX_WAIT = 60 * 30       # 30 minutes max wait per issue
FALLBACK_POLL_INTERVAL = 300  # 5-minute safety-net poll (only if SSE breaks)


@dataclass
class RunResult:
    repo: str
    fork: str
    issue_number: int
    issue_url: str
    task_type: str
    status: Status
    pr_number: int | None
    pr_url: str | None
    final_label: str
    elapsed_seconds: float
    error: str | None
    completed_at: str | None


def _find_pr_for_issue(gh_repo, issue_number: int) -> tuple[int | None, str | None]:
    """Search open/closed PRs that reference this issue number."""
    try:
        for pr in gh_repo.get_pulls(state="all"):
            body = pr.body or ""
            if f"#{issue_number}" in body or f"Closes #{issue_number}" in body:
                return pr.number, pr.html_url
    except GithubException:
        pass
    return None, None


def _get_current_label(gh_repo, issue_number: int) -> str:
    """Return the current ai:* label on the issue (fallback safety check)."""
    try:
        issue = gh_repo.get_issue(issue_number)
        ai_labels = [l.name for l in issue.labels if l.name.startswith("ai:")]
        return ai_labels[0] if ai_labels else ""
    except GithubException:
        return ""


def _get_run_error(workspace_dir: str = "workspace") -> str:
    """Return the error from the most recently modified run.json in workspace/runs/."""
    try:
        runs_dir = Path(workspace_dir) / "runs"
        run_files = sorted(
            runs_dir.glob("*/run.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not run_files:
            return ""
        data = json.loads(run_files[0].read_text())
        # Top-level error
        if data.get("error"):
            return str(data["error"])
        # Step-level errors
        for step in ("plan", "implement", "test", "pr"):
            err = data.get("steps", {}).get(step, {}).get("error")
            if err:
                return f"{step}: {err}"
        return ""
    except Exception:
        return ""


# ── SSE listener ──────────────────────────────────────────────────────────────

class EvalSSEListener:
    """Connects to Phoenix's /eval/stream SSE endpoint in a background thread.

    Issues are registered before the listener starts.  When a terminal label
    event arrives the corresponding threading.Event is set, unblocking the
    waiter in the main thread.
    """

    TERMINAL = {"ai:review", "ai:done", "ai:failed"}

    def __init__(self, sse_url: str = PHOENIX_SSE_URL) -> None:
        self.sse_url = sse_url
        # key: (fork_full_name, issue_number)
        self._events: dict[tuple[str, int], threading.Event] = {}
        self._results: dict[tuple[str, int], str] = {}  # key → terminal label
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def register(self, fork: str, issue_number: int) -> threading.Event:
        """Register an issue to wait for.  Returns the event to block on."""
        key = (fork, issue_number)
        ev = threading.Event()
        with self._lock:
            self._events[key] = ev
        return ev

    def get_label(self, fork: str, issue_number: int) -> str:
        return self._results.get((fork, issue_number), "")

    def start(self) -> None:
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        logger.info(f"SSE listener started → {self.sse_url}")

    def stop(self) -> None:
        self._stop.set()

    def _listen(self) -> None:
        import urllib.request
        backoff = 2
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(
                    self.sse_url,
                    headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
                )
                with urllib.request.urlopen(req, timeout=MAX_WAIT + 60) as resp:
                    backoff = 2  # reset on successful connect
                    for raw_line in resp:
                        if self._stop.is_set():
                            return
                        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                        if not line.startswith("data:"):
                            continue
                        try:
                            payload = json.loads(line[5:].strip())
                        except json.JSONDecodeError:
                            continue
                        self._handle(payload)
            except Exception as e:
                if self._stop.is_set():
                    return
                logger.warning(f"SSE connection error: {e}. Reconnecting in {backoff}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def _handle(self, payload: dict) -> None:
        label = payload.get("label", "")
        if label not in self.TERMINAL:
            return
        repo = payload.get("repo", "")
        issue_number = payload.get("issue_number")
        if issue_number is None:
            return
        key = (repo, issue_number)
        with self._lock:
            ev = self._events.get(key)
        if ev:
            self._results[key] = label
            ev.set()
            logger.info(f"SSE: {repo}#{issue_number} → {label}")
        else:
            logger.debug(f"SSE received event for untracked issue {repo}#{issue_number}")


# ── Public API ────────────────────────────────────────────────────────────────

def wait_for_completion(
    issue: CreatedIssue,
    github_token: str,
    completion_event: threading.Event,
    listener: EvalSSEListener,
    max_wait: int = MAX_WAIT,
) -> RunResult:
    """Wait for the SSE event signalling Phoenix completion, with a fallback poll."""
    g = Github(github_token)
    gh_repo = g.get_repo(issue.fork)
    start = time.time()

    # Wait for SSE signal (or timeout)
    fired = completion_event.wait(timeout=max_wait)

    elapsed = time.time() - start

    if fired:
        final_label = listener.get_label(issue.fork, issue.issue_number)
    else:
        # SSE may have missed the event — do one final label check
        final_label = _get_current_label(gh_repo, issue.issue_number)
        if not final_label or final_label not in EvalSSEListener.TERMINAL:
            logger.warning(
                f"Timeout after {elapsed:.0f}s for {issue.fork}#{issue.issue_number} "
                f"(label={final_label!r})"
            )
            return RunResult(
                repo=issue.repo,
                fork=issue.fork,
                issue_number=issue.issue_number,
                issue_url=issue.issue_url,
                task_type=issue.task_type,
                status="timeout",
                pr_number=None,
                pr_url=None,
                final_label=final_label,
                elapsed_seconds=round(elapsed, 1),
                error=f"Timed out after {max_wait}s in state '{final_label}'",
                completed_at=None,
            )

    status: Status = "succeeded" if final_label in ("ai:review", "ai:done") else "failed"
    pr_number, pr_url = _find_pr_for_issue(gh_repo, issue.issue_number)
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_error = _get_run_error() if status == "failed" else None

    logger.info(
        f"  Done in {elapsed:.0f}s → status={status} label={final_label} pr={pr_number}"
    )
    return RunResult(
        repo=issue.repo,
        fork=issue.fork,
        issue_number=issue.issue_number,
        issue_url=issue.issue_url,
        task_type=issue.task_type,
        status=status,
        pr_number=pr_number,
        pr_url=pr_url,
        final_label=final_label,
        elapsed_seconds=round(elapsed, 1),
        error=run_error,
        completed_at=completed_at,
    )


def _apply_trigger_label(issue: CreatedIssue, github_token: str, label: str = "ai:ready") -> None:
    """Apply the trigger label to a single issue to kick off Phoenix."""
    g = Github(auth=__import__("github").Auth.Token(github_token))
    gh_repo = g.get_repo(issue.fork)
    gh_issue = gh_repo.get_issue(issue.issue_number)
    current = {l.name for l in gh_issue.labels}
    if label not in current:
        gh_issue.add_to_labels(label)
        logger.info(f"  Applied {label} to {issue.fork}#{issue.issue_number}")
    else:
        logger.info(f"  {label} already on {issue.fork}#{issue.issue_number}")


def run_eval(
    issues: list[CreatedIssue],
    github_token: str,
    output_file: str = "eval/results/run_results.json",
    max_wait: int = MAX_WAIT,
    sse_url: str = PHOENIX_SSE_URL,
    trigger_label: str = "ai:ready",
) -> list[RunResult]:
    """
    Trigger Phoenix on each issue one at a time and wait for completion via SSE.

    Issues should be created WITHOUT the trigger label (via generate_all_issues
    with apply_label=False). This function applies the label to one issue,
    waits for the SSE completion event, then moves to the next — ensuring we
    never hit the concurrency limit with a flood of simultaneous webhooks.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Start the SSE listener and pre-register all issues
    listener = EvalSSEListener(sse_url=sse_url)
    events: dict[tuple[str, int], threading.Event] = {}
    for issue in issues:
        ev = listener.register(issue.fork, issue.issue_number)
        events[(issue.fork, issue.issue_number)] = ev

    listener.start()

    results: list[RunResult] = []
    total = len(issues)

    # Track live status for the progress table
    live: dict[tuple[str, int], str] = {}  # key → status emoji

    def _status_line(r: RunResult) -> str:
        icon = {"succeeded": "✅", "failed": "❌", "timeout": "⏱", "skipped": "⚠️"}.get(r.status, "?")
        pr = f" → PR #{r.pr_number}" if r.pr_number else ""
        reason = f" ({r.error[:60]})" if r.error and r.status != "succeeded" else ""
        return f"  {icon} {r.fork}#{r.issue_number}  [{r.elapsed_seconds:.0f}s]{pr}{reason}"

    def _print_live_table(results_so_far: list[RunResult], current_issue: CreatedIssue | None) -> None:
        print("\n" + "─" * 70)
        print(f"  PROGRESS  {len(results_so_far)}/{total} done")
        print("─" * 70)
        for r in results_so_far:
            print(_status_line(r))
        if current_issue:
            print(f"  🔄 {current_issue.fork}#{current_issue.issue_number}  (running...)")
        pending = total - len(results_so_far) - (1 if current_issue else 0)
        if pending > 0:
            print(f"  ⏳ {pending} issue(s) pending")
        print("─" * 70 + "\n")

    for i, issue in enumerate(issues, 1):
        logger.info(
            f"\n[{i}/{total}] {issue.fork}#{issue.issue_number} "
            f"— \"{issue.title[:55]}...\""
            f"\n  Upstream: {issue.upstream_issue_url or 'n/a'}"
            f"\n  Fork issue: {issue.issue_url}"
        )

        # Apply trigger label now (not all at once during issue creation)
        try:
            _apply_trigger_label(issue, github_token, trigger_label)
        except Exception as e:
            logger.warning(f"  Could not apply label: {e} — Phoenix may already have it")

        _print_live_table(results, current_issue=issue)

        ev = events[(issue.fork, issue.issue_number)]
        try:
            result = wait_for_completion(issue, github_token, ev, listener, max_wait)
        except Exception as e:
            logger.error(f"  Exception: {e}")
            result = RunResult(
                repo=issue.repo,
                fork=issue.fork,
                issue_number=issue.issue_number,
                issue_url=issue.issue_url,
                task_type=issue.task_type,
                status="skipped",
                pr_number=None,
                pr_url=None,
                final_label="",
                elapsed_seconds=0,
                error=str(e),
                completed_at=None,
            )
        results.append(result)
        _print_live_table(results, current_issue=None)

        # Brief pause between issues to avoid gateway burst throttling
        if i < total:
            time.sleep(10)

        # Save incrementally so partial results aren't lost
        Path(output_file).write_text(
            json.dumps([asdict(r) for r in results], indent=2)
        )

    listener.stop()
    logger.info(f"\nEval complete. {len(results)} results saved to {output_file}")
    _print_summary(results)

    # Write failure log
    failures = [r for r in results if r.status != "succeeded"]
    if failures:
        log_file = Path(output_file).parent / (Path(output_file).stem.replace("run_results", "failures") + ".md")
        _write_failure_log(failures, log_file)
        logger.info(f"Failure analysis log: {log_file}")

    return results


def _write_failure_log(failures: list[RunResult], log_file: Path) -> None:
    """Write a structured markdown log of all non-succeeded runs for analysis."""
    import datetime

    # Categorise failure reasons
    def _categorise(r: RunResult) -> str:
        err = (r.error or "").lower()
        if r.status == "timeout":
            return "webhook_not_received"
        if "403" in err or "forbidden" in err:
            return "gateway_rate_limit"
        if "invalid json" in err or "parse" in err:
            return "coder_invalid_json"
        if "plan failed" in err:
            return "planner_error"
        if "test" in err or "pytest" in err:
            return "test_failure"
        if "import" in err or "module" in err:
            return "missing_dependency"
        return "unknown"

    categories: dict[str, list[RunResult]] = {}
    for r in failures:
        cat = _categorise(r)
        categories.setdefault(cat, []).append(r)

    lines = [
        f"# Phoenix Eval — Failure Analysis",
        f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"Total failures: {len(failures)}",
        "",
        "## Summary by failure category",
        "",
        "| Category | Count | Description |",
        "|---|---|---|",
    ]

    category_descriptions = {
        "gateway_rate_limit":   "CMU AI gateway returned 403/429 — rate limited",
        "webhook_not_received": "ai:ready label applied but Phoenix never received webhook",
        "coder_invalid_json":   "Coder returned JSON that failed to parse after repair",
        "planner_error":        "Planner step threw an exception",
        "test_failure":         "Tests failed after code changes",
        "missing_dependency":   "Code referenced an import that isn't installed",
        "unknown":              "Could not categorise from error message",
    }

    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        desc = category_descriptions.get(cat, cat)
        lines.append(f"| `{cat}` | {len(items)} | {desc} |")

    lines += ["", "---", ""]

    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        lines += [
            f"## {cat.replace('_', ' ').title()} ({len(items)} issues)",
            "",
        ]
        for r in items:
            lines += [
                f"### {r.fork}#{r.issue_number}",
                f"- **Repo:** {r.repo}",
                f"- **Fork issue:** {r.issue_url}",
                f"- **Status:** `{r.status}`",
                f"- **Final label:** `{r.final_label or 'none'}`",
                f"- **Elapsed:** {r.elapsed_seconds:.0f}s",
                f"- **Error:**",
                f"  ```",
                f"  {(r.error or 'no error recorded')[:500]}",
                f"  ```",
                "",
            ]

    log_file.write_text("\n".join(lines))


def _print_summary(results: list[RunResult]) -> None:
    succeeded = sum(1 for r in results if r.status == "succeeded")
    failed = sum(1 for r in results if r.status == "failed")
    timeout = sum(1 for r in results if r.status == "timeout")
    total = len(results)
    cp = succeeded / total if total else 0

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"  Total issues  : {total}")
    print(f"  Succeeded     : {succeeded}")
    print(f"  Failed        : {failed}")
    print(f"  Timed out     : {timeout}")
    print(f"  CP (pass rate): {cp:.1%}")
    if results:
        avg_t = sum(r.elapsed_seconds for r in results) / total
        print(f"  Avg time/run  : {avg_t:.0f}s")
    print("=" * 60)

    task_types: dict[str, dict] = {}
    for r in results:
        tt = task_types.setdefault(r.task_type, {"ok": 0, "fail": 0})
        if r.status == "succeeded":
            tt["ok"] += 1
        else:
            tt["fail"] += 1
    print("\nPer task type:")
    for tt, counts in sorted(task_types.items()):
        n = counts["ok"] + counts["fail"]
        print(f"  {tt:<30} {counts['ok']}/{n} passed")


if __name__ == "__main__":
    import os
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    token = os.getenv("GITHUB_TOKEN", "")
    issues_file = sys.argv[1] if len(sys.argv) > 1 else "eval/results/issues.json"

    issues_data = json.loads(Path(issues_file).read_text())
    issues = [CreatedIssue(**d) for d in issues_data]

    run_eval(issues, token)
