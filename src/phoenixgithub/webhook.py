"""Webhook server — receives GitHub App events and dispatches runs."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import threading
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from phoenixgithub.config import Config
from phoenixgithub.github_app import GitHubAppAuth
from phoenixgithub.github_client import GitHubClient
from phoenixgithub.models import Run
from phoenixgithub.state import StateManager

logger = logging.getLogger(__name__)

# ── Eval event broadcast ──────────────────────────────────────────────────────
# When Phoenix transitions an issue to a terminal label (ai:review, ai:done,
# ai:failed), the label webhook fires back to this server.  We broadcast those
# events over SSE so the eval runner can react instantly instead of polling.

_eval_subscribers: list[asyncio.Queue] = []
_eval_subscribers_lock = threading.Lock()

TERMINAL_LABELS = {"ai:review", "ai:done", "ai:failed"}


def _broadcast_eval_event(event: dict) -> None:
    """Put an eval event on every active SSE subscriber queue (thread-safe)."""
    with _eval_subscribers_lock:
        subscribers = list(_eval_subscribers)
    for q in subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify the X-Hub-Signature-256 HMAC digest from GitHub."""
    if not signature.startswith("sha256="):
        return False
    expected = hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


def create_webhook_app(
    config: Config,
    app_auth: GitHubAppAuth,
    state: StateManager,
    on_dispatch: Callable[[Run, GitHubClient], None],
) -> FastAPI:
    """Create a FastAPI application that handles GitHub webhook events."""
    app = FastAPI(title="PhoenixGitHub Webhook", docs_url=None, redoc_url=None)
    webhook_secret = config.github_app.webhook_secret

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/eval/stream")
    async def eval_stream() -> StreamingResponse:
        """SSE stream of terminal label events for the eval runner.

        Each event is a JSON object:
            {"repo": "owner/name", "issue_number": 42, "label": "ai:review"}

        The eval runner subscribes here instead of polling the GitHub API.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        with _eval_subscribers_lock:
            _eval_subscribers.append(q)

        async def generate():
            try:
                # Heartbeat every 15 s to keep the connection alive
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
            finally:
                with _eval_subscribers_lock:
                    try:
                        _eval_subscribers.remove(q)
                    except ValueError:
                        pass

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/webhook")
    async def handle_webhook(
        request: Request,
        x_hub_signature_256: str = Header(None),
        x_github_event: str = Header(None),
    ) -> dict[str, Any]:
        body = await request.body()

        # Verify HMAC signature
        if webhook_secret:
            if not x_hub_signature_256:
                raise HTTPException(status_code=401, detail="Missing signature")
            if not verify_signature(body, x_hub_signature_256, webhook_secret):
                raise HTTPException(status_code=401, detail="Invalid signature")

        payload: dict[str, Any] = await request.json()

        # Only handle issue label events
        if x_github_event != "issues":
            return {"status": "ignored", "reason": f"event={x_github_event}"}

        action = payload.get("action")
        if action != "labeled":
            return {"status": "ignored", "reason": f"action={action}"}

        label_name = payload.get("label", {}).get("name", "")
        issue_number = payload.get("issue", {}).get("number")
        repo_full_name = payload.get("repository", {}).get("full_name", "")

        # Broadcast terminal label transitions to eval SSE subscribers
        if label_name in TERMINAL_LABELS:
            _broadcast_eval_event({
                "repo": repo_full_name,
                "issue_number": issue_number,
                "label": label_name,
            })
            logger.info(
                f"Eval event broadcast: {repo_full_name}#{issue_number} → {label_name}"
            )

        trigger_labels = {config.labels.ready, config.labels.revise}
        if label_name not in trigger_labels:
            return {"status": "ignored", "reason": f"label={label_name}"}

        # Extract event context
        issue = payload["issue"]
        repo_data = payload["repository"]
        installation_id = payload.get("installation", {}).get("id")

        if not installation_id:
            logger.error(f"No installation_id in webhook payload for {repo_full_name}")
            raise HTTPException(status_code=400, detail="Missing installation_id")

        # Check if already dispatched
        if state.is_dispatched(issue_number):
            logger.info(f"Issue #{issue_number} already dispatched — skipping")
            return {"status": "skipped", "reason": "already_dispatched"}

        # Check concurrency limit
        if state.watcher.active_runs >= config.github.max_concurrent_runs:
            logger.info(
                f"At concurrency limit ({state.watcher.active_runs}/"
                f"{config.github.max_concurrent_runs}) — skipping"
            )
            return {"status": "skipped", "reason": "concurrency_limit"}

        # Build a GitHubClient scoped to this installation
        github_client = GitHubClient.from_app_auth(
            config=config,
            app_auth=app_auth,
            installation_id=installation_id,
            repo=repo_full_name,
        )

        # Create and dispatch the run
        run = Run(
            repo=repo_full_name,
            issues=[issue_number],
            branch_name=f"phoenix/issue-{issue_number}",
        )
        run.context["trigger_label"] = label_name
        run.context["installation_id"] = installation_id

        state.mark_dispatched(issue_number, run.run_id)
        state.save_run(run)

        github_client.transition_label(
            issue_number,
            label_name,
            config.labels.in_progress,
        )
        github_client.comment_on_issue(
            issue_number,
            f"🤖 **Phoenix AI** picked up this issue.\n\n"
            f"**Run ID:** `{run.run_id}`\n"
            f"**Branch:** `{run.branch_name}`\n\n"
            f"Triggered by label: `{label_name}`\n\n"
            f"Working on it now...",
        )

        logger.info(
            f"Webhook dispatched run {run.run_id} for "
            f"{repo_full_name}#{issue_number} (label={label_name})"
        )

        # Dispatch in background thread
        thread = threading.Thread(
            target=on_dispatch,
            args=(run, github_client),
            daemon=True,
        )
        thread.start()

        return {
            "status": "dispatched",
            "run_id": run.run_id,
            "issue": issue_number,
            "repo": repo_full_name,
        }

    return app
