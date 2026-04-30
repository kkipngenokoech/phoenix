"""State manager — persists run state and watcher state to JSON files."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from phoenixgithub.models import Run, RunStatus, WatcherState

logger = logging.getLogger(__name__)


class StateManager:
    """Manages watcher state (dispatched issues) and per-run state (run.json)."""

    def __init__(self, state_file: str, workspace_dir: str) -> None:
        self._state_file = Path(state_file)
        self._workspace_dir = Path(workspace_dir)
        self._lock = threading.Lock()
        self._watcher = self._load_watcher_state()

    # ------------------------------------------------------------------
    # Watcher state
    # ------------------------------------------------------------------

    def _load_watcher_state(self) -> WatcherState:
        if self._state_file.exists():
            data = json.loads(self._state_file.read_text())
            return WatcherState.model_validate(data)
        return WatcherState()

    def save_watcher_state(self) -> None:
        with self._lock:
            self._flush()

    @property
    def watcher(self) -> WatcherState:
        return self._watcher

    def is_dispatched(self, issue_number: int) -> bool:
        return f"issue-{issue_number}" in self._watcher.dispatched

    def mark_dispatched(self, issue_number: int, run_id: str) -> None:
        with self._lock:
            self._watcher.dispatched[f"issue-{issue_number}"] = run_id
            self._watcher.active_runs += 1
            self._watcher.last_poll = datetime.now(timezone.utc)
            self._flush()

    def mark_run_finished(self, run_id: str) -> None:
        # Release any issue dispatch locks owned by this run so a relabel to
        # ai:ready can be picked up again in future polling cycles.
        with self._lock:
            stale_keys = [k for k, v in self._watcher.dispatched.items() if v == run_id]
            for key in stale_keys:
                self._watcher.dispatched.pop(key, None)
            self._watcher.active_runs = max(0, self._watcher.active_runs - 1)
            self._flush()

    def clear_dispatched(self, issue_number: int) -> None:
        """Allow an issue to be re-dispatched (e.g. after ai:revise)."""
        with self._lock:
            self._watcher.dispatched.pop(f"issue-{issue_number}", None)
            self._flush()

    def _flush(self) -> None:
        """Write watcher state to disk. Must be called while holding self._lock."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(self._watcher.model_dump_json(indent=2))

    # ------------------------------------------------------------------
    # Per-run state
    # ------------------------------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        d = self._workspace_dir / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_run(self, run: Run) -> None:
        run.updated_at = datetime.now(timezone.utc)
        path = self._run_dir(run.run_id) / "run.json"
        path.write_text(run.model_dump_json(indent=2))
        logger.debug(f"Saved run state: {path}")

    def load_run(self, run_id: str) -> Optional[Run]:
        path = self._run_dir(run_id) / "run.json"
        if not path.exists():
            return None
        return Run.model_validate_json(path.read_text())

    def list_runs(self, status: RunStatus | None = None) -> list[Run]:
        runs_dir = self._workspace_dir / "runs"
        if not runs_dir.exists():
            return []
        runs = []
        for d in runs_dir.iterdir():
            run_file = d / "run.json"
            if run_file.exists():
                run = Run.model_validate_json(run_file.read_text())
                if status is None or run.status == status:
                    runs.append(run)
        return sorted(runs, key=lambda r: r.created_at, reverse=True)
