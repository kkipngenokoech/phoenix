"""Data models for runs, steps, and state transitions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepID(str, Enum):
    PLAN = "plan"
    REPRODUCE = "reproduce"
    IMPLEMENT = "implement"
    TEST = "test"
    PR = "pr"


class StepState(BaseModel):
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    retries: int = 0


class Run(BaseModel):
    """Single end-to-end run triggered by one or more GitHub issues."""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: RunStatus = RunStatus.PENDING
    repo: str = ""
    issues: list[int] = Field(default_factory=list)
    branch_name: str = ""
    pr_number: Optional[int] = None
    pr_url: Optional[str] = None

    steps: dict[str, StepState] = Field(default_factory=lambda: {
        StepID.PLAN: StepState(),
        StepID.REPRODUCE: StepState(),
        StepID.IMPLEMENT: StepState(),
        StepID.TEST: StepState(),
        StepID.PR: StepState(),
    })

    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None

    def step(self, step_id: str | StepID) -> StepState:
        key = step_id.value if isinstance(step_id, StepID) else step_id
        return self.steps[key]

    def set_step_running(self, step_id: StepID) -> None:
        s = self.step(step_id)
        s.status = StepStatus.RUNNING
        s.started_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def set_step_done(self, step_id: StepID, outputs: dict | None = None) -> None:
        s = self.step(step_id)
        s.status = StepStatus.DONE
        s.finished_at = datetime.now(timezone.utc)
        if outputs:
            s.outputs = outputs
        self.updated_at = datetime.now(timezone.utc)

    def set_step_failed(self, step_id: StepID, error: str) -> None:
        s = self.step(step_id)
        s.status = StepStatus.FAILED
        s.finished_at = datetime.now(timezone.utc)
        s.error = error
        self.updated_at = datetime.now(timezone.utc)


class WatcherState(BaseModel):
    """Persistent state for the watcher daemon — tracks dispatched issues."""
    dispatched: dict[str, str] = Field(default_factory=dict)  # "issue-42" -> "run-abc123"
    active_runs: int = 0
    last_poll: Optional[datetime] = None
