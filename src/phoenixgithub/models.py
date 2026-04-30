"""Data models for runs, steps, and state transitions."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore[assignment]


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

    # LLM usage totals for the entire run (all agents combined)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    inference_seconds: float = 0.0

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

    def flush_llm_usage(self) -> None:
        """Copy the current thread's LLM usage accumulator into this run's fields."""
        from phoenixgithub.agents.usage import get_usage
        stats = get_usage()
        self.llm_calls = stats.llm_calls
        self.input_tokens = stats.input_tokens
        self.output_tokens = stats.output_tokens
        self.inference_seconds = round(stats.inference_seconds, 2)


class WatcherState(BaseModel):
    """Persistent state for the watcher daemon — tracks dispatched issues."""
    dispatched: dict[str, str] = Field(default_factory=dict)  # "issue-42" -> "run-abc123"
    active_runs: int = 0
    last_poll: Optional[datetime] = None


class RunContext(TypedDict, total=False):
    """Typed pipeline context threaded through orchestrator and all agents.

    All fields are optional (total=False) because context is built up
    incrementally — each step adds its outputs for subsequent steps to read.
    """
    # Identity
    run_id: str
    repo: str
    issue_number: int
    issue_title: str
    issue_body: str
    branch_name: str
    trigger_label: str

    # Issue metadata
    issue_comments: list[dict[str, str]]
    revision_notes: str
    issue_image_urls: list[str]
    issue_image_paths: list[str]

    # Clone
    clone_path: str

    # Planner outputs
    plan: dict[str, Any]
    planner_no_code: bool
    project_type: str

    # Reproducer outputs
    relevant_code: str
    relevant_code_for_reproducer: str
    reproduced: bool
    reproducer_file: str
    reproducer_test: str
    reproducer_skipped: bool
    false_positive: bool

    # Coder inputs/outputs
    step_attempt: int
    verify_feedback: str
    applied_files: list[str]
    changes: list[dict[str, Any]]
    commit_message: str
    agentic: bool

    # Safety
    safety_warnings: list[str]

    # Tester outputs
    test_output: dict[str, Any]
    test_verdict: str
    issue_resolved: bool
    test_passed: bool
    feedback: str
    last_test_feedback: str
    last_test_output: dict[str, Any]
    auto_guidance: str

    # PR Agent outputs
    pr_title: str
    pr_body: str
    commit_sha: str

    # Failure analyst inputs
    run_summary: str
    test_feedback: str

    # Visual context (screenshots)
    visual_context: str
