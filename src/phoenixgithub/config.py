"""Configuration — all settings from environment variables."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class GitHubAppConfig(BaseModel):
    """Configuration for GitHub App authentication (webhook mode)."""
    app_id: int = Field(default_factory=lambda: int(os.getenv("GITHUB_APP_ID", "0")))
    private_key_path: str = Field(default_factory=lambda: os.getenv("GITHUB_PRIVATE_KEY_PATH", ""))
    private_key: str = Field(default_factory=lambda: os.getenv("GITHUB_PRIVATE_KEY", ""))
    webhook_secret: str = Field(default_factory=lambda: os.getenv("GITHUB_WEBHOOK_SECRET", ""))
    webhook_port: int = Field(default_factory=lambda: int(os.getenv("WEBHOOK_PORT", "8000")))

    @property
    def is_configured(self) -> bool:
        return self.app_id > 0 and bool(self.private_key_path or self.private_key)

    def get_private_key(self) -> str:
        """Return the private key contents, reading from file if needed."""
        if self.private_key:
            return self.private_key
        if self.private_key_path:
            return Path(self.private_key_path).read_text()
        raise ValueError("No GitHub App private key configured")


class GitHubConfig(BaseModel):
    token: str = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    repo: str = Field(default_factory=lambda: os.getenv("GITHUB_REPO", ""))  # owner/repo
    poll_interval: int = Field(default_factory=lambda: int(os.getenv("POLL_INTERVAL", "60")))
    max_concurrent_runs: int = Field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_RUNS", "2")))

    @property
    def owner(self) -> str:
        return self.repo.split("/")[0] if "/" in self.repo else ""

    @property
    def repo_name(self) -> str:
        return self.repo.split("/")[1] if "/" in self.repo else self.repo


class LabelConfig(BaseModel):
    """GitHub labels used as state machine transitions."""
    ready: str = "ai:ready"
    in_progress: str = "ai:in-progress"
    review: str = "ai:review"
    revise: str = "ai:revise"
    done: str = "ai:done"
    failed: str = "ai:failed"


class LLMConfig(BaseModel):
    provider: LLMProvider = Field(
        default_factory=lambda: LLMProvider(os.getenv("LLM_PROVIDER", "anthropic").lower())
    )
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"))
    api_key: str = Field(default_factory=lambda: os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY", ""))
    base_url: str | None = Field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    temperature: float = 0.2
    max_tokens: int = 8192


class AgentConfig(BaseModel):
    max_retries: int = Field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "2")))
    test_command: str = Field(
        default_factory=lambda: os.getenv("TEST_COMMAND", "pytest --import-mode=importlib --rootdir=.")
    )
    build_command: str = Field(default_factory=lambda: os.getenv("BUILD_COMMAND", ""))
    auto_revise_on_test_failure: bool = Field(
        default_factory=lambda: os.getenv("AUTO_REVISE_ON_TEST_FAILURE", "true").lower() in {"1", "true", "yes"}
    )
    auto_revise_max_cycles: int = Field(default_factory=lambda: int(os.getenv("AUTO_REVISE_MAX_CYCLES", "3")))
    no_progress_root_cause_repeat_limit: int = Field(
        default_factory=lambda: int(os.getenv("NO_PROGRESS_ROOT_CAUSE_REPEAT_LIMIT", "2"))
    )
    revise_incremental: bool = Field(
        default_factory=lambda: os.getenv("REVISE_INCREMENTAL", "true").lower() in {"1", "true", "yes"}
    )
    allow_no_tests: bool = Field(
        default_factory=lambda: os.getenv("ALLOW_NO_TESTS", "false").lower() in {"1", "true", "yes"}
    )
    validation_profile: str = Field(default_factory=lambda: os.getenv("VALIDATION_PROFILE", "auto"))
    use_reproducer: bool = Field(
        default_factory=lambda: os.getenv("USE_REPRODUCER", "true").lower() in {"1", "true", "yes"}
    )
    # How Phoenix decides the issue is "resolved" after the test step (PR still requires tests
    # unless reproducer-only mode is intentionally strict).
    #   tests      — project test suite verdict only (default; matches typical GitHub review).
    #   reproducer — only the synthetic reproducer test passing counts (strict R→F→V).
    #   both       — suite must pass AND reproducer must not still fail when one was written.
    resolution_mode: str = Field(
        default_factory=lambda: os.getenv("RESOLUTION_MODE", "tests").strip().lower()
    )

    @field_validator("resolution_mode", mode="before")
    @classmethod
    def _validate_resolution_mode(cls, v: object) -> str:
        s = (str(v) if v is not None else "tests").strip().lower()
        if s in {"tests", "reproducer", "both"}:
            return s
        return "tests"


class Config(BaseModel):
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    github_app: GitHubAppConfig = Field(default_factory=GitHubAppConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    workspace_dir: str = Field(
        default_factory=lambda: os.getenv("WORKSPACE_DIR", str(Path.cwd() / "workspace"))
    )
    state_file: str = Field(
        default_factory=lambda: os.getenv("STATE_FILE", str(Path.cwd() / ".watcher-state.json"))
    )
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @classmethod
    def from_env(cls) -> Config:
        return cls()
