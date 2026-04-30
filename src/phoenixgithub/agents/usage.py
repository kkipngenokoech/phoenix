"""Thread-local LLM usage tracker.

Each run resets its thread's accumulator at start. BaseAgent.invoke() increments
it after every LLM call. The orchestrator reads totals at the end and writes them
to run.json so the SWE-bench harness can report tokens and inference time per issue.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass


@dataclass
class LLMUsageStats:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    inference_seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    def __iadd__(self, other: "LLMUsageStats") -> "LLMUsageStats":
        self.llm_calls += other.llm_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.inference_seconds += other.inference_seconds
        return self


_thread_local = threading.local()


def reset_usage() -> None:
    """Call at the start of each run to zero out the per-thread accumulator."""
    _thread_local.stats = LLMUsageStats()


def get_usage() -> LLMUsageStats:
    """Return the accumulator for the current thread (auto-initialises if absent)."""
    if not hasattr(_thread_local, "stats"):
        _thread_local.stats = LLMUsageStats()
    return _thread_local.stats


def record_call(
    *,
    elapsed: float,
    response: object,
) -> None:
    """Extract token counts from a LangChain AIMessage and add to the accumulator.

    Tries usage_metadata (LangChain ≥0.2) then response_metadata.usage (older).
    Silently skips if neither attribute is present (provider doesn't report usage).
    """
    stats = get_usage()
    stats.llm_calls += 1
    stats.inference_seconds += elapsed

    # LangChain ≥0.2: AIMessage.usage_metadata = {'input_tokens': N, 'output_tokens': M, ...}
    meta = getattr(response, "usage_metadata", None)
    if isinstance(meta, dict):
        stats.input_tokens += meta.get("input_tokens", 0)
        stats.output_tokens += meta.get("output_tokens", 0)
        return

    # Older LangChain / Anthropic adapter: response_metadata['usage']
    rm = getattr(response, "response_metadata", None)
    if isinstance(rm, dict):
        u = rm.get("usage", {})
        if isinstance(u, dict):
            stats.input_tokens += u.get("input_tokens", 0)
            stats.output_tokens += u.get("output_tokens", 0)
