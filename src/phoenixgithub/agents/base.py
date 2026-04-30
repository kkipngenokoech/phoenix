"""Base agent — shared LLM invocation logic for all specialized agents."""

from __future__ import annotations

import base64
import logging
import mimetypes
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from phoenixgithub.agents.usage import record_call
from phoenixgithub.config import LLMProvider
from phoenixgithub.models import RunContext

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all specialized agents."""

    role: str = ""
    system_prompt: str = ""

    def __init__(self, llm: BaseChatModel, provider: LLMProvider = LLMProvider.ANTHROPIC) -> None:
        self.llm = llm
        self.provider = provider

    def invoke(
        self,
        user_prompt: str,
        *,
        trace_name: str | None = None,
        trace_tags: list[str] | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send a prompt to the LLM with this agent's system prompt."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        config = self._build_trace_config(
            trace_name=trace_name,
            trace_tags=trace_tags,
            trace_metadata=trace_metadata,
        )
        _t0 = time.perf_counter()
        response = self.llm.invoke(messages, config=config or None)
        record_call(elapsed=time.perf_counter() - _t0, response=response)
        return self._stringify_content(response.content)

    def invoke_with_images(
        self,
        user_prompt: str,
        image_paths: list[str],
        *,
        trace_name: str | None = None,
        trace_tags: list[str] | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send a prompt with image attachments for vision-capable models."""
        messages = [SystemMessage(content=self.system_prompt)]
        if self.provider == LLMProvider.ANTHROPIC:
            content_blocks: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
            for image_path in image_paths:
                path = Path(image_path)
                if not path.exists():
                    continue
                data_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                media_type = mimetypes.guess_type(str(path))[0] or "image/png"
                content_blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data_b64},
                    }
                )
            messages.append(HumanMessage(content=content_blocks))
        elif self.provider == LLMProvider.OPENAI:
            content_blocks = [{"type": "text", "text": user_prompt}]
            for image_path in image_paths:
                path = Path(image_path)
                if not path.exists():
                    continue
                data_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
                media_type = mimetypes.guess_type(str(path))[0] or "image/png"
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data_b64}"},
                    }
                )
            messages.append(HumanMessage(content=content_blocks))
        else:
            # Fallback: no native image support for this model adapter.
            messages.append(
                HumanMessage(
                    content=(
                        f"{user_prompt}\n\n"
                        f"(Image paths attached but provider has no multimodal adapter: "
                        f"{', '.join(image_paths)})"
                    )
                )
            )

        config = self._build_trace_config(
            trace_name=trace_name,
            trace_tags=trace_tags,
            trace_metadata=trace_metadata,
        )
        _t0 = time.perf_counter()
        response = self.llm.invoke(messages, config=config or None)
        record_call(elapsed=time.perf_counter() - _t0, response=response)
        return self._stringify_content(response.content)

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if isinstance(part, str):
                    chunks.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks)
        return str(content)

    @staticmethod
    def _build_trace_config(
        *,
        trace_name: str | None,
        trace_tags: list[str] | None,
        trace_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if trace_name:
            config["run_name"] = trace_name
        if trace_tags:
            config["tags"] = trace_tags
        if trace_metadata:
            config["metadata"] = trace_metadata
        return config

    @staticmethod
    def _sanitize_body_for_waf(body: str, max_chars: int = 1500) -> str:
        """Extract signal-dense content from a GitHub issue body for safe LLM use.

        Strategy (in order):
        1. Strip full code blocks but keep their first 3 lines as context clues.
        2. Extract error/exception lines from tracebacks (the most useful part).
        3. Keep prose paragraphs up to the character limit.
        4. Append the most useful extracted error lines at the end.

        This preserves the semantic content of the issue while avoiding WAF triggers
        from large code blocks, stack traces, and JSON payloads.
        """
        import re

        TRACEBACK_ERROR_RE = re.compile(
            r"^((?:[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+)?"
            r"[A-Z][a-zA-Z0-9_]*(?:Error|Exception)[:\s].*)$",
            re.MULTILINE,
        )
        CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_.-]*)\n?(.*?)```", re.DOTALL)
        TRACEBACK_FRAME_RE = re.compile(r'^\s*File ".*", line \d+')
        CARET_LINE_RE = re.compile(r"^\s*[\^~]+\s*$")

        # Extract error lines from code blocks before stripping them
        error_lines: list[str] = []
        for m in CODE_BLOCK_RE.finditer(body):
            for err_m in TRACEBACK_ERROR_RE.finditer(m.group(2)):
                line = err_m.group(0).strip()
                if line and line not in error_lines:
                    error_lines.append(line)

        # Replace code blocks with stub + first 3 meaningful lines
        def _replace_block(m: re.Match) -> str:
            lang = (m.group(1) or "code").strip() or "code"
            preview_lines = [
                ln for ln in m.group(2).splitlines()[:3]
                if ln.strip() and not TRACEBACK_FRAME_RE.match(ln)
            ]
            stub = f"[{lang} block]"
            return (stub + "\n" + "\n".join(preview_lines)) if preview_lines else stub

        sanitized = CODE_BLOCK_RE.sub(_replace_block, body)

        # Remove raw traceback frames and caret lines from prose
        cleaned = [
            line for line in sanitized.splitlines()
            if not TRACEBACK_FRAME_RE.match(line) and not CARET_LINE_RE.match(line)
        ]
        sanitized = "\n".join(cleaned)

        # Budget prose chars to leave room for key error lines
        error_suffix = (
            "\n\nKey errors from issue:\n" + "\n".join(error_lines[:3])
            if error_lines else ""
        )
        prose_budget = max_chars - len(error_suffix)
        if len(sanitized) > prose_budget:
            sanitized = sanitized[:prose_budget] + "\n...(truncated)"

        return sanitized + error_suffix

    @abstractmethod
    def run(self, context: RunContext) -> dict[str, Any]:
        """Execute this agent's task. Returns outputs to merge into run context."""
        ...
