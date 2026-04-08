"""Base agent — shared LLM invocation logic for all specialized agents."""

from __future__ import annotations

import base64
import logging
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all specialized agents."""

    role: str = ""
    system_prompt: str = ""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

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
        response = self.llm.invoke(messages, config=config or None)
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
        model_name = type(self.llm).__name__.lower()
        if "anthropic" in model_name:
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
        elif "openai" in model_name:
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
        response = self.llm.invoke(messages, config=config or None)
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
        """Remove code blocks / tracebacks and truncate to avoid WAF 403s.

        Stack traces and JSON payloads in GitHub issue bodies frequently match
        WAF rules for code injection, path traversal, and prompt injection.
        """
        import re

        def _replace_block(m: re.Match) -> str:
            lang = (m.group(1) or "code").strip() or "code"
            return f"[{lang} block omitted]"

        sanitized = re.sub(
            r"```([a-zA-Z0-9_.-]*)\n?(.*?)```",
            _replace_block,
            body,
            flags=re.DOTALL,
        )

        lines = sanitized.splitlines()
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('File "') and ", line " in stripped:
                continue
            if stripped and all(c in "^~" for c in stripped):
                continue
            cleaned.append(line)
        sanitized = "\n".join(cleaned)

        if len(sanitized) > max_chars:
            sanitized = sanitized[:max_chars] + "\n...(truncated)"
        return sanitized

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute this agent's task. Returns outputs to merge into run context."""
        ...
