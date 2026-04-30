"""Agentic coder — iterative tool-using agent for code changes.

Replaces the one-shot CoderAgent with a multi-step loop: the model reads
files, writes fixes, runs verification commands, and calls finish() when done.
Model-agnostic — works with any LangChain BaseChatModel that supports tool calling
(Anthropic, OpenAI, Google, etc.).
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from phoenixgithub.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AgenticCoderAgent(BaseAgent):
    role = "agentic_coder"
    MAX_STEPS = 30

    system_prompt = """You are an expert software engineer fixing a real GitHub bug.

You have tools: read_file, write_file, list_files, bash, finish.

Workflow:
1. Read the issue, plan, and reproducer test carefully.
2. Use list_files / read_file to locate and understand the broken code.
3. Apply a surgical fix with write_file — change only what is broken.
4. Run: pip install -e . --quiet (to pick up your changes), then pytest <reproducer_file> -x --tb=short
5. If the test still fails, read the output and refine your fix.
6. Call finish only when the reproducer test passes (or you have exhausted attempts).

Rules:
- Only touch files directly related to the bug. Do not restructure the project.
- Never write to .github/workflows/.
- Never run curl, wget, git push, or git commit in bash.
- Write COMPLETE file content when calling write_file — no ellipsis or placeholders.
- Fewer lines changed = better. Prefer targeted edits over full rewrites.
- Call finish exactly once."""

    # ── Tool argument schemas ─────────────────────────────────────────────────

    class _ReadFileArgs(BaseModel):
        path: str = Field(description="Relative path from repo root")

    class _WriteFileArgs(BaseModel):
        path: str = Field(description="Relative path from repo root")
        content: str = Field(description="Complete new content for the file")

    class _ListFilesArgs(BaseModel):
        directory: str = Field(default=".", description="Directory to list (relative)")
        pattern: str = Field(default="**/*.py", description="Glob pattern")

    class _BashArgs(BaseModel):
        command: str = Field(description="Shell command to run in repo root (no network/install)")

    class _FinishArgs(BaseModel):
        commit_message: str = Field(description="Short commit message, e.g. 'fix: correct off-by-one in parser'")

    # ── Tool factory (closes over repo_root) ─────────────────────────────────

    def _make_tools(self, repo_root: Path) -> list:
        root = repo_root
        # Allow `pip install -e .` (editable install of current package) but block everything else
        _BLOCKED = ["apt-get", "apt ", "curl ", "wget ", "git push", "git commit", "rm -rf /"]
        _BLOCKED_PIP = ["pip install ", "pip3 install "]  # block arbitrary installs
        _ALLOWED_PIP = ["pip install -e .", "pip3 install -e ."]  # editable install is fine

        def read_file(path: str) -> str:
            try:
                full = (root / path).resolve()
                full.relative_to(root)
                if not full.exists():
                    return f"Error: {path} does not exist"
                text = full.read_text(errors="replace")
                if len(text) > 12000:
                    return text[:12000] + f"\n...(truncated — {len(text)} chars total)"
                return text
            except Exception as e:
                return f"Error reading {path}: {e}"

        def write_file(path: str, content: str) -> str:
            try:
                full = (root / path).resolve()
                rel = full.relative_to(root)
                if rel.parts[:2] == (".github", "workflows"):
                    return "Error: cannot write to .github/workflows/ — permission denied"
                full.parent.mkdir(parents=True, exist_ok=True)
                full.write_text(content)
                return f"Written: {path} ({len(content)} chars)"
            except Exception as e:
                return f"Error writing {path}: {e}"

        def list_files(directory: str = ".", pattern: str = "**/*.py") -> str:
            try:
                full = (root / directory).resolve()
                full.relative_to(root)
                files = sorted(str(p.relative_to(root)) for p in full.glob(pattern))[:60]
                return "\n".join(files) if files else "(no files match)"
            except Exception as e:
                return f"Error listing {directory}: {e}"

        def bash(command: str) -> str:
            for blocked in _BLOCKED:
                if blocked in command:
                    return f"Error: blocked — '{blocked}' is not allowed"
            # Block arbitrary pip installs but allow editable install of current package
            for pip_cmd in _BLOCKED_PIP:
                if pip_cmd in command and not any(ok in command for ok in _ALLOWED_PIP):
                    return "Error: use 'pip install -e .' to install the current package; arbitrary pip installs are blocked"
            try:
                result = subprocess.run(
                    command, shell=True, cwd=root,
                    capture_output=True, text=True, timeout=30,
                )
                out = (result.stdout + result.stderr).strip()
                if len(out) > 3000:
                    out = out[:3000] + "\n...(truncated)"
                return out or "(no output)"
            except subprocess.TimeoutExpired:
                return "Error: command timed out after 30s"
            except Exception as e:
                return f"Error: {e}"

        def finish(commit_message: str) -> str:
            return json.dumps({"done": True, "commit_message": commit_message})

        return [
            StructuredTool.from_function(read_file, name="read_file", args_schema=self._ReadFileArgs,
                                         description="Read a file from the repository"),
            StructuredTool.from_function(write_file, name="write_file", args_schema=self._WriteFileArgs,
                                         description="Write complete content to a file"),
            StructuredTool.from_function(list_files, name="list_files", args_schema=self._ListFilesArgs,
                                         description="List files matching a glob pattern"),
            StructuredTool.from_function(bash, name="bash", args_schema=self._BashArgs,
                                         description="Run a shell command in the repo root (read-only, no network)"),
            StructuredTool.from_function(finish, name="finish", args_schema=self._FinishArgs,
                                         description="Signal that the fix is complete"),
        ]

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        clone_path = context["clone_path"]
        repo_root = Path(clone_path).resolve()
        tools = self._make_tools(repo_root)
        tool_map = {t.name: t for t in tools}

        # Bind tools — fall back to one-shot CoderAgent if model doesn't support it
        try:
            llm_with_tools = self.llm.bind_tools(tools)
        except Exception:
            logger.warning("AgenticCoderAgent: bind_tools unsupported — falling back to CoderAgent")
            from phoenixgithub.agents.coder import CoderAgent
            return CoderAgent(self.llm).run(context)

        issue_body_safe = self._sanitize_body_for_waf(context.get("issue_body", ""), max_chars=1500)
        verify_feedback = context.get("verify_feedback", "")
        reproducer_file = context.get("reproducer_file") or ""
        reproducer_test = context.get("reproducer_test") or ""

        feedback_section = (
            f"\n\n## Previous Attempt Feedback\n{verify_feedback}\nFix these issues in this attempt."
            if verify_feedback else ""
        )

        reproducer_section = ""
        if reproducer_file and reproducer_test:
            reproducer_section = (
                f"\n\n## Reproducer Test (must PASS after your fix)\n"
                f"File: `{reproducer_file}`\n"
                f"```python\n{reproducer_test[:3000]}\n```\n"
                f"After writing your fix, run:\n"
                f"  pip install -e . --quiet\n"
                f"  pytest {reproducer_file} -x --tb=short\n"
                f"The test must exit 0 before you call finish."
            )

        initial_prompt = (
            f"## Issue\n"
            f"**Title:** {context.get('issue_title', '')}\n\n"
            f"**Description:**\n{issue_body_safe}\n\n"
            f"## Implementation Plan\n"
            f"```json\n{json.dumps(context.get('plan', {}), indent=2)}\n```"
            f"{reproducer_section}"
            f"{feedback_section}\n\n"
            f"Explore the repo, apply a surgical fix, verify it, then call finish."
        )

        messages: list = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=initial_prompt),
        ]

        written_files: dict[str, str] = {}
        commit_message = f"fix: resolve #{context.get('issue_number', '?')}"
        steps_taken = 0

        for step in range(self.MAX_STEPS):
            steps_taken = step + 1
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error(f"AgenticCoderAgent step {step} error: {e}")
                break

            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                logger.info(f"AgenticCoderAgent: no tool calls at step {step} — stopping")
                break

            # Warn model when approaching step limit so it wraps up
            remaining = self.MAX_STEPS - step - 1
            if remaining == 5 and written_files:
                messages.append(HumanMessage(
                    content=f"You have {remaining} steps left. If your fix is written, call finish now with a commit message."
                ))

            finished = False
            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("args", {})
                tc_id = tc["id"]

                if name == "finish":
                    commit_message = args.get("commit_message", commit_message)
                    messages.append(ToolMessage(content="done", tool_call_id=tc_id))
                    finished = True
                    continue

                try:
                    result = tool_map[name].invoke(args)
                except Exception as e:
                    result = f"Tool error: {e}"

                logger.debug(f"[{name}] → {str(result)[:100]}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))

                if name == "write_file" and "Error" not in str(result):
                    written_files[args.get("path", "")] = args.get("content", "")

            if finished:
                break

        changes = [
            {"file_path": path, "action": "modify", "content": content}
            for path, content in written_files.items()
            if path
        ]

        logger.info(
            f"AgenticCoderAgent: {len(changes)} file(s) written across {steps_taken} steps"
        )

        return {
            "changes": changes,
            "applied_files": list(written_files.keys()),
            "commit_message": commit_message,
            "agentic": True,  # signals orchestrator to skip plan-membership hallucination guard
        }
