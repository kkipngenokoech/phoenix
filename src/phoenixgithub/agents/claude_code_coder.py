"""ClaudeCodeCoderAgent — uses the Claude Code CLI (claude -p) to fix bugs.

Strategy:
  1. Build a prompt with the issue, plan, and reproducer.
  2. Run `claude --print --dangerously-skip-permissions` in the repo directory.
     Claude Code handles its own file exploration and editing (Read/Edit/Bash tools).
  3. After the CLI exits, capture changed files via `git status --porcelain`.
  4. Return the changes list for the orchestrator to commit.

Fallback: if the `claude` binary is not found or the run fails, delegates to
AgenticCoderAgent (LangChain tool-calling loop).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from phoenixgithub.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class ClaudeCodeCoderAgent(BaseAgent):
    role = "claude_code_coder"

    def __init__(self, llm: Any) -> None:
        super().__init__(llm)
        # Read LLM config from environment for CLI invocation
        self._api_key = os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
        self._base_url = os.getenv("LLM_BASE_URL", "")
        raw_model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
        self._cli_model = self._resolve_cli_model(raw_model)

    @staticmethod
    def _resolve_cli_model(model: str) -> str:
        """Map gateway model IDs to claude CLI aliases."""
        m = model.lower()
        if "opus" in m:
            return "opus"
        if "haiku" in m:
            return "haiku"
        return "sonnet"  # default

    def _claude_available(self) -> bool:
        return shutil.which("claude") is not None

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(self, context: dict[str, Any]) -> str:
        issue_body_safe = self._sanitize_body_for_waf(
            context.get("issue_body", ""), max_chars=2000
        )
        plan = context.get("plan", {})
        reproducer_file = context.get("reproducer_file") or ""
        reproducer_test = context.get("reproducer_test") or ""
        verify_feedback = context.get("verify_feedback", "")

        # Extract the specific source files the planner identified
        files_to_modify = plan.get("files_to_modify", [])
        if isinstance(files_to_modify, list):
            target_files = [f if isinstance(f, str) else f.get("path", "") for f in files_to_modify]
            target_files = [f for f in target_files if f]
        else:
            target_files = []

        parts = [
            f"## Bug Fix Task",
            f"**Issue #{context.get('issue_number', '?')}:** {context.get('issue_title', '')}",
            f"\n**Description:**\n{issue_body_safe}",
        ]

        if target_files:
            parts.append(
                f"\n## Files to Modify\n"
                f"The planner identified these source files as needing changes:\n"
                + "\n".join(f"- `{f}`" for f in target_files)
                + "\n\nRead these files first, understand the bug, then apply your fix."
            )

        parts.append(f"\n## Full Plan\n```json\n{json.dumps(plan, indent=2)}\n```")

        if reproducer_file and reproducer_test:
            parts.append(
                f"\n## Reproducer Test\n"
                f"File: `{reproducer_file}` — **READ-ONLY, do not modify this file.**\n"
                f"```python\n{reproducer_test[:3000]}\n```\n"
                f"After writing your fix, verify it passes:\n"
                f"  pip install -e . --quiet && pytest {reproducer_file} -x --tb=short"
            )

        if verify_feedback:
            parts.append(
                f"\n## Previous Attempt Feedback\n{verify_feedback}\nFix these issues."
            )

        parts.append(
            "\n\n## Rules\n"
            "1. Only edit source files — never test files, never the reproducer, never docs.\n"
            "2. Make the smallest change that fixes the bug.\n"
            "3. Verify the reproducer passes before finishing."
        )
        return "\n".join(parts)

    # ── Git helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_changed_files(repo_root: Path) -> list[str]:
        """Return relative paths of files modified or created since the last commit."""
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )
        files: list[str] = []
        for line in result.stdout.splitlines():
            if len(line) < 3:
                continue
            status = line[:2]
            path = line[3:].strip()
            if any(c in status for c in ("M", "A", "?", "R")):
                # Renames: "old -> new"
                if " -> " in path:
                    path = path.split(" -> ")[-1]
                files.append(path)
        return files

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self._claude_available():
            logger.warning("ClaudeCodeCoderAgent: `claude` not found — falling back to AgenticCoderAgent")
            from phoenixgithub.agents.agentic_coder import AgenticCoderAgent
            return AgenticCoderAgent(self.llm).run(context)

        repo_root = Path(context["clone_path"]).resolve()
        prompt = self._build_prompt(context)

        # Build subprocess environment
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)  # allow nested invocation inside a Claude Code session
        if self._api_key:
            env["ANTHROPIC_API_KEY"] = self._api_key
        if self._base_url:
            env["ANTHROPIC_BASE_URL"] = self._base_url.rstrip("/")

        cmd = [
            "claude", "--print",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--model", self._cli_model,
            prompt,
        ]

        logger.info(
            f"ClaudeCodeCoderAgent: running claude CLI in {repo_root} (model={self._cli_model})"
        )

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=360,
            )
            if proc.returncode not in (0, 1):
                logger.warning(
                    f"ClaudeCodeCoderAgent: claude exited {proc.returncode}\n"
                    f"stderr: {proc.stderr[:500]}"
                )
        except subprocess.TimeoutExpired:
            logger.error("ClaudeCodeCoderAgent: timed out after 360s — capturing partial changes")
        except Exception as e:
            logger.error(f"ClaudeCodeCoderAgent: subprocess error: {e} — falling back")
            from phoenixgithub.agents.agentic_coder import AgenticCoderAgent
            return AgenticCoderAgent(self.llm).run(context)

        # Collect changed files — skip test files and the reproducer
        reproducer_file = context.get("reproducer_file") or ""
        changed_paths = self._get_changed_files(repo_root)
        changes: list[dict[str, str]] = []
        for rel_path in changed_paths:
            name = Path(rel_path).name
            if name == reproducer_file:
                logger.info(f"ClaudeCodeCoderAgent: skipping reproducer file {rel_path}")
                continue
            if name.startswith("test_") or name.endswith("_test.py"):
                logger.info(f"ClaudeCodeCoderAgent: skipping test file {rel_path}")
                continue
            full = repo_root / rel_path
            if full.is_file():
                content = full.read_text(errors="replace")
                changes.append({"file_path": rel_path, "action": "modify", "content": content})

        logger.info(f"ClaudeCodeCoderAgent: {len(changes)} source file(s) changed")

        if not changes:
            logger.warning(
                "ClaudeCodeCoderAgent: 0 source files changed (claude only touched tests/reproducer) "
                "— falling back to AgenticCoderAgent"
            )
            from phoenixgithub.agents.agentic_coder import AgenticCoderAgent
            return AgenticCoderAgent(self.llm).run(context)

        return {
            "changes": changes,
            "applied_files": [c["file_path"] for c in changes],
            "commit_message": f"fix: resolve #{context.get('issue_number', '?')}",
            "agentic": True,
        }
