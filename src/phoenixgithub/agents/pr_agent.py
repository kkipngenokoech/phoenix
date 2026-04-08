"""PR agent — creates a pull request with a summary of all changes.

Role: pr (read + GitHub CLI). Cannot modify code.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from phoenixgithub.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class PRAgent(BaseAgent):
    role = "pr"
    system_prompt = """You are a technical writer. Given the issue, implementation plan, and
test results, produce a clean pull request description.

Respond with valid JSON:
{
    "title": "PR title (conventional commits style: feat/fix/refactor: description)",
    "body": "Markdown PR body with ## sections: Summary, Changes, Testing"
}

Rules:
- The title should be concise and follow conventional commits.
- The body should explain WHAT changed and WHY.
- Include a Testing section describing how changes were verified.
- Mention the issue number with Closes #N.
- Respond ONLY with the JSON object, no markdown fences."""

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        issue_number = context.get("issue_number", "?")
        issue_title = context["issue_title"]
        issue_body = self._sanitize_body_for_waf(context["issue_body"], max_chars=800)
        plan = context.get("plan", {})
        test_verdict = context.get("test_verdict", {})
        applied_files = context.get("applied_files", [])

        prompt = (
            f"## Original Issue (#{issue_number})\n"
            f"**Title:** {issue_title}\n"
            f"**Body:** {issue_body}\n\n"
            f"## Implementation Plan\n```json\n{json.dumps(plan, indent=2)}\n```\n\n"
            f"## Files Changed\n{chr(10).join(f'- {f}' for f in applied_files)}\n\n"
            f"## Test Results\n{json.dumps(test_verdict, indent=2) if test_verdict else 'All tests passed.'}\n\n"
            f"Write the PR title and body as JSON."
        )

        repo_tag = str(context.get("repo", "unknown")).replace("/", "__")
        issue_tag = f"issue:{issue_number}"
        run_tag = f"run:{context.get('run_id', 'unknown')}"
        raw = self.invoke(
            prompt,
            trace_name="pr_agent.compose",
            trace_tags=["phoenixgithub", "pr", "compose", f"repo:{repo_tag}", issue_tag, run_tag],
            trace_metadata={
                "agent": self.role,
                "run_id": context.get("run_id"),
                "issue_number": issue_number,
                "repo": context.get("repo"),
                "branch_name": context.get("branch_name"),
                "step": "pr",
                "commit_sha": context.get("commit_sha"),
                "changed_files_count": len(applied_files),
            },
        )

        try:
            pr_info = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            pr_info = {
                "title": f"feat: {issue_title}",
                "body": f"Implements #{issue_number}.\n\n{raw[:1000]}",
            }

        return {
            "pr_title": pr_info.get("title", f"feat: {issue_title}"),
            "pr_body": pr_info.get("body", f"Implements #{issue_number}."),
        }
