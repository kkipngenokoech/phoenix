"""Failure analyst agent — diagnoses failed runs and suggests concrete fixes.

Role: analysis (read-only). Does not modify code directly.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from phoenixgithub.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class FailureAnalystAgent(BaseAgent):
    role = "failure_analyst"
    system_prompt = """You are a senior debugging engineer.
Given issue context and a failed run summary, identify likely root causes and
produce specific actionable fixes for the next implementation attempt.

Respond with valid JSON:
{
  "summary": "One-sentence diagnosis",
  "root_cause": "Most likely root cause",
  "suggested_fixes": [
    "Concrete fix step 1",
    "Concrete fix step 2"
  ],
  "confidence": "low | medium | high"
}

Rules:
- Be specific and technical.
- Prioritize the most likely cause from actual failure evidence.
- Avoid vague advice.
- Respond ONLY with JSON, no markdown fences.
"""

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        issue_number = context.get("issue_number")
        issue_title = context.get("issue_title", "")
        issue_body = context.get("issue_body", "")
        run_summary = context.get("run_summary", "")
        test_feedback = context.get("test_feedback", "")
        test_output = context.get("test_output", {})

        prompt = (
            f"## Issue\n"
            f"Number: {issue_number}\n"
            f"Title: {issue_title}\n"
            f"Body: {self._sanitize_body_for_waf(issue_body, max_chars=800)}\n\n"
            f"## Failed Run Summary\n{run_summary[:12000]}\n\n"
            f"## Test Feedback\n{test_feedback[:4000]}\n\n"
            f"## Test Output (raw)\n```json\n{json.dumps(test_output, indent=2)[:12000]}\n```\n\n"
            "Provide diagnostic JSON."
        )

        repo_tag = str(context.get("repo", "unknown")).replace("/", "__")
        run_tag = f"run:{context.get('run_id', 'unknown')}"
        issue_tag = f"issue:{issue_number or 'unknown'}"
        raw = self.invoke(
            prompt,
            trace_name="failure_analyst.diagnose",
            trace_tags=["phoenixgithub", "failure", "diagnostics", f"repo:{repo_tag}", issue_tag, run_tag],
            trace_metadata={
                "agent": self.role,
                "run_id": context.get("run_id"),
                "issue_number": issue_number,
                "repo": context.get("repo"),
                "step": "failure_analysis",
            },
        )
        try:
            parsed = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
            fixes = parsed.get("suggested_fixes", [])
            if not isinstance(fixes, list):
                fixes = [str(fixes)]
            return {
                "summary": parsed.get("summary", "Run failed during automated validation."),
                "root_cause": parsed.get("root_cause", "Unknown"),
                "suggested_fixes": [str(f) for f in fixes][:8],
                "confidence": parsed.get("confidence", "medium"),
            }
        except json.JSONDecodeError:
            logger.warning("Failure analyst returned invalid JSON")
            return {
                "summary": "Run failed during automated validation.",
                "root_cause": "Could not parse analysis output.",
                "suggested_fixes": [
                    "Review failing test output and fix the first concrete assertion/import error.",
                    "Keep file/module paths consistent with test imports.",
                ],
                "confidence": "low",
            }
