"""Reproducer Agent — writes a failing test that proves the issue exists.

Role: test synthesis (read + write test file only).

Protocol:
  1. Read the issue and plan.
  2. Write a minimal pytest test that should FAIL on the current (buggy) code.
  3. Run that test to verify it actually fails.
  4. If it passes (false positive), retry once with corrective feedback.
  5. Return the test code, file path, and reproduction status.

The generated test is written to the repo so the Coder can see exactly what
"fixed" looks like. It is included in the PR as evidence of resolution.

This is the key mechanism that closes the gap between correctness preservation
(CP) and true issue resolution: if the reproducer test passes after the fix,
the issue is resolved — not just preserved.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from phoenixgithub.agents.base import BaseAgent

logger = logging.getLogger(__name__)

_REPRODUCER_SYSTEM_PROMPT = """You are a test engineer specializing in bug reproduction.

Given a GitHub issue and the relevant source code, write a minimal pytest test that:
1. Demonstrates the specific bug or missing behaviour described in the issue
2. FAILS on the current (buggy) code
3. Will PASS once the issue is correctly fixed

Rules:
- One test function named `test_issue_reproduction`
- Import only from the existing package (no new external dependencies)
- Keep it minimal — exercise the exact behaviour the issue describes
- Do NOT mock the core logic under test
- The file must be runnable with: pytest <file_path> -x

Respond with valid JSON only (no markdown fences, no text before or after the object):
{
    "test_code": "import ...\\n\\ndef test_issue_reproduction():\\n    ...",
    "test_file": "path/to/test_file.py",
    "rationale": "One sentence explaining what the test exercises and why it fails"
}"""


def _parse_llm_json(raw: str) -> dict[str, Any]:
    """Extract a single JSON object from model output (handles fences and trailing prose)."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    start = s.find("{")
    if start < 0:
        raise json.JSONDecodeError("No JSON object start", raw, 0)
    obj, _end = json.JSONDecoder().raw_decode(s, start)
    if not isinstance(obj, dict):
        raise json.JSONDecodeError("Root JSON value is not an object", raw, start)
    return obj


class ReproducerAgent(BaseAgent):
    role = "reproducer"
    system_prompt = _REPRODUCER_SYSTEM_PROMPT

    def _repo_layout_hint(self, clone_path: str) -> str:
        """Steer test file placement so pytest can collect imports like the upstream suite."""
        root = Path(clone_path)
        parts: list[str] = []
        if (root / "testing").is_dir() and (root / "src" / "_pytest").is_dir():
            parts.append(
                "This repository is **pytest**: add new tests under `testing/` "
                "(not `tests/` at repo root). Prefer `testing/test_issue_<N>_repro.py`. "
                "Import the code under test the same way neighboring files in `testing/` do."
            )
        elif (root / "tests").is_dir() and not (root / "testing").is_dir():
            parts.append(
                "Place the new test under the existing `tests/` tree unless the project "
                "clearly uses another convention visible in the repo root."
            )
        if not parts:
            return ""
        return "\n## Test file location\n" + " ".join(parts) + "\n"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Attempt to write and verify a failing reproduction test.

        Returns context keys:
            reproducer_test      str | None  — test source code
            reproducer_file      str | None  — relative path in repo
            reproduced           bool        — test failed on base code (confirmed reproduction)
            false_positive       bool        — test passed before fix (agent misunderstood)
            reproducer_skipped   bool        — could not reproduce (non-blocking)
        """
        clone_path = context["clone_path"]
        issue_number = context.get("issue_number", 0)
        issue_title = context.get("issue_title", "")
        issue_body = context.get("issue_body", "")
        plan = context.get("plan", {})
        relevant_code = context.get("relevant_code_for_reproducer", "")

        issue_body_safe = self._sanitize_body_for_waf(issue_body, max_chars=1400)
        layout_hint = self._repo_layout_hint(clone_path)

        user_prompt = (
            f"## GitHub Issue #{issue_number}\n"
            f"**Title:** {issue_title}\n"
            f"**Description:**\n{issue_body_safe}\n\n"
            f"## Implementation Plan\n"
            f"**Approach:** {plan.get('approach', 'N/A')}\n"
            f"**Files to modify:** {', '.join(plan.get('files_to_modify', []))}\n\n"
            f"## Relevant Source Code\n{relevant_code or '(not available)'}\n\n"
            f"{layout_hint}"
            f"Write a pytest test that reproduces this issue. "
            f"The test MUST FAIL on the current code (exit code non-zero when run alone)."
        )

        repo_tag = str(context.get("repo", "unknown")).replace("/", "__")
        issue_tag = f"issue:{issue_number}"
        run_tag = f"run:{context.get('run_id', 'unknown')}"

        max_attempts = 3
        max_parse_retries = 3
        last_test_code: str | None = None
        last_test_file: str | None = None

        for attempt in range(1, max_attempts + 1):
            parse_suffix = ""
            parsed: dict[str, Any] | None = None
            for parse_try in range(1, max_parse_retries + 1):
                raw = self.invoke(
                    user_prompt + parse_suffix,
                    trace_name="reproducer.write",
                    trace_tags=["phoenixgithub", "reproducer", f"repo:{repo_tag}", issue_tag, run_tag],
                    trace_metadata={
                        "agent": self.role,
                        "run_id": context.get("run_id"),
                        "issue_number": issue_number,
                        "repo": context.get("repo"),
                        "attempt": attempt,
                        "parse_try": parse_try,
                    },
                )
                try:
                    parsed = _parse_llm_json(raw)
                    break
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning(
                        "Reproducer attempt %s parse try %s/%s: %s",
                        attempt, parse_try, max_parse_retries, e,
                    )
                    snippet = raw[:450].replace("`", "'")
                    parse_suffix = (
                        "\n\n## Response format\n"
                        "Your last reply was not a single valid JSON object "
                        f"({e}). Output ONLY JSON with keys test_code, test_file, rationale — "
                        "no markdown, no commentary. Start with `{` and end with `}`.\n"
                        f"Snippet of your reply:\n```\n{snippet}\n```\n"
                    )
            if parsed is None:
                user_prompt += (
                    "\n\n## Reminder\n"
                    "Previous response(s) could not be parsed as one JSON object. "
                    "Reply with only: {\"test_code\": \"...\", \"test_file\": \"...\", \"rationale\": \"...\"}\n"
                )
                continue

            test_code: str = parsed.get("test_code", "")
            default_file = (
                f"testing/test_issue_{issue_number}_repro.py"
                if (Path(clone_path) / "testing").is_dir()
                else f"tests/test_issue_{issue_number}_repro.py"
            )
            test_file: str = parsed.get("test_file", default_file)
            rationale: str = parsed.get("rationale", "")

            if not test_code.strip():
                logger.warning("Reproducer returned empty test code")
                break

            last_test_code = test_code
            last_test_file = test_file

            # Write the test file to the repo
            abs_path = Path(clone_path) / test_file
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(test_code)
            logger.info(f"Reproducer: wrote test to {test_file}")

            # Run the test — it should FAIL (reproduction confirmed)
            exit_code, output = self._run_test(clone_path, test_file)

            if exit_code != 0:
                # Test fails as expected — reproduction confirmed
                logger.info(
                    f"Reproducer: test fails on base code (exit {exit_code}) — reproduction confirmed. "
                    f"Rationale: {rationale}"
                )
                return {
                    "reproducer_test": test_code,
                    "reproducer_file": test_file,
                    "reproduced": True,
                    "false_positive": False,
                    "reproducer_skipped": False,
                    "reproducer_rationale": rationale,
                }
            else:
                # Test passes before any fix — false positive
                logger.warning(
                    f"Reproducer attempt {attempt}: test PASSES on base code "
                    f"(false positive) — retrying with feedback"
                )
                # Remove false-positive test so it doesn't confuse the Coder
                abs_path.unlink(missing_ok=True)

                if attempt < max_attempts:
                    user_prompt = (
                        f"{user_prompt}\n\n"
                        f"## Feedback on previous attempt\n"
                        f"Your test `{test_file}` PASSED on the current (buggy) code — "
                        f"which means it is NOT reproducing the bug.\n"
                        f"The test must FAIL on the existing code. "
                        f"Look more carefully at what the issue describes as broken behavior.\n"
                        f"Previous test output:\n```\n{output[:2000]}\n```"
                    )

        # Could not reproduce — non-blocking, pipeline continues without test
        logger.info(
            "Reproducer: could not write a confirmed failing test after "
            f"{max_attempts} attempt(s) — skipping (non-blocking)"
        )
        return {
            "reproducer_test": last_test_code,
            "reproducer_file": last_test_file,
            "reproduced": False,
            "false_positive": last_test_code is not None,  # had code but it passed
            "reproducer_skipped": True,
            "reproducer_rationale": "",
        }

    def _run_test(self, clone_path: str, test_file: str, timeout: int = 90) -> tuple[int, str]:
        """Run a single test file with pytest. Returns (exit_code, output)."""
        from pathlib import Path as _Path
        root = _Path(clone_path)

        # Ensure the package under test is installed so the test can import it.
        # Without this, ImportErrors cause exit_code != 0 for the wrong reason.
        if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
            subprocess.run(
                ["pip", "install", "-e", ".", "--quiet", "--no-build-isolation"],
                cwd=clone_path, capture_output=True, timeout=120,
            )

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_file, "-x", "--tb=short", "-q", "--no-header"],
                cwd=clone_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (result.stdout + result.stderr)[:1500]
            return result.returncode, output
        except subprocess.TimeoutExpired:
            return -1, "Reproducer test timed out"
        except Exception as e:
            return -1, f"Reproducer test error: {e}"
