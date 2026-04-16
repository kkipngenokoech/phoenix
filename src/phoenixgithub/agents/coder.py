"""Coder agent — implements changes according to the plan.

Role: coding (read + write). Can modify files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from phoenixgithub.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    role = "coder"
    system_prompt = """You are an expert software engineer. You receive an implementation plan
and must produce the exact file changes needed.

For each step, respond with valid JSON matching this schema:
{
    "changes": [
        {
            "file_path": "relative/path/to/file.py",
            "action": "modify" | "create" | "patch",
            "content": "the complete new file content"   // for modify/create
            // OR for patch (preferred for large existing files):
            "search": "exact original lines to replace",
            "replace": "new lines to substitute"
        }
    ],
    "commit_message": "feat: concise description of what changed"
}

Rules:
- CRITICAL: Your changes MUST target the files listed in the plan's `files_to_modify`.
  These files are shown in full under "Current File Contents" — read them carefully and
  apply surgical edits to fix the issue. Do NOT ignore these files.
- CRITICAL: Do NOT produce generic boilerplate files (database.py, models.py, config.py,
  utils.py, core/*, etc.) unless they are explicitly listed in the plan. If your output
  contains files unrelated to the plan, you have hallucinated — start over.
- For LARGE existing files (shown as truncated or > ~200 lines): use action "patch" with
  "search" (unique existing lines to replace) and "replace" (the new lines). This avoids
  reproducing the entire file.
- For small files or new files: use "modify" or "create" with "content" (complete file content).
- Write COMPLETE content for modify/create — no placeholders or ellipsis.
- Follow existing code style and conventions.
- Add appropriate imports.
- Do NOT add unnecessary comments explaining the change.
- Handle edge cases.
- Avoid creating duplicate test module names across directories (for example,
  do not create two files both named test_foo.py in different folders).
- Reuse the existing project layout; do not create new top-level package
  directories unless the issue explicitly requires restructuring.
- LANGUAGE RULE: Only create files in the language the project uses. If the project
  is JavaScript/TypeScript, do NOT create .py files. If it is Python, do NOT create .js/.ts files.
- SELF-VERIFICATION: Before finalizing, mentally trace through each test case in your
  test files against your implementation to confirm the implementation returns the
  correct result. If a test would fail, fix the implementation first.
- Respond ONLY with the JSON object, no markdown fences."""

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        clone_path = context["clone_path"]
        plan = context["plan"]
        issue_title = context["issue_title"]
        issue_body = context["issue_body"]
        visual_context = context.get("visual_context", "")
        auto_guidance = context.get("auto_guidance", "")
        revision_notes = context.get("revision_notes", "")
        trigger_label = context.get("trigger_label", "")

        # If we have feedback from a failed verify-reject-retry, include it
        verify_feedback = context.get("verify_feedback", "")
        project_type = context.get("project_type", "")

        files_context = self._read_files_for_plan(clone_path, plan)

        feedback_section = ""
        if verify_feedback:
            feedback_section = (
                f"\n\n## Previous Attempt Failed\n"
                f"The following feedback was received. Fix these issues:\n{verify_feedback}\n"
            )

        project_type_section = f"## Project Type\n{project_type}\n\n" if project_type else ""

        # Strip code blocks / tracebacks to avoid WAF 403s on the CMU AI gateway.
        issue_body_safe = self._sanitize_body_for_waf(issue_body, max_chars=1000)

        prompt = (
            f"## Task\n"
            f"**Issue:** {issue_title}\n"
            f"**Description:** {issue_body_safe}\n\n"
            f"## Trigger Context\n"
            f"Trigger label: {trigger_label}\n"
            f"{'Revise mode: apply minimal targeted fixes only.' if trigger_label == 'ai:revise' else ''}\n\n"
            f"{project_type_section}"
            f"## Revision Directives\n"
            f"{revision_notes or '(none)'}\n\n"
            f"## Screenshot-Derived Context\n"
            f"{visual_context or '(none)'}\n\n"
            f"## Automatic Retry Guidance\n"
            f"{auto_guidance or '(none)'}\n\n"
            f"## Implementation Plan\n```json\n{json.dumps(plan, indent=2)}\n```\n\n"
            f"## Current File Contents\n{files_context}"
            f"{feedback_section}\n\n"
            f"Produce the file changes as JSON."
        )

        trace_meta = {
            "agent": self.role,
            "run_id": context.get("run_id"),
            "issue_number": context.get("issue_number"),
            "repo": context.get("repo"),
            "branch_name": context.get("branch_name"),
            "step": "implement",
            "attempt": context.get("step_attempt"),
        }
        repo_tag = str(context.get("repo", "unknown")).replace("/", "__")
        issue_tag = f"issue:{context.get('issue_number', 'unknown')}"
        run_tag = f"run:{context.get('run_id', 'unknown')}"
        attempt_tag = f"attempt:{context.get('step_attempt', 'unknown')}"
        raw = self.invoke(
            prompt,
            trace_name="coder.implement",
            trace_tags=["phoenixgithub", "coder", "implement", f"repo:{repo_tag}", issue_tag, run_tag, attempt_tag],
            trace_metadata=trace_meta,
        )
        logger.info(f"Coder response length: {len(raw)} chars")

        result = self._parse_coder_json(raw)
        if result is None:
            logger.warning("Coder returned invalid JSON — requesting one repair pass")
            files_to_modify = plan.get("files_to_modify", [])
            files_hint = (
                f"\nCRITICAL: You must ONLY output changes for these files: {files_to_modify}\n"
                f"Do NOT create generic placeholder files (main.py, database.py, models.py, etc.).\n"
            ) if files_to_modify else ""
            repair_prompt = (
                "Your previous response was not valid JSON.\n"
                "Rules:\n"
                "1. Respond ONLY with a JSON object — no markdown, no prose.\n"
                "2. For large files use action 'patch' with 'search'/'replace' instead of "
                "full 'content' — this keeps the response small and valid.\n"
                f"{files_hint}"
                "3. Use this exact schema:\n"
                "{\n"
                '  "changes": [\n'
                '    {"file_path": "path.py", "action": "patch", "search": "old lines", "replace": "new lines"}\n'
                '    // OR for small/new files:\n'
                '    {"file_path": "path.py", "action": "modify|create", "content": "complete content"}\n'
                "  ],\n"
                '  "commit_message": "feat: concise description"\n'
                "}\n\n"
                "Produce the corrected JSON now:"
            )
            repaired_raw = self.invoke(
                repair_prompt,
                trace_name="coder.repair_json",
                trace_tags=["phoenixgithub", "coder", "repair", f"repo:{repo_tag}", issue_tag, run_tag, attempt_tag],
                trace_metadata=trace_meta,
            )
            logger.info(f"Coder repair response length: {len(repaired_raw)} chars")
            result = self._parse_coder_json(repaired_raw)

        if result is None:
            logger.error(f"Coder returned invalid JSON after repair:\n{raw[:500]}")
            return {"changes": [], "commit_message": "failed to parse coder output", "error": raw[:1000]}

        # Apply changes to disk
        changes = result.get("changes", [])
        applied: list[str] = []
        repo_root = Path(clone_path).resolve()
        created_dirs: set[Path] = set()

        # Snapshot which top-level directories already exist BEFORE any writes.
        # We only enforce the README guardrail on dirs that are genuinely new.
        pre_existing_top_level = {
            d for d in repo_root.iterdir() if d.is_dir()
        }

        def _new_ancestor_dirs(path: Path) -> list[Path]:
            missing: list[Path] = []
            current = path
            while current != repo_root and not current.exists():
                missing.append(current)
                current = current.parent
            return missing

        for change in changes:
            file_path = change.get("file_path", "")
            action = change.get("action", "modify")
            content = change.get("content", "")

            if not file_path:
                continue
            if action != "patch" and not content:
                continue

            full_path = (repo_root / file_path).resolve()
            try:
                full_path.relative_to(repo_root)
            except ValueError:
                logger.warning(f"Skipped unsafe file path outside repo root: {file_path}")
                continue

            # Skip .github/workflows/ files — GitHub App lacks `workflows` permission
            # to push workflow files, which causes the git push to be rejected.
            try:
                rel_check = full_path.relative_to(repo_root)
                if rel_check.parts[:2] == (".github", "workflows"):
                    logger.warning(f"Skipped workflow file (no permission to push): {file_path}")
                    continue
            except ValueError:
                pass

            for d in _new_ancestor_dirs(full_path.parent):
                created_dirs.add(d)

            full_path.parent.mkdir(parents=True, exist_ok=True)

            if action == "patch":
                search = change.get("search", "")
                replace = change.get("replace", "")
                if not search:
                    logger.warning(f"Patch action missing 'search' field for {file_path} — skipping")
                    continue
                if full_path.exists():
                    original = full_path.read_text(errors="replace")
                    if search in original:
                        patched = original.replace(search, replace, 1)
                        full_path.write_text(patched)
                        applied.append(file_path)
                        logger.info(f"Patched: {file_path} ({len(search)} → {len(replace)} chars)")
                    else:
                        # Fuzzy fallback: normalize trailing whitespace per line
                        def _norm(s: str) -> str:
                            return "\n".join(line.rstrip() for line in s.splitlines())

                        norm_orig = _norm(original)
                        norm_search = _norm(search)
                        norm_replace = _norm(replace)
                        if norm_search in norm_orig:
                            idx = norm_orig.index(norm_search)
                            # Map index back to original (character offset may differ slightly)
                            # Rebuild by replacing the normalized block
                            patched = norm_orig.replace(norm_search, norm_replace, 1)
                            full_path.write_text(patched)
                            applied.append(file_path)
                            logger.info(f"Patched (fuzzy): {file_path} ({len(search)} → {len(replace)} chars)")
                        else:
                            logger.warning(f"Patch search string not found in {file_path} — skipping")
                else:
                    logger.warning(f"Patch target does not exist: {file_path} — skipping")
            else:
                full_path.write_text(content)
                applied.append(file_path)
                logger.info(f"Wrote: {file_path} ({len(content)} chars)")

        # README guardrail: only enforce when the coder creates a genuinely NEW
        # top-level folder (depth == 1 AND it didn't exist before our writes).
        readme_violations: list[str] = []
        for folder in sorted(created_dirs):
            rel = folder.relative_to(repo_root)
            # Only require README for new top-level dirs (depth==1)
            if len(rel.parts) != 1:
                continue
            # Skip dirs that already existed before this run
            if folder in pre_existing_top_level:
                continue
            readme_path = folder / "README.md"
            if not readme_path.exists():
                readme_violations.append(f"{rel} (missing README.md)")
                continue
            readme_len = len(readme_path.read_text(errors="replace").strip())
            if readme_len < 200:
                readme_violations.append(
                    f"{rel} (README.md too short: {readme_len} chars)"
                )

        if readme_violations:
            logger.warning(
                "README guardrail: new folder(s) missing README — "
                + "; ".join(readme_violations)
                + " (proceeding anyway for eval)"
            )

        return {
            "changes": changes,
            "applied_files": applied,
            "commit_message": result.get("commit_message", f"feat: implement #{context.get('issue_number', '?')}"),
        }

    def _parse_coder_json(self, raw: str) -> dict[str, Any] | None:
        candidates: list[str] = []
        text = raw.strip()
        candidates.append(text)

        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                candidates.append("\n".join(lines[1:-1]).strip())

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1].strip())

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    # Show primary file up to this many chars. With the patch action, the coder
    # outputs only the changed section (~1-3k chars), so we can afford to show
    # more context. If the fix is deep in the file, truncation causes the coder
    # to write search strings for code it hasn't seen (from training memory).
    _PRIMARY_FILE_LIMIT = 20000

    def _read_files_for_plan(self, clone_path: str, plan: dict) -> str:
        """Read files referenced in the plan so the coder has targeted context."""
        modify_files = plan.get("files_to_modify", [])
        create_files = plan.get("files_to_create", [])
        root = Path(clone_path)
        chunks: list[str] = []

        # Show the primary modify target in full (up to limit); rest as name-only
        for i, rel_path in enumerate(modify_files):
            full = root / rel_path
            if full.exists():
                try:
                    content = full.read_text(errors="replace")
                    if i == 0:
                        # Primary target — show content up to limit
                        if len(content) > self._PRIMARY_FILE_LIMIT:
                            content = content[: self._PRIMARY_FILE_LIMIT] + f"\n... ({len(content)} chars total)"
                        chunks.append(f"### {rel_path} (existing)\n```\n{content}\n```")
                    else:
                        # Secondary targets — name + size hint only to avoid token explosion
                        chunks.append(
                            f"### {rel_path} (existing, {len(content)} chars — "
                            f"secondary target; modify only if the plan explicitly requires it)"
                        )
                except Exception:
                    chunks.append(f"### {rel_path}\n(could not read)")
            else:
                chunks.append(f"### {rel_path}\n(new file — does not exist yet)")

        for rel_path in create_files:
            full = root / rel_path
            if full.exists():
                try:
                    content = full.read_text(errors="replace")
                    chunks.append(f"### {rel_path} (existing)\n```\n{content}\n```")
                except Exception:
                    chunks.append(f"### {rel_path}\n(could not read)")
            else:
                chunks.append(f"### {rel_path}\n(new file — does not exist yet)")

        return "\n\n".join(chunks) if chunks else "(no files to show)"
