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
    system_prompt = """You are an expert software engineer. Produce surgical search/replace edits.

## Output Format (preferred)

Use search/replace blocks — one per logical change:

FILE: relative/path/to/file.py
<<<<<<< SEARCH
exact lines from the file (copy verbatim, whitespace matters)
=======
replacement lines
>>>>>>> REPLACE

Rules for search/replace blocks:
- SEARCH must be an exact verbatim copy of lines from the file shown in "Current File Contents".
  Copy the lines character-for-character including leading spaces/tabs.
- Keep SEARCH blocks small and focused — just enough context to uniquely identify the location.
- One FILE: header per file; multiple SEARCH/REPLACE pairs in the same file are allowed.
- For a NEW file, use an empty SEARCH block (nothing between SEARCH and =======).
- End each block with >>>>>>> REPLACE on its own line.

## Fallback Format (if blocks won't work)

If you cannot use blocks, respond with JSON:
{
    "changes": [{"file_path": "path.py", "action": "modify", "content": "<complete file>"}],
    "commit_message": "fix: description"
}

## General Rules
- CRITICAL: Only touch files listed in the plan's `files_to_modify`. Do NOT create generic
  boilerplate files (database.py, models.py, config.py, utils.py, core/*) unless in the plan.
- Follow existing code style and conventions. Add appropriate imports.
- Do NOT add unnecessary comments. Handle edge cases.
- LANGUAGE RULE: Only create files in the language the project uses.
- SELF-VERIFICATION: Mentally trace through the fix — would the failing test now pass?"""

    # ── Diagnose → Fix → Verify pipeline ────────────────────────────────────

    def _diagnose(
        self,
        issue_title: str,
        issue_body_safe: str,
        files_context: str,
        reproducer_test: str,
        reproducer_file: str,
        trace_tags: list[str],
        trace_meta: dict,
    ) -> str:
        """Call 1 — understand the root cause before writing any code."""
        repro_hint = (
            f"\n## Reproducer Test (currently FAILING)\n"
            f"File: `{reproducer_file}`\n"
            f"```python\n{reproducer_test[:2000]}\n```\n"
            if reproducer_test else ""
        )
        prompt = (
            f"## Bug to Diagnose\n"
            f"**Issue:** {issue_title}\n"
            f"**Description:** {issue_body_safe}\n\n"
            f"## Relevant Source Code\n{files_context}\n"
            f"{repro_hint}\n"
            f"## Task\n"
            f"Identify in 3-5 sentences:\n"
            f"1. The exact root cause — which line/condition/function is wrong and why\n"
            f"2. What the correct behavior should be\n"
            f"3. The minimal change required (plain English, no code)\n"
        )
        diagnosis = self.invoke(
            prompt,
            trace_name="coder.diagnose",
            trace_tags=trace_tags,
            trace_metadata=trace_meta,
        )
        # Hard cap — diagnosis is context for call 2, not a full essay
        diagnosis = diagnosis[:1500]
        logger.info(f"Bug diagnosis ({len(diagnosis)} chars): {diagnosis[:150]}")
        return diagnosis

    def _verify_and_correct(
        self,
        changes: list[dict],
        reproducer_test: str,
        reproducer_file: str,
        trace_tags: list[str],
        trace_meta: dict,
    ) -> list[dict]:
        """Call 3 — mentally simulate the reproducer; return corrected changes if needed."""
        if not reproducer_test or not changes:
            return changes

        changes_text = "\n\n".join(
            f"### {c['file_path']}\n```python\n{(c.get('content') or '')[:3000]}\n```"
            for c in changes
            if c.get("content")
        )
        if not changes_text:
            return changes

        prompt = (
            f"## Proposed Fix\n{changes_text}\n\n"
            f"## Reproducer Test (must PASS after fix)\n"
            f"File: `{reproducer_file}`\n"
            f"```python\n{reproducer_test[:2000]}\n```\n\n"
            f"## Task\n"
            f"Mentally trace through the reproducer test with the proposed fix applied.\n"
            f"Step through the logic line by line. Does the test pass?\n\n"
            f"Respond ONLY with JSON (no markdown):\n"
            f'{{"passes": true, "issue": null, "corrected_changes": null}}\n'
            f"OR if the test still fails:\n"
            f'{{"passes": false, "issue": "what is still wrong", '
            f'"corrected_changes": [{{"file_path": "...", "action": "modify", "content": "complete corrected file"}}]}}'
        )
        raw = self.invoke(
            prompt,
            trace_name="coder.verify",
            trace_tags=trace_tags,
            trace_metadata=trace_meta,
        )
        try:
            result = self._parse_coder_json(raw)
            if not isinstance(result, dict):
                return changes
            if result.get("passes"):
                logger.info("Coder self-verify: fix passes ✓")
                return changes
            issue = result.get("issue", "")
            corrected = result.get("corrected_changes")
            logger.info(f"Coder self-verify: fix fails — {issue}")
            if corrected and isinstance(corrected, list) and corrected:
                logger.info(f"Coder self-verify: applying {len(corrected)} corrected file(s)")
                return corrected
        except Exception as e:
            logger.warning(f"Coder self-verify parse error: {e}")
        return changes

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        clone_path = context["clone_path"]
        plan = context["plan"]
        issue_title = context["issue_title"]
        issue_body = context["issue_body"]
        visual_context = context.get("visual_context", "")
        auto_guidance = context.get("auto_guidance", "")
        revision_notes = context.get("revision_notes", "")
        trigger_label = context.get("trigger_label", "")
        verify_feedback = context.get("verify_feedback", "")
        current_diff = context.get("current_diff", "")
        project_type = context.get("project_type", "")
        reproducer_file = context.get("reproducer_file") or ""
        reproducer_test = context.get("reproducer_test") or ""

        files_context = self._read_files_for_plan(clone_path, plan)
        issue_body_safe = self._sanitize_body_for_waf(issue_body, max_chars=2500)

        repo_tag = str(context.get("repo", "unknown")).replace("/", "__")
        issue_tag = f"issue:{context.get('issue_number', 'unknown')}"
        run_tag = f"run:{context.get('run_id', 'unknown')}"
        attempt_tag = f"attempt:{context.get('step_attempt', 'unknown')}"
        trace_tags = ["phoenixgithub", "coder", f"repo:{repo_tag}", issue_tag, run_tag, attempt_tag]
        trace_meta = {
            "agent": self.role,
            "run_id": context.get("run_id"),
            "issue_number": context.get("issue_number"),
            "repo": context.get("repo"),
            "branch_name": context.get("branch_name"),
            "step": "implement",
            "attempt": context.get("step_attempt"),
        }

        # ── Call 1: Diagnose the root cause ──────────────────────────────────
        diagnosis = self._diagnose(
            issue_title, issue_body_safe, files_context,
            reproducer_test, reproducer_file, trace_tags, trace_meta,
        )

        # ── Call 2: Generate the fix ──────────────────────────────────────────
        feedback_section = (
            f"\n\n## Previous Attempt Failed\n"
            f"The following feedback was received. Fix these issues:\n{verify_feedback}\n"
            if verify_feedback else ""
        )
        diff_section = (
            f"\n## Your Previous Changes (git diff HEAD)\n"
            f"This is exactly what you changed last attempt — review before making new edits:\n"
            f"```diff\n{current_diff}\n```\n"
            if current_diff else ""
        )
        project_type_section = f"## Project Type\n{project_type}\n\n" if project_type else ""
        reproducer_section = (
            f"\n## Reproducer Test (must PASS after your fix)\n"
            f"File: `{reproducer_file}`\n"
            f"```python\n{reproducer_test[:3000]}\n```\n"
            f"Use this test to understand exactly which function/behavior is broken.\n"
            if reproducer_file and reproducer_test else ""
        )

        prompt = (
            f"## Task\n"
            f"**Issue:** {issue_title}\n"
            f"**Description:** {issue_body_safe}\n\n"
            f"## Root Cause Analysis\n{diagnosis}\n\n"
            f"## Trigger Context\n"
            f"Trigger label: {trigger_label}\n"
            f"{'Revise mode: apply minimal targeted fixes only.' if trigger_label == 'ai:revise' else ''}\n\n"
            f"{project_type_section}"
            f"## Revision Directives\n{revision_notes or '(none)'}\n\n"
            f"## Screenshot-Derived Context\n{visual_context or '(none)'}\n\n"
            f"## Automatic Retry Guidance\n{auto_guidance or '(none)'}\n\n"
            f"## Implementation Plan\n```json\n{json.dumps(plan, indent=2)}\n```\n\n"
            f"{reproducer_section}"
            f"## Current File Contents\n{files_context}"
            f"{diff_section}"
            f"{feedback_section}\n\n"
            f"Produce search/replace blocks (or JSON fallback) for the fix."
        )

        raw = self.invoke(
            prompt,
            trace_name="coder.implement",
            trace_tags=trace_tags,
            trace_metadata=trace_meta,
        )
        logger.info(f"Coder response length: {len(raw)} chars")

        # ── Parse: prefer Aider-style search/replace blocks, fall back to JSON ──
        sr_blocks = self._parse_search_replace_blocks(raw)
        if sr_blocks:
            logger.info("Coder: parsed %d search/replace block(s)", len(sr_blocks))
            result = {
                "changes": [
                    {"file_path": b["file_path"], "action": "patch",
                     "search": b["search"], "replace": b["replace"]}
                    for b in sr_blocks
                ],
                "commit_message": f"fix: implement #{context.get('issue_number', '?')}",
            }
        else:
            result = self._parse_coder_json(raw)

        if result is None:
            logger.warning("Coder returned neither SR blocks nor valid JSON — requesting repair")
            files_to_modify = plan.get("files_to_modify", [])
            files_hint = (
                f"\nCRITICAL: You must ONLY output changes for these files: {files_to_modify}\n"
                f"Do NOT create generic placeholder files (main.py, database.py, models.py, etc.).\n"
            ) if files_to_modify else ""
            repair_prompt = (
                "Your previous response could not be parsed. Produce search/replace blocks:\n\n"
                "FILE: path/to/file.py\n"
                "<<<<<<< SEARCH\n"
                "exact lines to replace\n"
                "=======\n"
                "new lines\n"
                ">>>>>>> REPLACE\n\n"
                "Or fall back to JSON:\n"
                "{\n"
                '  "changes": [{"file_path": "path.py", "action": "modify", "content": "complete file"}],\n'
                '  "commit_message": "fix: description"\n'
                "}\n"
                f"{files_hint}\n"
                "Produce the fix now:"
            )
            repaired_raw = self.invoke(
                repair_prompt,
                trace_name="coder.repair_json",
                trace_tags=trace_tags,
                trace_metadata=trace_meta,
            )
            logger.info(f"Coder repair response length: {len(repaired_raw)} chars")
            sr_blocks = self._parse_search_replace_blocks(repaired_raw)
            if sr_blocks:
                result = {
                    "changes": [
                        {"file_path": b["file_path"], "action": "patch",
                         "search": b["search"], "replace": b["replace"]}
                        for b in sr_blocks
                    ],
                    "commit_message": f"fix: implement #{context.get('issue_number', '?')}",
                }
            else:
                result = self._parse_coder_json(repaired_raw)

        if result is None:
            logger.error(f"Coder returned invalid output after repair:\n{raw[:500]}")
            return {"changes": [], "commit_message": "failed to parse coder output", "error": raw[:1000]}

        # ── Call 3: Self-verify and correct (JSON whole-file path only) ───────
        changes = result.get("changes", [])
        # Skip self-verify for patch/SR format — surgical blocks are self-contained
        has_sr = any(c.get("action") == "patch" for c in changes)
        if not has_sr:
            changes = self._verify_and_correct(
                changes, reproducer_test, reproducer_file, trace_tags, trace_meta,
            )
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
                        # Fuzzy fallback 1: normalize all whitespace per line
                        def _norm(s: str) -> str:
                            return "\n".join(line.strip() for line in s.splitlines())

                        norm_orig = _norm(original)
                        norm_search = _norm(search)
                        norm_replace = _norm(replace)
                        if norm_search in norm_orig:
                            patched = norm_orig.replace(norm_search, norm_replace, 1)
                            full_path.write_text(patched)
                            applied.append(file_path)
                            logger.info(f"Patched (fuzzy-ws): {file_path} ({len(search)} → {len(replace)} chars)")
                        else:
                            # Fuzzy fallback 2: match by stripping blank lines too
                            def _norm2(s: str) -> str:
                                return "\n".join(
                                    line.strip() for line in s.splitlines() if line.strip()
                                )

                            norm2_orig = _norm2(original)
                            norm2_search = _norm2(search)
                            if norm2_search and norm2_search in norm2_orig:
                                # Apply replace on the whitespace-normalized original
                                patched = norm_orig.replace(norm_search, norm_replace, 1) if norm_search in norm_orig else norm2_orig.replace(norm2_search, _norm2(replace), 1)
                                full_path.write_text(patched)
                                applied.append(file_path)
                                logger.info(f"Patched (fuzzy-blank): {file_path}")
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

    def _parse_search_replace_blocks(self, raw: str) -> list[dict[str, str]]:
        """Parse Aider-style FILE/SEARCH/REPLACE blocks from LLM output.

        Returns list of {file_path, search, replace} dicts, or [] if none found.
        """
        blocks: list[dict[str, str]] = []
        current_file: str = ""
        search_lines: list[str] = []
        replace_lines: list[str] = []
        state = "outside"  # outside | in_search | in_replace

        for line in raw.splitlines():
            stripped = line.strip()

            if stripped.upper().startswith("FILE:"):
                # Flush any open block before switching files
                if state == "in_replace" and current_file:
                    blocks.append({
                        "file_path": current_file,
                        "search": "\n".join(search_lines),
                        "replace": "\n".join(replace_lines),
                    })
                current_file = stripped[5:].strip().strip("`").strip()
                search_lines, replace_lines = [], []
                state = "outside"

            elif stripped in ("<<<<<<< SEARCH", "<<<<<<<SEARCH", "<<<<<<< search"):
                search_lines, replace_lines = [], []
                state = "in_search"

            elif stripped in ("=======", "======"):
                if state == "in_search":
                    state = "in_replace"

            elif stripped in (">>>>>>> REPLACE", ">>>>>>>REPLACE", ">>>>>>> replace"):
                if state == "in_replace" and current_file:
                    blocks.append({
                        "file_path": current_file,
                        "search": "\n".join(search_lines),
                        "replace": "\n".join(replace_lines),
                    })
                search_lines, replace_lines = [], []
                state = "outside"

            elif state == "in_search":
                search_lines.append(line)

            elif state == "in_replace":
                replace_lines.append(line)

        return blocks

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
                        # Secondary targets — show content up to a smaller limit
                        _SECONDARY_LIMIT = 8000
                        preview = content if len(content) <= _SECONDARY_LIMIT else content[:_SECONDARY_LIMIT] + f"\n... ({len(content)} chars total)"
                        chunks.append(f"### {rel_path} (existing — secondary target)\n```\n{preview}\n```")
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
