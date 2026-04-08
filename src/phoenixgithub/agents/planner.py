"""Planner agent — reads issue + codebase, produces implementation plan.

Role: analysis (read-only). Cannot modify code.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from phoenixgithub.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    role = "planner"
    system_prompt = """You are a senior software architect. Your job is to read a GitHub issue
and the relevant codebase, then produce a concrete implementation plan.

You MUST respond with valid JSON matching this schema:
{
    "summary": "One-sentence summary of the change",
    "approach": "High-level approach description",
    "files_to_modify": ["path/to/file1.py", "path/to/file2.py"],
    "files_to_create": ["path/to/new_file.py"],
    "steps": [
        {
            "step_id": 1,
            "description": "What to do",
            "target_file": "path/to/file.py",
            "action": "modify or create"
        }
    ],
    "test_strategy": "How to verify the changes work",
    "risk_level": "low | medium | high"
}

Rules:
- Be specific about file paths (relative to repo root).
- Keep steps ordered by dependency — things that must happen first go first.
- Consider edge cases and backwards compatibility.
- Only include files that actually need changes.
- Respond ONLY with the JSON object, no markdown fences."""

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        clone_path = context["clone_path"]
        issue_title = context["issue_title"]
        issue_body = context["issue_body"]
        issue_image_paths: list[str] = context.get("issue_image_paths", [])
        issue_image_urls: list[str] = context.get("issue_image_urls", [])
        revision_notes = context.get("revision_notes", "")
        trigger_label = context.get("trigger_label", "")
        issue_comments: list[dict[str, str]] = context.get("issue_comments", [])

        file_tree = self._scan_tree(clone_path)
        project_type = self._detect_project_type(clone_path)

        no_code = context.get("planner_no_code", False)
        relevant_code = (
            "(code context omitted to avoid gateway content filter)"
            if no_code
            else self._read_relevant_files(
                clone_path, issue_hint=f"{issue_title} {issue_body[:200]}"
            )
        )
        visual_context = self._analyze_screenshots(
            issue_title,
            issue_body,
            issue_image_paths,
            repo=context.get("repo"),
            issue_number=context.get("issue_number"),
            run_id=context.get("run_id"),
        )
        image_urls_text = "\n".join(f"- {u}" for u in issue_image_urls) if issue_image_urls else "(none)"
        comments_text = "\n".join(
            f"- @{c.get('author', 'unknown')}: {self._sanitize_body_for_waf(c.get('body') or '', max_chars=150)}"
            for c in issue_comments[-8:]
        ) or "(none)"
        revise_instruction = ""
        if trigger_label == "ai:revise":
            revise_instruction = (
                "You are in revise mode. Prioritize the latest human feedback and "
                "apply the smallest targeted change set needed to resolve it. "
                "Do not redesign unrelated parts."
            )

        # Strip code blocks + truncate to avoid WAF content-filter triggers.
        # Large issues (bug reports with stack traces, code blocks) regularly
        # contain patterns (tracebacks, JSON payloads, file paths) that cause
        # the CMU AI gateway to return 403 Blocked.
        issue_body_truncated = self._sanitize_body_for_waf(
            issue_body, max_chars=500 if no_code else 1500
        )

        prompt = (
            f"## GitHub Issue\n"
            f"**Title:** {issue_title}\n"
            f"**Description:**\n{issue_body_truncated}\n\n"
            f"## Trigger Context\n"
            f"Trigger label: {trigger_label}\n"
            f"{revise_instruction}\n\n"
            f"## Project Type\n{project_type}\n\n"
            f"## Issue Discussion (recent comments)\n{comments_text}\n\n"
            f"## Revision Directives\n{revision_notes or '(none)'}\n\n"
            f"## Visual Context Extracted From Screenshots\n{visual_context}\n\n"
            f"## Repository Structure\n```\n{file_tree}\n```\n\n"
            f"## Key Source File Excerpts (modify these, do not recreate from scratch)\n{relevant_code}\n\n"
            f"Produce the implementation plan as JSON."
        )

        trace_meta = {
            "agent": self.role,
            "run_id": context.get("run_id"),
            "issue_number": context.get("issue_number"),
            "repo": context.get("repo"),
            "branch_name": context.get("branch_name"),
            "step": "plan",
            "image_count": len(issue_image_paths),
            "image_url_count": len(issue_image_urls),
        }
        repo_tag = str(context.get("repo", "unknown")).replace("/", "__")
        issue_tag = f"issue:{context.get('issue_number', 'unknown')}"
        run_tag = f"run:{context.get('run_id', 'unknown')}"
        raw = self.invoke(
            prompt,
            trace_name="planner.plan",
            trace_tags=["phoenixgithub", "planner", "plan", f"repo:{repo_tag}", issue_tag, run_tag],
            trace_metadata=trace_meta,
        )
        logger.info(f"Planner response length: {len(raw)} chars")

        try:
            plan = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            logger.error(f"Planner returned invalid JSON:\n{raw[:500]}")
            plan = {
                "summary": issue_title,
                "approach": raw[:1000],
                "files_to_modify": [],
                "files_to_create": [],
                "steps": [],
                "test_strategy": "manual",
                "risk_level": "medium",
            }

        return {"plan": plan, "visual_context": visual_context, "project_type": project_type}

    def _detect_project_type(self, root: str) -> str:
        """Return a short description of the project's language and package structure."""
        p = Path(root)
        lines: list[str] = []

        if (p / "package.json").exists():
            try:
                pkg = __import__("json").loads((p / "package.json").read_text())
                name = pkg.get("name", "")
                lang = pkg.get("scripts", {})
                ts = (p / "tsconfig.json").exists() or any(p.rglob("*.ts"))
                lines.append(f"Language: {'TypeScript' if ts else 'JavaScript'} (Node.js)")
                if name:
                    lines.append(f"Package name: {name}")
                lines.append("IMPORTANT: This is a JavaScript/TypeScript project. Do NOT create Python files.")
                lines.append("All implementation files must be .js or .ts. Tests use the existing JS/TS test framework.")
            except Exception:
                lines.append("Language: JavaScript/TypeScript (Node.js)")
                lines.append("IMPORTANT: Do NOT create Python files in this repo.")
        elif (p / "pom.xml").exists() or (p / "build.gradle").exists():
            lines.append("Language: Java")
            lines.append("IMPORTANT: This is a Java project. Do NOT create Python or JS files.")
        else:
            # Python
            pyproject = p / "pyproject.toml"
            setup_py = p / "setup.py"
            pkg_name = ""
            if pyproject.exists():
                try:
                    text = pyproject.read_text()
                    import re
                    m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', text)
                    if m:
                        pkg_name = m.group(1)
                except Exception:
                    pass
            lines.append(f"Language: Python")
            if pkg_name:
                lines.append(f"Package name: {pkg_name}")
            # Detect source layout
            src = p / "src"
            if src.exists():
                subdirs = [d.name for d in src.iterdir() if d.is_dir() and not d.name.startswith(".")]
                if subdirs:
                    lines.append(f"Source layout: src/{subdirs[0]}/ (src-layout)")
                    lines.append(f"IMPORTANT: Source files live under src/{subdirs[0]}/, NOT a top-level src/ or package/.")
            else:
                # Flat layout — find main package dir
                pkgs = [d.name for d in p.iterdir()
                        if d.is_dir() and (d / "__init__.py").exists()
                        and d.name not in ("tests", "test", "docs", ".git")]
                if pkgs:
                    lines.append(f"Source layout: flat — main package is {pkgs[0]}/")

        return "\n".join(lines)

    def _analyze_screenshots(
        self,
        issue_title: str,
        issue_body: str,
        image_paths: list[str],
        *,
        repo: str | None,
        issue_number: int | None,
        run_id: str | None,
    ) -> str:
        if not image_paths:
            return "(no screenshots provided)"

        try:
            prompt = (
                "Analyze the attached screenshots from a GitHub issue.\n\n"
                f"Issue title: {issue_title}\n"
                f"Issue body: {issue_body[:2000]}\n\n"
                "Return concise plain text with:\n"
                "1) Visible UI/state facts\n"
                "2) Errors/messages shown\n"
                "3) Concrete implementation requirements implied by the screenshots\n"
                "4) Any ambiguity that still needs clarification\n"
            )
            analysis = self.invoke_with_images(
                prompt,
                image_paths[:6],
                trace_name="planner.vision",
                trace_tags=[
                    "phoenixgithub",
                    "planner",
                    "vision",
                    f"repo:{str(repo or 'unknown').replace('/', '__')}",
                    f"issue:{issue_number or 'unknown'}",
                    f"run:{run_id or 'unknown'}",
                ],
                trace_metadata={
                    "agent": self.role,
                    "step": "plan_vision",
                    "image_count": len(image_paths[:6]),
                },
            )
            return analysis.strip()[:8000]
        except Exception as e:
            logger.warning(f"Screenshot analysis failed: {e}")
            return "(screenshot analysis unavailable)"

    def _scan_tree(self, root: str, max_depth: int = 3) -> str:
        lines: list[str] = []
        root_path = Path(root)
        skip = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox", ".mypy_cache"}

        def walk(p: Path, depth: int, prefix: str = "") -> None:
            if depth > max_depth:
                return
            entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
            for i, entry in enumerate(entries):
                if entry.name in skip:
                    continue
                connector = "└── " if i == len(entries) - 1 else "├── "
                lines.append(f"{prefix}{connector}{entry.name}")
                if entry.is_dir():
                    ext = "    " if i == len(entries) - 1 else "│   "
                    walk(entry, depth + 1, prefix + ext)

        lines.append(root_path.name + "/")
        walk(root_path, 0)
        return "\n".join(lines[:200])

    def _read_relevant_files(
        self,
        root: str,
        max_files: int = 5,
        max_chars_per_file: int = 1200,
        issue_hint: str = "",
    ) -> str:
        """Read Python/JS/TS source files from the main package only.

        Skips docs/, examples/, tests/, scripts/ directories and entry-point
        files (cli.py, __main__.py, setup.py) that rarely contain the logic
        relevant to bug fixes and tend to have docstrings/comments that trigger
        WAF rules on the CMU AI gateway.

        Files are ranked by relevance to the issue hint (title + description)
        before alphabetical fallback, so the most useful context comes first.
        """
        root_path = Path(root)
        # Java files trigger the CMU AI gateway WAF even at small sizes due to
        # verbose annotations and import statements. Skip them entirely and let
        # the planner rely on the file tree + issue text instead.
        code_exts = {".py", ".js", ".ts", ".tsx", ".jsx"}
        skip_dirs = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "docs", "doc", "examples", "example", "tests", "test",
            "scripts", "script", "fixtures", "spec", "bench", "benchmarks",
        }
        # Entry-point / boilerplate files — skip to avoid WAF-triggering content
        skip_stems = {"cli", "setup", "conftest", "__main__", "manage", "wsgi", "asgi"}

        hint_lower = issue_hint.lower()

        def _score(f: Path) -> int:
            """Higher score = more relevant (picked first)."""
            stem = f.stem.lower()
            name = f.name.lower()
            score = 0
            if hint_lower:
                if stem in hint_lower or name in hint_lower:
                    score += 10
                # Check if any word in the hint appears in the stem
                for word in hint_lower.split():
                    if len(word) > 3 and word in stem:
                        score += 3
            # Prefer files deeper in the tree (more likely to be domain logic)
            score += min(len(f.parts), 5)
            return score

        # For large repos (>500 source files) reduce per-file limit further
        source_file_count = sum(1 for _ in root_path.rglob("*") if _.suffix in code_exts)
        if source_file_count > 500:
            max_files = min(max_files, 3)
            max_chars_per_file = min(max_chars_per_file, 800)

        candidates = [
            f for f in root_path.rglob("*")
            if f.is_file()
            and f.suffix in code_exts
            and not any(sd in f.parts for sd in skip_dirs)
            and f.stem not in skip_stems
            and not f.stem.startswith("test_")
            and not f.stem.endswith("_test")
        ]
        # Sort: highest relevance score first, then alphabetical as tiebreaker
        candidates.sort(key=lambda f: (-_score(f), str(f)))

        chunks: list[str] = []
        for f in candidates:
            if len(chunks) >= max_files:
                break
            try:
                content = f.read_text(errors="replace")
                rel = f.relative_to(root_path)
                if len(content) > max_chars_per_file:
                    content = content[:max_chars_per_file] + "\n... (truncated)"
                chunks.append(f"### {rel}\n```\n{content}\n```")
            except Exception:
                continue

        return "\n\n".join(chunks) if chunks else "(no source files found)"
