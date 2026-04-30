"""Planner agent — reads issue + codebase, produces implementation plan.

Role: analysis (read-only). Cannot modify code.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

_STOP_WORDS = {
    "about", "after", "before", "being", "below", "could", "fixed",
    "given", "issue", "other", "right", "should", "since", "their",
    "there", "these", "those", "using", "value", "where", "which",
    "while", "would", "error", "false", "true", "null", "none",
    "return", "import", "class", "function", "method", "object",
}

_NAVIGATION_SYSTEM_PROMPT = """You are a code navigator. Your sole job is to find the \
source files in a repository that are relevant to a GitHub issue.

Use the tools to explore — list directories, search for symbols, read files. \
Navigate like a developer would:
1. Scan the top-level structure to understand the repo layout.
2. Search for identifiers, error messages, or class names mentioned in the issue.
3. Read the most promising files to confirm relevance.
4. Stop once you have read 2–4 relevant files.

Do NOT write any code. Do NOT suggest fixes. Just navigate and read."""

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
- CRITICAL: Every path in "files_to_modify" MUST already exist in the repository
  file tree shown in the prompt. Do NOT invent generic paths like src/core/config.py,
  src/core/utils.py, src/core/models.py, or any path not visible in the tree.
  If you cannot identify the correct existing file, set "files_to_modify": [] and
  describe the uncertainty in "approach" — do not guess.
- "files_to_modify" should contain SOURCE files that implement the broken behavior,
  not test files. Fix the code that the tests exercise, not the tests themselves.
- "files_to_create" is only for genuinely new files (e.g. a new test file alongside
  an existing one). Never create a substitute for a file that already exists.
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
            else self._agentic_localize(clone_path, issue_title, issue_body)
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
            logger.warning("Planner returned invalid JSON — requesting one repair pass")
            repair_prompt = (
                f"Your previous response was not valid JSON. Here is what you produced:\n\n"
                f"{raw[:2000]}\n\n"
                "Respond ONLY with a valid JSON object matching this exact schema — "
                "no markdown fences, no prose:\n"
                "{\n"
                '  "summary": "One-sentence summary",\n'
                '  "approach": "High-level approach",\n'
                '  "files_to_modify": ["path/to/file.py"],\n'
                '  "files_to_create": [],\n'
                '  "steps": [{"step_id": 1, "description": "...", "target_file": "...", "action": "modify"}],\n'
                '  "test_strategy": "How to verify",\n'
                '  "risk_level": "low"\n'
                "}\n\n"
                "Produce the corrected JSON now:"
            )
            repaired_raw = self.invoke(
                repair_prompt,
                trace_name="planner.repair_json",
                trace_tags=["phoenixgithub", "planner", "repair_json", f"repo:{repo_tag}", issue_tag, run_tag],
                trace_metadata=trace_meta,
            )
            logger.info(f"Planner repair response length: {len(repaired_raw)} chars")
            try:
                plan = json.loads(repaired_raw.strip().removeprefix("```json").removesuffix("```").strip())
                logger.info("Planner JSON repair succeeded")
            except json.JSONDecodeError:
                logger.error(f"Planner returned invalid JSON after repair:\n{raw[:500]}")
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

    def _agentic_localize(
        self,
        root: str,
        issue_title: str,
        issue_body: str,
        max_steps: int = 12,
        max_chars_per_file: int = 3000,
    ) -> str:
        """Navigate the repository with tool calls to find relevant files.

        Gives the LLM four read-only tools (list_directory, read_file,
        search_code, find_files) and lets it explore the repo interactively —
        the same approach used by top SWE-bench systems.  Falls back to
        keyword-scoring if tool calling is unsupported or nothing is found.

        Returns a formatted excerpts string compatible with _read_relevant_files.
        """
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.tools import tool as lc_tool

        root_path = Path(root).resolve()
        skip_dirs = {
            "node_modules", ".venv", "venv", ".git", "__pycache__",
            "dist", "build", ".next", ".nuxt",
        }
        read_files: dict[str, str] = {}   # rel_path → full content

        # ── safety helper ──────────────────────────────────────────────────
        def _safe(p: str) -> Path | None:
            try:
                resolved = (root_path / p.lstrip("/")).resolve()
                resolved.relative_to(root_path)
                return resolved
            except Exception:
                return None

        # ── tool definitions ───────────────────────────────────────────────
        @lc_tool
        def list_directory(path: str = ".") -> str:
            """List the contents of a directory in the repository. Use '.' for root."""
            target = _safe(path)
            if not target or not target.is_dir():
                return f"Directory not found: {path}"
            entries: list[str] = []
            try:
                for e in sorted(target.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
                    if e.name in skip_dirs:
                        continue
                    entries.append(f"  {'[dir]  ' if e.is_dir() else '[file] '}{e.name}")
            except PermissionError:
                return "Permission denied"
            rel = str(target.relative_to(root_path)) or "."
            return f"{rel}/\n" + "\n".join(entries[:80])

        @lc_tool
        def read_file(path: str, start_line: int = 1, end_line: int = 120) -> str:
            """Read a source file from the repository, with an optional line range."""
            target = _safe(path)
            if not target or not target.is_file():
                return f"File not found: {path}"
            try:
                text = target.read_text(errors="replace")
                lines = text.splitlines()
                sl = max(0, start_line - 1)
                el = min(len(lines), end_line)
                excerpt = "\n".join(
                    f"{sl + i + 1:4}: {line}" for i, line in enumerate(lines[sl:el])
                )
                rel = str(target.relative_to(root_path))
                # Store full content for the plan prompt
                read_files[rel] = text[:max_chars_per_file * 2]
                return f"### {rel} (lines {sl+1}–{el} of {len(lines)})\n```\n{excerpt}\n```"
            except Exception as e:
                return f"Error reading {path}: {e}"

        @lc_tool
        def search_code(pattern: str, extension: str = "") -> str:
            """Grep the codebase for a pattern. extension: file extension without dot, e.g. 'py' or 'ts'."""
            exts = [f"*.{extension}"] if extension else ["*.py", "*.ts", "*.tsx", "*.js", "*.jsx"]
            include_flags = [f"--include={e}" for e in exts]
            cmd = ["grep", "-rn", "-i"] + include_flags + [pattern, str(root_path)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
                lines = [
                    ln for ln in result.stdout.splitlines()
                    if not any(sd in ln for sd in skip_dirs)
                ][:25]
                if not lines:
                    return f"No matches for '{pattern}'"
                out: list[str] = []
                for ln in lines:
                    parts = ln.split(":", 1)
                    if parts:
                        try:
                            rel = str(Path(parts[0]).resolve().relative_to(root_path))
                            out.append(rel + (":" + parts[1] if len(parts) > 1 else ""))
                        except ValueError:
                            out.append(ln)
                return "\n".join(out)
            except Exception as e:
                return f"Search error: {e}"

        @lc_tool
        def find_files(name_pattern: str) -> str:
            """Find files whose name contains name_pattern (e.g. 'auth', 'config', 'renderer.ts')."""
            glob = name_pattern if "*" in name_pattern else f"*{name_pattern}*"
            matches: list[str] = []
            for f in root_path.rglob(glob):
                if f.is_file() and not any(sd in f.parts for sd in skip_dirs):
                    try:
                        matches.append(str(f.relative_to(root_path)))
                    except ValueError:
                        pass
            if not matches:
                return f"No files matching '{name_pattern}'"
            return "\n".join(sorted(matches)[:30])

        # ── navigation loop ────────────────────────────────────────────────
        tools = [list_directory, read_file, search_code, find_files]
        tool_map = {t.name: t for t in tools}

        try:
            llm_with_tools = self.llm.bind_tools(tools)
        except Exception as e:
            logger.warning(f"Planner: bind_tools failed ({e}) — falling back to keyword scoring")
            return self._read_relevant_files(
                root, issue_hint=f"{issue_title} {issue_body[:200]}"
            )

        from langchain_core.messages import SystemMessage as SM, HumanMessage as HM

        issue_hint = self._sanitize_body_for_waf(issue_body, max_chars=1500)

        # BM25 pre-ranking: give the navigator a head start instead of navigating blind
        bm25_ranked = self._bm25_rank_files(root, issue_title, issue_body)
        if bm25_ranked:
            bm25_hint = (
                "**BM25 pre-ranking (start here — highest keyword relevance to the issue):**\n"
                + "\n".join(f"  {i+1}. {p}  (score: {s:.2f})" for i, (p, s) in enumerate(bm25_ranked))
                + "\nRead these files first before exploring elsewhere.\n\n"
            )
            logger.info("BM25 pre-ranking: %s", [p for p, _ in bm25_ranked])
        else:
            bm25_hint = ""

        messages: list = [
            SM(content=_NAVIGATION_SYSTEM_PROMPT),
            HM(content=(
                f"Repository: {root_path.name}/\n\n"
                f"{bm25_hint}"
                f"Issue title: {issue_title}\n"
                f"Issue description:\n{issue_hint}\n\n"
                "Navigate the repository to find the 2–4 most relevant source files that "
                "need to be changed to fix this issue. You may also read existing test files "
                "to understand what correct behavior is expected. Search for the class/function "
                "names mentioned in the issue."
            )),
        ]

        for step in range(max_steps):
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.warning(f"Planner nav step {step} error: {e}")
                break

            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                logger.info(
                    "Planner navigator finished in %d step(s), %d file(s) read",
                    step + 1, len(read_files),
                )
                break

            for tc in tool_calls:
                fn = tool_map.get(tc["name"])
                try:
                    result = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tc['name']}"
                except Exception as e:
                    result = f"Tool error: {e}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        # ── format output ──────────────────────────────────────────────────
        if not read_files:
            logger.info("Agentic localize: no files read — falling back to keyword scoring")
            return self._read_relevant_files(
                root, issue_hint=f"{issue_title} {issue_body[:200]}"
            )

        chunks: list[str] = []
        for rel, content in list(read_files.items())[:5]:
            if len(content) > max_chars_per_file:
                content = content[:max_chars_per_file] + "\n... (truncated)"
            chunks.append(f"### {rel}\n```\n{content}\n```")

        logger.info("Planner agentic localize: returning %d file(s)", len(chunks))
        return "<!-- Found via agentic navigation -->\n\n" + "\n\n".join(chunks)

    def _bm25_rank_files(
        self,
        root: str,
        issue_title: str,
        issue_body: str,
        top_n: int = 8,
    ) -> list[tuple[str, float]]:
        """Score all code files by BM25 relevance to the issue. Returns (rel_path, score) sorted desc."""
        import re as _re
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return []
        try:
            root_path = Path(root)
            skip_dirs = {"node_modules", ".venv", "venv", ".git", "__pycache__", "dist", "build", ".next"}
            code_exts = {".py", ".ts", ".js", ".tsx", ".jsx"}

            files: list[Path] = []
            for f in root_path.rglob("*"):
                if f.suffix not in code_exts:
                    continue
                if any(sd in f.parts for sd in skip_dirs):
                    continue
                files.append(f)

            if not files:
                return []

            def _tokenize(text: str) -> list[str]:
                tokens = _re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower())
                return [t for t in tokens if t not in _STOP_WORDS]

            # Build corpus: path tokens + first 200 lines of content
            corpus: list[list[str]] = []
            for f in files:
                try:
                    lines = f.read_text(errors="replace").splitlines()[:200]
                    path_tokens = _tokenize(str(f.relative_to(root_path)))
                    content_tokens = _tokenize(" ".join(lines))
                    corpus.append(path_tokens + content_tokens)
                except Exception:
                    corpus.append([])

            # Build query: issue text + extracted identifiers (error classes, function names)
            combined = issue_title + " " + issue_body
            identifiers = _re.findall(r"\b[A-Z][a-zA-Z]+(?:Error|Exception|Warning)\b", combined)
            func_names = _re.findall(r"def ([a-z_][a-z0-9_]+)", combined)
            module_names = _re.findall(r"(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]+)", combined)
            query_text = combined + " " + " ".join(identifiers + func_names + module_names)
            query = _tokenize(query_text)

            if not query:
                return []

            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query)

            ranked = sorted(
                ((str(files[i].relative_to(root_path)), float(scores[i])) for i in range(len(files))),
                key=lambda x: x[1],
                reverse=True,
            )
            return [(p, s) for p, s in ranked[:top_n] if s > 0]
        except Exception as e:
            logger.warning("BM25 ranking failed: %s", e)
            return []

    def _grep_for_keywords(self, root: str, keywords: list[str]) -> dict[str, int]:
        """Return {relative_path: keyword_hit_count} by grepping the repo.

        Used as a content-based fallback when path-name scoring finds no
        confident candidates.  Files that contain more issue keywords rank higher.
        """
        root_path = Path(root)
        code_exts = {".py", ".js", ".ts", ".tsx", ".jsx"}
        skip_dirs = {"node_modules", ".venv", "venv", ".git", "__pycache__"}
        hits: dict[str, int] = {}

        for keyword in keywords[:8]:
            try:
                result = subprocess.run(
                    [
                        "grep", "-rl",
                        "--include=*.py", "--include=*.js",
                        "--include=*.ts", "--include=*.tsx", "--include=*.jsx",
                        "-i", keyword, str(root_path),
                    ],
                    capture_output=True, text=True, timeout=8,
                )
                for line in result.stdout.splitlines():
                    p = Path(line.strip())
                    if not p.is_file() or p.suffix not in code_exts:
                        continue
                    if any(sd in p.parts for sd in skip_dirs):
                        continue
                    try:
                        rel = str(p.relative_to(root_path))
                        hits[rel] = hits.get(rel, 0) + 1
                    except ValueError:
                        pass
            except Exception:
                continue

        return hits

    def _read_relevant_files(
        self,
        root: str,
        max_files: int = 5,
        max_chars_per_file: int = 1200,
        issue_hint: str = "",
    ) -> str:
        """Read Python/JS/TS source files most relevant to the issue hint.

        Two-stage ranking:
        1. Path-name scoring: file stem/name matches issue keywords → fast.
        2. Content-grep fallback: when path scores are all weak (≤ depth bonus
           only), grep the repo for issue keywords and re-rank by content hits.
           This handles issues where the relevant file has a generic name
           (e.g. ``runtime-core.ts`` for a Vue SSR bug).

        Java files are skipped entirely — the CMU AI gateway WAF rejects their
        annotation-heavy syntax.  The planner uses the file tree + issue text
        for Java repos instead.
        """
        root_path = Path(root)
        code_exts = {".py", ".js", ".ts", ".tsx", ".jsx"}
        skip_dirs = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "docs", "doc", "examples", "example", "tests", "test",
            "scripts", "script", "fixtures", "spec", "bench", "benchmarks",
        }
        skip_stems = {"cli", "setup", "conftest", "__main__", "manage", "wsgi", "asgi"}

        hint_lower = issue_hint.lower()
        hint_words = [
            w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{3,}", hint_lower)
            if w not in _STOP_WORDS
        ]

        def _path_score(f: Path) -> int:
            """Score based on file path/stem matching issue keywords."""
            stem = f.stem.lower()
            score = 0
            if hint_words:
                if stem in hint_lower:
                    score += 10
                for word in hint_words:
                    if word in stem:
                        score += 3
            score += min(len(f.parts), 5)
            return score

        candidates = [
            f for f in root_path.rglob("*")
            if f.is_file()
            and f.suffix in code_exts
            and not any(sd in f.parts for sd in skip_dirs)
            and f.stem not in skip_stems
            and not f.stem.startswith("test_")
            and not f.stem.endswith("_test")
        ]

        # For large repos reduce excerpt size to stay within gateway limits.
        if len(candidates) > 500:
            max_files = min(max_files, 3)
            max_chars_per_file = min(max_chars_per_file, 800)

        candidates.sort(key=lambda f: (-_path_score(f), str(f)))

        # --- Content-grep fallback ---
        # If the best path score is ≤ 5 (only depth bonus, no keyword match),
        # run a grep-based search and re-rank candidates by content hits.
        used_grep = False
        top_path_score = _path_score(candidates[0]) if candidates else 0
        if top_path_score <= 5 and hint_words:
            grep_hits = self._grep_for_keywords(root, hint_words)
            if grep_hits:
                used_grep = True
                def _combined_score(f: Path) -> int:
                    rel = str(f.relative_to(root_path))
                    return _path_score(f) + grep_hits.get(rel, 0) * 5
                candidates.sort(key=lambda f: (-_combined_score(f), str(f)))
                logger.info(
                    "Planner: low path-score (%d); grep fallback found %d matching files",
                    top_path_score, len(grep_hits),
                )

        header_note = (
            "Found via content search (grep) — these files contain issue keywords."
            if used_grep
            else "Modify these files; do not recreate them from scratch."
        )

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

        if not chunks:
            return "(no source files found)"
        return f"<!-- {header_note} -->\n\n" + "\n\n".join(chunks)
