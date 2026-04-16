"""Orchestrator — runs the full pipeline: plan → implement → test → PR.

The orchestrator never implements code itself. It only dispatches to agents
and manages the verify-reject-retry loop.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any

from phoenixgithub.agents.coder import CoderAgent
from phoenixgithub.agents.failure_analyst import FailureAnalystAgent
from phoenixgithub.agents.planner import PlannerAgent
from phoenixgithub.agents.pr_agent import PRAgent
from phoenixgithub.agents.reproducer import ReproducerAgent
from phoenixgithub.agents.tester import TesterAgent
from phoenixgithub.config import Config
from phoenixgithub.github_client import GitHubClient
from phoenixgithub.models import Run, RunStatus, StepID
from phoenixgithub.provider import create_llm
from phoenixgithub.state import StateManager

logger = logging.getLogger(__name__)
BOT_COMMENT_MARKERS = (
    "🤖 **Phoenix AI** picked up this issue.",
    "📋 **Plan ready**",
    "❌ **Phoenix AI** failed to complete this issue.",
    "✅ **Phoenix AI** created a PR for review.",
    "### Phoenix Failure Analysis",
)


class Orchestrator:
    """Executes a run: PLAN → IMPLEMENT → TEST → PR with retry loop."""

    def __init__(self, config: Config, github: GitHubClient, state: StateManager, *, webhook_mode: bool = False) -> None:
        self.config = config
        self.github = github
        self.webhook_mode = webhook_mode
        self.state = state
        # Single local clone/worktree per repo means runs must execute serially
        # to avoid git checkout/reset collisions across threads.
        self._execute_lock = threading.Lock()

        llm = create_llm(config.llm)
        self.planner = PlannerAgent(llm)
        self.reproducer = ReproducerAgent(llm)
        self.coder = CoderAgent(llm)
        self.tester = TesterAgent(
            llm,
            test_command=config.agent.test_command,
            allow_no_tests=config.agent.allow_no_tests,
            validation_profile=config.agent.validation_profile,
            resolution_mode=config.agent.resolution_mode,
        )
        self.pr_agent = PRAgent(llm)
        self.failure_analyst = FailureAnalystAgent(llm)

    def execute(self, run: Run) -> Run:
        """Execute the full pipeline for a run. Returns updated run."""
        with self._execute_lock:
            run.status = RunStatus.RUNNING
            self.state.save_run(run)

            issue_number = run.issues[0]
            issue = self.github.get_issue(issue_number)

            context: dict[str, Any] = {
                "run_id": run.run_id,
                "repo": run.repo,
                "issue_number": issue_number,
                "issue_title": issue.title,
                "issue_body": issue.body or "",
                "branch_name": run.branch_name,
                "trigger_label": run.context.get("trigger_label", self.config.labels.ready),
            }
            issue_comments = self.github.get_issue_comments(issue_number, limit=40)
            context["issue_comments"] = issue_comments
            context["revision_notes"] = self._derive_revision_notes(issue_comments)

            try:
                incremental_revise = (
                    self.config.agent.revise_incremental
                    and context.get("trigger_label") == self.config.labels.revise
                )
                full_reset = not incremental_revise

                # 1. Clone and branch
                clone_path = self.github.ensure_clone(self.config.workspace_dir, full_reset=full_reset)
                repo = self.github.create_branch(clone_path, run.branch_name, full_reset=full_reset)
                context["clone_path"] = clone_path
                if incremental_revise:
                    logger.info(f"[{run.run_id}] Incremental revise mode enabled (no full branch reset)")
                image_urls = self.github.get_issue_image_urls(issue_number)
                context["issue_image_urls"] = image_urls
                if image_urls:
                    image_dir = f"{self.config.workspace_dir}/runs/{run.run_id}/issue_images"
                    image_paths = self.github.download_issue_images(image_urls, image_dir)
                    context["issue_image_paths"] = image_paths
                    logger.info(
                        f"[{run.run_id}] Downloaded {len(image_paths)}/{len(image_urls)} issue screenshot(s)"
                    )
                else:
                    context["issue_image_paths"] = []

                # 2. PLAN
                run = self._step_plan(run, context)
                if run.status == RunStatus.FAILED:
                    return self._finalize_failure(run, issue_number, context)

                # 3. REPRODUCE (non-blocking — failure does not stop the pipeline)
                if self.config.agent.use_reproducer:
                    run = self._step_reproduce(run, context)

                # 4. IMPLEMENT + TEST (with retry loop)
                run = self._step_implement_and_test(run, context)
                if run.status == RunStatus.FAILED:
                    return self._finalize_failure(run, issue_number, context)

                # 4. Commit & Push
                commit_msg = context.get("commit_message", f"feat: implement #{issue_number}")
                sha = self.github.commit_and_push(
                    clone_path, run.branch_name, commit_msg, context.get("applied_files")
                )
                context["commit_sha"] = sha
                logger.info(f"Committed: {sha[:8]}")

                # 5. CREATE PR
                run = self._step_pr(run, context)
                if run.status == RunStatus.FAILED:
                    return self._finalize_failure(run, issue_number, context)

                # 6. Success
                run.status = RunStatus.SUCCEEDED
                self.state.save_run(run)
                self.state.mark_run_finished(run.run_id)

                self.github.transition_label(
                    issue_number,
                    self.config.labels.in_progress,
                    self.config.labels.review,
                )
                self.github.comment_on_issue(
                    issue_number,
                    f"✅ **Phoenix AI** created a PR for review.\n\n"
                    f"**PR:** {run.pr_url}\n"
                    f"**Branch:** `{run.branch_name}`\n\n"
                    f"Please review and merge when ready.",
                )

                logger.info(f"Run {run.run_id} succeeded — PR: {run.pr_url}")
                return run

            except Exception as e:
                logger.error(f"Run {run.run_id} failed: {e}", exc_info=True)
                run.status = RunStatus.FAILED
                run.error = str(e)
                self.state.save_run(run)
                self._write_run_log(run, context, error=e)
                return self._finalize_failure(run, issue_number, context)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _step_plan(self, run: Run, context: dict) -> Run:
        logger.info(f"[{run.run_id}] PLAN — analyzing issue #{context['issue_number']}")
        run.set_step_running(StepID.PLAN)
        self.state.save_run(run)

        import time as _time
        # Warm-up ping to wake the gateway before the real plan request
        try:
            self.planner.llm.invoke("ping")
        except Exception:
            pass
        _time.sleep(3)

        max_plan_attempts = 5
        for plan_attempt in range(1, max_plan_attempts + 1):
            try:
                outputs = self.planner.run(context)
                context.update(outputs)

                # Validate files_to_modify — strip paths that don't exist in the repo.
                # The LLM sometimes hallucinates paths despite being told not to.
                plan = outputs.get("plan", {})
                clone_path = context.get("clone_path", "")
                if clone_path and plan:
                    from pathlib import Path as _Path
                    valid, phantom = [], []
                    for p in plan.get("files_to_modify", []):
                        if (_Path(clone_path) / p).exists():
                            valid.append(p)
                        else:
                            phantom.append(p)
                    if phantom:
                        logger.warning(
                            "[%s] Planner hallucinated %d path(s) not in repo: %s — removing",
                            run.run_id, len(phantom), phantom,
                        )
                        plan["files_to_modify"] = valid
                        outputs["plan"] = plan
                        context["plan"] = plan

                run.set_step_done(StepID.PLAN, outputs)

                plan = outputs.get("plan", {})
                self.github.comment_on_issue(
                    context["issue_number"],
                    f"📋 **Plan ready**\n\n"
                    f"**Approach:** {plan.get('approach', 'N/A')}\n"
                    f"**Files:** {', '.join(plan.get('files_to_modify', []))}\n"
                    f"**Risk:** {plan.get('risk_level', 'unknown')}",
                )
                break  # success
            except Exception as e:
                is_rate_limit = any(
                    code in str(e) for code in ("403", "429", "rate limit", "Forbidden")
                )
                if is_rate_limit and plan_attempt < max_plan_attempts:
                    # WAF content blocks don't improve with longer waits.
                    # Cap at 30s and strip code context on second attempt.
                    wait = min(30 * plan_attempt, 60)
                    logger.warning(
                        f"[{run.run_id}] Plan attempt {plan_attempt} hit gateway limit "
                        f"({e}). Retrying in {wait}s (no-code mode)..."
                    )
                    _time.sleep(wait)
                    # Tell planner to skip source file excerpts on next attempt
                    context["planner_no_code"] = True
                    continue
                run.set_step_failed(StepID.PLAN, str(e))
                run.status = RunStatus.FAILED
                run.error = f"Plan failed: {e}"
                break

        self.state.save_run(run)
        return run

    def _read_plan_files_for_reproducer(self, context: dict[str, Any]) -> str:
        """Load source excerpts for files in the plan — planner never returns `relevant_code`."""
        from pathlib import Path

        clone_path = context.get("clone_path", "")
        plan = context.get("plan") or {}
        paths = [p for p in (plan.get("files_to_modify") or []) if p]
        if not clone_path or not paths:
            return ""

        root = Path(clone_path)
        chunks: list[str] = []
        max_per_file = 20_000
        max_total = 100_000
        total = 0

        for rel in paths[:8]:
            path = root / rel
            if not path.is_file():
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            excerpt = text[:max_per_file]
            if len(text) > max_per_file:
                excerpt += "\n... (truncated)"
            block = f"### {rel}\n```\n{excerpt}\n```"
            if total + len(block) > max_total:
                break
            chunks.append(block)
            total += len(block)

        return "\n\n".join(chunks)

    def _step_reproduce(self, run: Run, context: dict) -> Run:
        """Write and verify a failing reproduction test (non-blocking step).

        On success: adds reproducer_test, reproducer_file, reproduced=True to context.
        On failure: sets reproducer_skipped=True and continues — does NOT fail the run.
        """
        logger.info(f"[{run.run_id}] REPRODUCE — writing reproduction test for #{context['issue_number']}")
        run.set_step_running(StepID.REPRODUCE)
        self.state.save_run(run)

        # Planner builds excerpts internally but does not put them in context;
        # read the plan's files_to_modify from disk so the reproducer sees real source.
        disk = self._read_plan_files_for_reproducer(context)
        legacy = (context.get("relevant_code") or "").strip()
        context["relevant_code_for_reproducer"] = disk or legacy

        try:
            outputs = self.reproducer.run(context)
            context.update(outputs)

            reproduced = outputs.get("reproduced", False)
            skipped = outputs.get("reproducer_skipped", False)
            test_file = outputs.get("reproducer_file", "")

            run.set_step_done(StepID.REPRODUCE, {
                "reproduced": reproduced,
                "skipped": skipped,
                "test_file": test_file,
                "false_positive": outputs.get("false_positive", False),
            })

            if reproduced:
                self.github.comment_on_issue(
                    context["issue_number"],
                    f"🔬 **Reproduction test written**\n\n"
                    f"Phoenix confirmed the issue is reproducible via `{test_file}`.\n"
                    f"The fix will be verified against this test.",
                )
                logger.info(f"[{run.run_id}] Reproduction confirmed: {test_file}")
            else:
                logger.info(
                    f"[{run.run_id}] Reproduction skipped (non-blocking) — "
                    f"continuing without reproducer test"
                )

        except Exception as e:
            logger.warning(f"[{run.run_id}] Reproducer step error (non-blocking): {e}")
            context["reproducer_skipped"] = True
            run.set_step_done(StepID.REPRODUCE, {"skipped": True, "error": str(e)})

        self.state.save_run(run)
        return run  # always continue — reproducer failure is non-blocking

    def _step_implement_and_test(self, run: Run, context: dict) -> Run:
        """Implement + test with verify-reject-retry loop."""
        max_retries = self.config.agent.max_retries
        all_applied_files: set[str] = set()

        for attempt in range(1, max_retries + 1):
            context["step_attempt"] = attempt
            logger.info(f"[{run.run_id}] IMPLEMENT (attempt {attempt}/{max_retries})")
            run.set_step_running(StepID.IMPLEMENT)
            self.state.save_run(run)

            try:
                coder_outputs = self.coder.run(context)
                context.update(coder_outputs)

                # Detect hallucinated output: coder ignored plan's files_to_modify
                # and produced unrelated generic files (database.py, models.py, etc.)
                from pathlib import Path as _Path
                clone_path = context.get("clone_path", "")
                plan_files = set(context.get("plan", {}).get("files_to_modify", []))
                applied = coder_outputs.get("applied_files", [])
                changes = coder_outputs.get("changes", [])

                if plan_files and clone_path:
                    # plan_all = files the coder is allowed to touch
                    plan_all = plan_files | set(context.get("plan", {}).get("files_to_create", []))
                    # Check if coder ignored the plan entirely (no overlap at all)
                    overlap = set(applied) & plan_all
                    if not overlap:
                        # Strip phantom changes — use plan membership only, NOT disk existence,
                        # because the coder already wrote these files to disk before we check.
                        valid_changes, phantom_changes = [], []
                        for ch in changes:
                            fp = ch.get("file_path", "")
                            if fp in plan_all:
                                valid_changes.append(ch)
                            else:
                                phantom_changes.append(fp)
                                # Delete phantom file from disk so it doesn't land in the PR
                                try:
                                    phantom_path = _Path(clone_path) / fp
                                    if phantom_path.exists():
                                        phantom_path.unlink()
                                except Exception:
                                    pass

                        if phantom_changes:
                            logger.warning(
                                "[%s] Coder hallucinated %d file(s) not in plan: %s",
                                run.run_id, len(phantom_changes), phantom_changes[:5],
                            )
                            coder_outputs["changes"] = valid_changes
                            coder_outputs["applied_files"] = [
                                ch["file_path"] for ch in valid_changes if ch.get("file_path")
                            ]
                            context.update(coder_outputs)

                        # If still no overlap with plan's modify-list, inject retry feedback
                        remaining_applied = set(coder_outputs.get("applied_files", []))
                        if not (remaining_applied & plan_files) and attempt < max_retries:
                            feedback = (
                                f"CRITICAL: Your previous attempt did not modify any of the required files.\n"
                                f"You MUST modify these existing files: {', '.join(sorted(plan_files))}\n"
                                f"The full contents of each file are shown under 'Current File Contents'.\n"
                                f"Make targeted surgical edits to the existing code to fix the issue.\n"
                                f"Do NOT create new placeholder files like main.py, database.py, models.py, or utils.py."
                            )
                            context["verify_feedback"] = feedback
                            logger.warning(
                                "[%s] Coder ignored all plan files — injecting retry feedback (attempt %d)",
                                run.run_id, attempt,
                            )
                            run.step(StepID.IMPLEMENT).retries += 1
                            continue  # skip tester, retry coder with explicit feedback

                all_applied_files.update(coder_outputs.get("applied_files", []))
                context["applied_files"] = sorted(all_applied_files)

                if not coder_outputs.get("applied_files"):
                    run.set_step_failed(StepID.IMPLEMENT, "Coder produced no file changes")
                    run.status = RunStatus.FAILED
                    run.error = "No changes produced"
                    self.state.save_run(run)
                    return run

                run.set_step_done(StepID.IMPLEMENT, {
                    "applied_files": context.get("applied_files", []),
                    "attempt": attempt,
                })
            except Exception as e:
                run.set_step_failed(StepID.IMPLEMENT, str(e))
                run.status = RunStatus.FAILED
                run.error = f"Implementation failed: {e}"
                self.state.save_run(run)
                return run

            # TEST
            logger.info(f"[{run.run_id}] TEST (attempt {attempt}/{max_retries})")
            run.set_step_running(StepID.TEST)
            self.state.save_run(run)

            try:
                test_outputs = self.tester.run(context)
                context.update(test_outputs)

                if test_outputs.get("issue_resolved", test_outputs.get("test_passed")):
                    run.set_step_done(StepID.TEST, test_outputs)
                    self.state.save_run(run)
                    logger.info(
                        f"[{run.run_id}] Resolution gate passed on attempt {attempt} "
                        f"(mode={self.config.agent.resolution_mode})"
                    )
                    return run

                # Tests failed — feed back to coder for retry
                feedback = test_outputs.get("feedback", "Tests failed — see output")
                context["verify_feedback"] = feedback
                context["last_test_feedback"] = feedback
                context["last_test_output"] = test_outputs.get("test_output", {})
                auto_guidance = self._derive_auto_guidance(context["last_test_output"], feedback)
                if auto_guidance:
                    context["auto_guidance"] = auto_guidance
                    logger.info(f"[{run.run_id}] Auto guidance: {auto_guidance[:180]}")
                logger.warning(f"[{run.run_id}] Tests failed (attempt {attempt}): {feedback[:200]}")
                run.step(StepID.TEST).retries += 1

            except Exception as e:
                run.set_step_failed(StepID.TEST, str(e))
                run.status = RunStatus.FAILED
                run.error = f"Testing failed: {e}"
                self.state.save_run(run)
                return run

        # Exhausted retries
        run.set_step_failed(StepID.TEST, f"Tests failed after {max_retries} attempts")
        run.status = RunStatus.FAILED
        run.error = f"Tests failed after {max_retries} retries"
        self.state.save_run(run)
        return run

    def _derive_auto_guidance(self, test_output: dict[str, Any], feedback: str) -> str:
        """Generate deterministic retry guidance from concrete failure patterns."""
        stdout = (test_output.get("stdout") or "")
        stderr = (test_output.get("stderr") or "")
        combined = f"{stdout}\n{stderr}\n{feedback}".lower()

        guidance: list[str] = []

        if "modulenotfounderror" in combined:
            missing_modules = sorted(set(re.findall(r"no module named '([^']+)'", f"{stdout}\n{stderr}", re.I)))
            if missing_modules:
                guidance.append(
                    "Fix import/module resolution first. Ensure these import targets exist at repo root "
                    f"or expected package paths: {', '.join(missing_modules)}."
                )
            else:
                guidance.append(
                    "Fix import/module resolution first. Ensure module/package names in tests match actual file paths."
                )

        if "assertionerror" in combined:
            guidance.append(
                "Do not rename behavior to bypass tests. Update implementation logic to satisfy current assertions exactly."
            )

        if "@patch(" in feedback or "mock" in combined:
            guidance.append(
                "Respect test mocking paths. Import modules (not symbols) for patched functions "
                "(e.g. `import pkg.mod as mod` then call `mod.func()`)."
            )

        if "duplicate test files" in combined or "same name" in combined:
            guidance.append(
                "Avoid duplicate test module basenames across directories; keep a single canonical test file per feature."
            )

        if "no changes produced" in combined:
            guidance.append(
                "Apply at least one concrete code edit addressing the failing test output. Do not return unchanged files."
            )

        if not guidance and "test" in combined and "failed" in combined:
            guidance.append(
                "Focus on the first failing test and implement the minimal targeted fix before broad refactors."
            )

        return "\n".join(f"- {item}" for item in guidance[:4])

    def _derive_revision_notes(self, comments: list[dict[str, str]]) -> str:
        """Extract likely human revision directives from issue comment history."""
        directives: list[str] = []
        for item in comments:
            body = (item.get("body") or "").strip()
            if not body:
                continue
            if any(marker in body for marker in BOT_COMMENT_MARKERS):
                continue
            directives.append(f"- @{item.get('author', 'unknown')}: {body[:1200]}")
        if not directives:
            return ""
        # Keep most recent guidance concise.
        return "\n".join(directives[-5:])

    def _step_pr(self, run: Run, context: dict) -> Run:
        logger.info(f"[{run.run_id}] PR — creating pull request")
        run.set_step_running(StepID.PR)
        self.state.save_run(run)

        try:
            pr_outputs = self.pr_agent.run(context)
            context.update(pr_outputs)

            pr = self.github.create_pull_request(
                branch_name=run.branch_name,
                title=pr_outputs["pr_title"],
                body=pr_outputs["pr_body"],
                issue_numbers=run.issues,
                labels=[self.config.labels.review],
            )
            run.pr_number = pr.number
            run.pr_url = pr.html_url
            run.set_step_done(StepID.PR, {"pr_number": pr.number, "pr_url": pr.html_url})

        except Exception as e:
            run.set_step_failed(StepID.PR, str(e))
            run.status = RunStatus.FAILED
            run.error = f"PR creation failed: {e}"

        self.state.save_run(run)
        return run

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    def _write_run_log(self, run: Run, context: dict, error: Exception | None = None) -> None:
        """Write a per-run failure log to eval/results/run_logs/ for offline analysis."""
        import datetime
        from pathlib import Path

        log_dir = Path("eval/results/run_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run.run_id}.md"

        repo = context.get("repo", run.repo)
        issue_number = context.get("issue_number", "?")
        plan = context.get("plan", {})
        test_output = context.get("test_output", {})
        feedback = context.get("feedback", "")

        lines = [
            f"# Run {run.run_id} — FAILED",
            f"**Date:** {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Repo:** {repo}",
            f"**Issue:** #{issue_number}",
            f"**Branch:** {run.branch_name}",
            "",
            "## Error",
            f"```",
            str(error or run.error or "unknown")[:2000],
            "```",
            "",
            "## Plan",
            f"```json",
            json.dumps(plan, indent=2)[:3000] if plan else "not reached",
            "```",
            "",
            "## Test output",
            f"```",
            (test_output.get("stdout", "") + test_output.get("stderr", ""))[:3000] if test_output else "not reached",
            "```",
            "",
            "## Tester feedback",
            feedback[:1000] if feedback else "none",
        ]
        log_path.write_text("\n".join(lines))
        logger.info(f"Run log written: {log_path}")

    def _finalize_failure(self, run: Run, issue_number: int, context: dict[str, Any] | None = None) -> Run:
        self.state.mark_run_finished(run.run_id)
        context = context or {}
        test_feedback = (context.get("last_test_feedback") or "").strip()
        triggered_by = context.get("trigger_label", self.config.labels.ready)
        test_output = context.get("last_test_output", {})

        # Always mark failed first to trigger explicit failure-state lifecycle.
        target_label = self.config.labels.failed

        try:
            self.github.transition_label(
                issue_number,
                self.config.labels.in_progress,
                target_label,
            )
            self.github.comment_on_issue(
                issue_number,
                f"❌ **Phoenix AI** failed to complete this issue.\n\n"
                f"**Error:** {run.error}\n"
                f"**Run ID:** `{run.run_id}`\n\n"
                f"**Triggered by:** `{triggered_by}`\n\n"
                f"Failure label set to `{self.config.labels.failed}`.",
            )

            # Always trigger secondary diagnostic agent from failed state.
            marker = "### Phoenix Failure Analysis"
            cycles = self.github.count_issue_comments_containing(issue_number, marker)
            if cycles < self.config.agent.auto_revise_max_cycles:
                step_states = {
                    step_name: {
                        "status": step.status.value,
                        "error": step.error,
                        "retries": step.retries,
                    }
                    for step_name, step in run.steps.items()
                }
                analysis = self.failure_analyst.run(
                    {
                        "run_id": run.run_id,
                        "repo": run.repo,
                        "issue_number": issue_number,
                        "issue_title": context.get("issue_title", ""),
                        "issue_body": context.get("issue_body", ""),
                        "run_summary": run.model_dump_json(indent=2),
                        "test_feedback": test_feedback or (run.error or ""),
                        "test_output": test_output or {"error": run.error, "steps": step_states},
                    }
                )
                root_cause = str(analysis.get("root_cause", "")).strip()
                same_root_count = 0
                if root_cause:
                    same_root_count = self.github.count_issue_comments_containing(
                        issue_number, f"**Root cause:** {root_cause}"
                    )

                suggested_fixes = analysis.get("suggested_fixes", [])
                fixes_md = "\n".join(f"- {f}" for f in suggested_fixes) if suggested_fixes else "- (none)"
                self.github.comment_on_issue(
                    issue_number,
                    f"{marker}\n\n"
                    f"**Run ID:** `{run.run_id}`\n"
                    f"**Summary:** {analysis.get('summary', 'N/A')}\n"
                    f"**Root cause:** {root_cause or 'N/A'}\n"
                    f"**Confidence:** {analysis.get('confidence', 'medium')}\n\n"
                    f"**Suggested fixes:**\n{fixes_md}\n\n"
                    f"Relabeling to `{self.config.labels.revise}` for another attempt "
                    f"({cycles + 1}/{self.config.agent.auto_revise_max_cycles}).",
                )
                repeated_root_cause_limit_hit = (
                    root_cause
                    and same_root_count >= (self.config.agent.no_progress_root_cause_repeat_limit - 1)
                )
                if repeated_root_cause_limit_hit:
                    self.github.comment_on_issue(
                        issue_number,
                        "### Phoenix Failure Analysis\n\n"
                        "No-progress guardrail triggered: the same root cause has repeated across retries. "
                        f"Keeping label `{self.config.labels.failed}` for manual intervention.",
                    )
                elif self.webhook_mode:
                    # In webhook mode, do NOT auto-relabel to ai:revise — it would
                    # trigger another webhook event and create an infinite loop.
                    self.github.comment_on_issue(
                        issue_number,
                        f"Keeping label `{self.config.labels.failed}`. "
                        f"Manually relabel to `{self.config.labels.revise}` to retry.",
                    )
                elif self.config.agent.auto_revise_on_test_failure:
                    self.github.transition_label(
                        issue_number,
                        self.config.labels.failed,
                        self.config.labels.revise,
                    )
            else:
                self.github.comment_on_issue(
                    issue_number,
                    "### Phoenix Failure Analysis\n\n"
                    "Automatic revise cycle limit reached. "
                    f"Keeping label `{self.config.labels.failed}` for manual intervention.",
                )
        except Exception as e:
            logger.error(f"Failed to update issue on failure: {e}")
        return run
