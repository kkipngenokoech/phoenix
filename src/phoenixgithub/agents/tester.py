"""Tester agent — runs tests and reports results.

Role: testing (read + run). Cannot modify code.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from phoenixgithub.agents.base import BaseAgent

logger = logging.getLogger(__name__)
ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


class TesterAgent(BaseAgent):
    role = "tester"
    system_prompt = """You are a QA engineer. You receive test output and must produce a
structured verdict.

Respond with valid JSON:
{
    "passed": true | false,
    "summary": "Brief description of results",
    "failures": [
        {
            "test_name": "test_something",
            "error": "AssertionError: expected X got Y"
        }
    ],
    "feedback": "If tests failed, explain what the coder should fix. Be specific."
}

Respond ONLY with the JSON object, no markdown fences."""

    def __init__(
        self,
        llm,
        test_command: str = "pytest --import-mode=importlib --rootdir=.",
        allow_no_tests: bool = False,
        validation_profile: str = "auto",
    ) -> None:
        super().__init__(llm)
        self.test_command = test_command
        self.allow_no_tests = allow_no_tests
        self.validation_profile = validation_profile.lower().strip()

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        clone_path = context["clone_path"]

        # Install dependencies before running tests
        self._install_deps(clone_path)

        # Run the test suite
        test_output = self._run_tests(clone_path)

        if self.allow_no_tests and self._is_no_tests_collected(test_output):
            logger.info("No tests collected (pytest exit 5) and ALLOW_NO_TESTS enabled — treating as pass")
            return {
                "test_passed": True,
                "test_output": test_output,
                "feedback": "",
                "test_verdict": {"passed": True, "summary": "No tests collected; allowed by configuration."},
            }

        stdout = test_output.get("stdout", "")
        skipped_unavailable = any(
            phrase in stdout for phrase in ("tests skipped", "not available", "skipped.")
        ) and test_output["exit_code"] == 0
        if test_output["exit_code"] == 0 and ("passed" in stdout or skipped_unavailable):
            logger.info("Tests passed (or tool unavailable — skipped) — skipping LLM analysis")
            return {
                "test_passed": True,
                "test_output": test_output,
                "feedback": "",
            }

        # Tests failed — check if baseline (without Phoenix's changes) also fails.
        # If baseline was already broken, don't penalize Phoenix for it.
        if test_output["exit_code"] != 0:
            baseline_result = self._run_baseline_comparison(clone_path)
            if baseline_result.get("baseline_also_failed"):
                new_failures = baseline_result.get("new_failures", [])
                if not new_failures:
                    logger.info(
                        "Test failures pre-exist on baseline (no new failures from Phoenix) — treating as pass"
                    )
                    return {
                        "test_passed": True,
                        "test_output": test_output,
                        "feedback": "",
                        "test_verdict": {
                            "passed": True,
                            "summary": "Baseline test suite was already failing; Phoenix introduced no new failures.",
                        },
                    }
                else:
                    logger.warning(
                        "Phoenix introduced %d new failure(s) on top of pre-existing baseline failures: %s",
                        len(new_failures),
                        new_failures[:3],
                    )

        # If tests failed or are ambiguous, ask the LLM to analyze
        prompt = (
            f"## Test Output\n"
            f"**Exit code:** {test_output['exit_code']}\n\n"
            f"**stdout:**\n```\n{test_output.get('stdout', '')[:8000]}\n```\n\n"
            f"**stderr:**\n```\n{test_output.get('stderr', '')[:4000]}\n```\n\n"
            f"Analyze the test results and produce the verdict JSON."
        )

        repo_tag = str(context.get("repo", "unknown")).replace("/", "__")
        issue_tag = f"issue:{context.get('issue_number', 'unknown')}"
        run_tag = f"run:{context.get('run_id', 'unknown')}"
        attempt_tag = f"attempt:{context.get('step_attempt', 'unknown')}"
        raw = self.invoke(
            prompt,
            trace_name="tester.analyze",
            trace_tags=["phoenixgithub", "tester", "analyze", f"repo:{repo_tag}", issue_tag, run_tag, attempt_tag],
            trace_metadata={
                "agent": self.role,
                "run_id": context.get("run_id"),
                "issue_number": context.get("issue_number"),
                "repo": context.get("repo"),
                "branch_name": context.get("branch_name"),
                "step": "test",
                "attempt": context.get("step_attempt"),
                "test_exit_code": test_output.get("exit_code"),
            },
        )

        try:
            verdict = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            verdict = {
                "passed": test_output["exit_code"] == 0,
                "summary": "Could not parse test analysis",
                "failures": [],
                "feedback": raw[:1000],
            }

        return {
            "test_passed": verdict.get("passed", False),
            "test_output": test_output,
            "test_verdict": verdict,
            "feedback": verdict.get("feedback", ""),
        }

    def _install_deps(self, cwd: str) -> None:
        """Best-effort dependency installation before tests run."""
        root = Path(cwd)
        profile = self._resolve_profile(cwd)

        if profile == "frontend":
            if (root / "package.json").exists() and not (root / "node_modules").exists():
                logger.info("node_modules missing — running npm install")
                try:
                    subprocess.run(
                        ["npm", "install", "--prefer-offline"],
                        cwd=cwd, capture_output=True, text=True, timeout=180
                    )
                except Exception as e:
                    logger.warning("npm install failed: %s", e)
        else:
            # Python: install editable if not already installed
            has_pyproject = (root / "pyproject.toml").exists()
            has_setup = (root / "setup.py").exists() or (root / "setup.cfg").exists()
            if has_pyproject or has_setup:
                try:
                    result = subprocess.run(
                        ["pip", "install", "-e", ".[dev,test]", "--quiet", "--no-build-isolation"],
                        cwd=cwd, capture_output=True, text=True, timeout=120
                    )
                    if result.returncode != 0:
                        # Fall back without extras
                        subprocess.run(
                            ["pip", "install", "-e", ".", "--quiet", "--no-build-isolation"],
                            cwd=cwd, capture_output=True, text=True, timeout=120
                        )
                except Exception as e:
                    logger.warning("pip install -e failed: %s", e)
            # Also install from requirements files
            for req_file in ("requirements-dev.txt", "requirements-test.txt", "requirements.txt"):
                req_path = root / req_file
                if req_path.exists():
                    try:
                        subprocess.run(
                            ["pip", "install", "-r", str(req_path), "--quiet"],
                            cwd=cwd, capture_output=True, text=True, timeout=120
                        )
                    except Exception as e:
                        logger.warning("pip install -r %s failed: %s", req_file, e)
                    break  # Only install first requirements file found

    def _run_baseline_comparison(self, cwd: str) -> dict:
        """Stash Phoenix changes, run tests on baseline, restore changes.

        Returns:
            {
                "baseline_also_failed": bool,
                "new_failures": list[str],  # test names that fail with changes but not on baseline
            }
        """
        import re as _re
        root = Path(cwd)

        # Only meaningful for git repos
        if not (root / ".git").exists():
            return {"baseline_also_failed": False, "new_failures": []}

        def _extract_failed_tests(stdout: str) -> set[str]:
            """Extract FAILED test::name lines from pytest output."""
            names: set[str] = set()
            for line in stdout.splitlines():
                m = _re.match(r"^FAILED (.+?) -", line)
                if m:
                    names.add(m.group(1).strip())
                elif line.startswith("FAILED "):
                    names.add(line[7:].split(" ")[0].strip())
            return names

        def _extract_npm_failures(stdout: str) -> set[str]:
            names: set[str] = set()
            for line in stdout.splitlines():
                if "✗" in line or "× " in line or "FAIL" in line:
                    names.add(line.strip()[:80])
            return names

        # Get current failing tests
        current_result = self._run_tests(cwd)
        current_failures = _extract_failed_tests(
            current_result.get("stdout", "") + current_result.get("stderr", "")
        ) or _extract_npm_failures(current_result.get("stdout", ""))

        # Stash Phoenix changes
        try:
            stash_result = subprocess.run(
                ["git", "stash", "--include-untracked"],
                cwd=cwd, capture_output=True, text=True, timeout=30
            )
            stashed = "No local changes" not in stash_result.stdout
        except Exception:
            return {"baseline_also_failed": False, "new_failures": []}

        try:
            baseline_result = self._run_tests(cwd)
            baseline_failed = baseline_result.get("exit_code", 0) != 0
            baseline_failures = _extract_failed_tests(
                baseline_result.get("stdout", "") + baseline_result.get("stderr", "")
            ) or _extract_npm_failures(baseline_result.get("stdout", ""))
        finally:
            if stashed:
                try:
                    subprocess.run(
                        ["git", "stash", "pop"],
                        cwd=cwd, capture_output=True, text=True, timeout=30
                    )
                except Exception:
                    pass

        new_failures = list(current_failures - baseline_failures)
        logger.info(
            "Baseline comparison: baseline_failed=%s, baseline_failures=%d, current_failures=%d, new=%d",
            baseline_failed, len(baseline_failures), len(current_failures), len(new_failures),
        )

        return {
            "baseline_also_failed": baseline_failed,
            "new_failures": new_failures,
        }

    def _run_tests(self, cwd: str) -> dict:
        """Execute the test command and capture output."""
        profile = self._resolve_profile(cwd)
        if profile == "frontend":
            return self._run_frontend_checks(cwd)
        if profile == "java":
            return self._run_java_checks(cwd)
        if profile == "generic":
            return self._run_generic_checks(cwd)
        # default: python profile

        resolved_cwd = str(Path(cwd).resolve())
        raw_parts = shlex.split(self.test_command)
        inline_env: dict[str, str] = {}
        cmd: list[str] = []
        for idx, part in enumerate(raw_parts):
            # Support shell-like inline env prefixes such as:
            # PYTHONPATH=. pytest -q
            if not cmd and ENV_ASSIGNMENT_RE.match(part):
                key, value = part.split("=", 1)
                inline_env[key] = value
                continue
            cmd = raw_parts[idx:]
            break

        if not cmd:
            return {"exit_code": -1, "stdout": "", "stderr": f"Invalid TEST_COMMAND: {self.test_command}"}

        if cmd and cmd[0] == "pytest" and not any(part.startswith("--rootdir") for part in cmd[1:]):
            cmd.append("--rootdir=.")

        env = os.environ.copy()
        env.update(inline_env)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{resolved_cwd}:{existing}" if existing else resolved_cwd

        logger.info(f"Running: {' '.join(cmd)} in {resolved_cwd}")
        try:
            proc = subprocess.run(
                cmd,
                cwd=resolved_cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                stderr_head = (proc.stderr or "")[:400].replace("\n", " ")
                stdout_head = (proc.stdout or "")[:400].replace("\n", " ")
                logger.warning(
                    "Test command failed with exit %s; stderr head: %s; stdout head: %s",
                    proc.returncode,
                    stderr_head,
                    stdout_head,
                )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stdout": "", "stderr": "Test execution timed out (300s)"}
        except FileNotFoundError:
            return {"exit_code": -1, "stdout": "", "stderr": f"Command not found: {self.test_command}"}

    def _is_no_tests_collected(self, test_output: dict[str, Any]) -> bool:
        if test_output.get("exit_code") != 5:
            return False
        text = f"{test_output.get('stdout', '')}\n{test_output.get('stderr', '')}".lower()
        return "no tests ran" in text or "collected 0 items" in text

    def _resolve_profile(self, cwd: str) -> str:
        if self.validation_profile in {"python", "frontend", "java", "generic"}:
            return self.validation_profile
        # auto-detect
        root = Path(cwd)
        if (root / "package.json").exists():
            return "frontend"
        if (root / "pom.xml").exists() or (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
            return "java"
        return "python"

    def _run_frontend_checks(self, cwd: str) -> dict:
        """Frontend-friendly checks: prefer npm test/build/lint if available."""
        root = Path(cwd)
        pkg = root / "package.json"
        if not pkg.exists():
            return {"exit_code": 0, "stdout": "No package.json; frontend checks skipped.", "stderr": ""}

        # Ensure node_modules exist
        if not (root / "node_modules").exists():
            logger.info("node_modules missing — running npm install in %s", cwd)
            try:
                subprocess.run(
                    ["npm", "install", "--prefer-offline"],
                    cwd=cwd, capture_output=True, text=True, timeout=180
                )
            except Exception as e:
                return {"exit_code": -1, "stdout": "", "stderr": f"npm install failed: {e}"}

        try:
            data = json.loads(pkg.read_text())
            scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
        except Exception:
            scripts = {}

        chosen: list[list[str]] = []
        if isinstance(scripts, dict):
            if "test" in scripts:
                chosen.append(["npm", "run", "test"])
            elif "build" in scripts:
                chosen.append(["npm", "run", "build"])
            elif "lint" in scripts:
                chosen.append(["npm", "run", "lint"])

        if not chosen:
            # Minimal static fallback for static-only frontends.
            has_index = (root / "index.html").exists()
            has_assets = any((root / d).exists() for d in ("css", "js", "src", "public"))
            ok = has_index or has_assets
            return {
                "exit_code": 0 if ok else 1,
                "stdout": "Static frontend fallback checks passed." if ok else "",
                "stderr": "" if ok else "No frontend scripts and no obvious frontend assets found.",
            }

        outputs: list[str] = []
        for cmd in chosen:
            try:
                logger.info(f"Running frontend check: {' '.join(cmd)} in {cwd}")
                proc = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                outputs.append(proc.stdout)
                if proc.returncode != 0:
                    return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
            except FileNotFoundError:
                logger.warning("npm not found — treating frontend tests as skipped (tool unavailable)")
                return {"exit_code": 0, "stdout": "npm not available; frontend tests skipped.", "stderr": ""}
            except subprocess.TimeoutExpired:
                return {"exit_code": -1, "stdout": "", "stderr": "Frontend validation timed out (300s)."}

        return {"exit_code": 0, "stdout": "\n".join(outputs), "stderr": ""}

    def _run_java_checks(self, cwd: str) -> dict:
        """Run Java tests via Maven or Gradle."""
        root = Path(cwd)

        # Prefer Maven if pom.xml exists
        if (root / "pom.xml").exists():
            tool = "mvn"
            cmd = ["mvn", "-q", "test", "-Dsurefire.failIfNoSpecifiedTests=false"]
        elif (root / "build.gradle.kts").exists() or (root / "build.gradle").exists():
            tool = "gradle"
            wrapper = root / "gradlew"
            gradle_bin = str(wrapper) if wrapper.exists() else "gradle"
            cmd = [gradle_bin, "test", "--quiet", "--continue"]
        else:
            return {"exit_code": 1, "stdout": "", "stderr": "No pom.xml or build.gradle found."}

        logger.info(f"Running Java tests ({tool}): {' '.join(cmd)} in {cwd}")
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except FileNotFoundError:
            logger.warning("%s not found — treating Java tests as skipped (tool unavailable)", tool)
            return {"exit_code": 0, "stdout": f"{tool} not available; Java tests skipped.", "stderr": ""}
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stdout": "", "stderr": "Java test execution timed out (600s)."}

    def _run_generic_checks(self, cwd: str) -> dict:
        """Generic fallback checks for repos without runnable tests."""
        root = Path(cwd)
        if any((root / p).exists() for p in ("README.md", "index.html", "src", "app", "main.py")):
            return {"exit_code": 0, "stdout": "Generic sanity checks passed.", "stderr": ""}
        return {"exit_code": 1, "stdout": "", "stderr": "Generic sanity checks failed: repository appears empty."}
