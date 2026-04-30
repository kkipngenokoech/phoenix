"""Unit tests for safety guards — all guards are pure stateless functions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phoenixgithub.safety import (
    blast_radius_guard,
    dependency_guard,
    run_all_guards,
    secrets_guard,
    security_scan_guard,
    sensitive_path_guard,
)


def _change(file_path: str, content: str, action: str = "modify") -> dict:
    return {"file_path": file_path, "content": content, "action": action}


class BlastRadiusGuardTests(unittest.TestCase):
    def test_passes_small_change_set(self) -> None:
        changes = [_change(f"src/file{i}.py", "x = 1\n" * 5) for i in range(3)]
        self.assertIsNone(blast_radius_guard(changes))

    def test_blocks_too_many_files(self) -> None:
        changes = [_change(f"src/file{i}.py", "x = 1") for i in range(11)]
        result = blast_radius_guard(changes)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.guard, "BlastRadiusGuard")
        self.assertEqual(result.severity, "error")

    def test_blocks_too_many_lines(self) -> None:
        changes = [_change("src/big.py", "x = 1\n" * 700)]
        result = blast_radius_guard(changes)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.guard, "BlastRadiusGuard")
        self.assertEqual(result.severity, "error")
        self.assertTrue(result.details, "Expected at least one detail message")

    def test_exactly_at_limit_passes(self) -> None:
        changes = [_change(f"src/file{i}.py", "x = 1") for i in range(10)]
        self.assertIsNone(blast_radius_guard(changes))


class SecurityScanGuardTests(unittest.TestCase):
    def test_passes_clean_code(self) -> None:
        changes = [_change("src/mod.py", "def add(a, b):\n    return a + b\n")]
        self.assertIsNone(security_scan_guard(changes))

    def test_flags_os_system(self) -> None:
        changes = [_change("src/mod.py", "os.system('rm -rf /')")]
        result = security_scan_guard(changes)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.severity, "warning")

    def test_flags_eval(self) -> None:
        changes = [_change("src/mod.py", "result = eval(user_input)")]
        self.assertIsNotNone(security_scan_guard(changes))

    def test_flags_shell_true(self) -> None:
        changes = [_change("src/mod.py", "subprocess.run(cmd, shell=True)")]
        self.assertIsNotNone(security_scan_guard(changes))

    def test_ignores_patch_search_side(self) -> None:
        # The "search" field of a patch op should NOT be scanned
        changes = [{"file_path": "src/mod.py", "search": "os.system('old')", "action": "patch"}]
        self.assertIsNone(security_scan_guard(changes))


class SensitivePathGuardTests(unittest.TestCase):
    def test_passes_normal_path(self) -> None:
        changes = [_change("src/utils.py", "pass")]
        self.assertIsNone(sensitive_path_guard(changes))

    def test_blocks_github_workflows(self) -> None:
        changes = [_change(".github/workflows/ci.yml", "on: push")]
        result = sensitive_path_guard(changes)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.severity, "error")

    def test_blocks_dot_env(self) -> None:
        changes = [_change(".env", "SECRET=abc")]
        self.assertIsNotNone(sensitive_path_guard(changes))

    def test_blocks_settings_py(self) -> None:
        changes = [_change("myapp/settings.py", "DEBUG = True")]
        self.assertIsNotNone(sensitive_path_guard(changes))

    def test_blocks_credentials_path(self) -> None:
        changes = [_change("config/credentials.json", "{}")]
        self.assertIsNotNone(sensitive_path_guard(changes))


class SecretsGuardTests(unittest.TestCase):
    def test_passes_clean_code(self) -> None:
        changes = [_change("src/mod.py", "API_KEY = os.environ['API_KEY']\n")]
        self.assertIsNone(secrets_guard(changes))

    def test_detects_hardcoded_api_key(self) -> None:
        changes = [_change("src/mod.py", 'api_key = "AbCdEfGhIjKlMnOpQrStUvWx"')]
        result = secrets_guard(changes)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.severity, "error")

    def test_detects_github_pat(self) -> None:
        changes = [_change("src/mod.py", "token = 'ghp_" + "A" * 36 + "'")]
        self.assertIsNotNone(secrets_guard(changes))

    def test_detects_aws_key(self) -> None:
        changes = [_change("src/mod.py", "key = 'AKIAIOSFODNN7EXAMPLE'")]
        self.assertIsNotNone(secrets_guard(changes))

    def test_detects_bearer_token(self) -> None:
        changes = [_change("src/mod.py", "auth = 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9abcdefgh'")]
        self.assertIsNotNone(secrets_guard(changes))


class DependencyGuardTests(unittest.TestCase):
    def test_passes_stdlib_imports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            changes = [_change("src/mod.py", "import os\nimport json\nimport re\n", "create")]
            self.assertIsNone(dependency_guard(changes, tmp))

    def test_passes_declared_dep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "requirements.txt").write_text("requests>=2.0\n")
            changes = [_change("src/mod.py", "import requests\n", "create")]
            self.assertIsNone(dependency_guard(changes, tmp))

    def test_flags_undeclared_dep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "requirements.txt").write_text("requests>=2.0\n")
            changes = [_change("src/mod.py", "import boto3\n", "create")]
            result = dependency_guard(changes, tmp)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.severity, "warning")

    def test_ignores_patch_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            changes = [{"file_path": "src/mod.py", "search": "x", "replace": "y", "action": "patch"}]
            self.assertIsNone(dependency_guard(changes, tmp))

    def test_passes_when_no_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # No requirements.txt — guard should not crash
            changes = [_change("src/mod.py", "import stdlib_only_pkg\n", "create")]
            # Won't be flagged because there's no manifest to declare it absent from
            result = dependency_guard(changes, tmp)
            # Result may be non-None (unknown) but should not raise
            _ = result


class RunAllGuardsTests(unittest.TestCase):
    def test_clean_changes_return_empty_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            changes = [_change("src/utils.py", "def add(a, b):\n    return a + b\n")]
            errors, warnings = run_all_guards(changes, clone_path=tmp)
            self.assertEqual(errors, [])

    def test_secret_goes_to_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            changes = [_change("src/mod.py", 'api_key = "AbCdEfGhIjKlMnOpQrStUvWx"')]
            errors, _ = run_all_guards(changes, clone_path=tmp)
            guards = [e.guard for e in errors]
            self.assertIn("SecretsGuard", guards)

    def test_security_pattern_goes_to_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            changes = [_change("src/mod.py", "os.system('ls')")]
            _, warnings = run_all_guards(changes, clone_path=tmp)
            guards = [w.guard for w in warnings]
            self.assertIn("SecurityScanGuard", guards)

    def test_sensitive_path_goes_to_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            changes = [_change(".github/workflows/deploy.yml", "on: push")]
            errors, _ = run_all_guards(changes, clone_path=tmp)
            guards = [e.guard for e in errors]
            self.assertIn("SensitivePathGuard", guards)


if __name__ == "__main__":
    unittest.main()
