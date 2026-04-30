"""Unit tests for config — env-var loading, provider enum, and max_retries."""

from __future__ import annotations

import os
import unittest


class LLMProviderTests(unittest.TestCase):
    def test_enum_values(self) -> None:
        from phoenixgithub.config import LLMProvider
        self.assertEqual(LLMProvider.ANTHROPIC.value, "anthropic")
        self.assertEqual(LLMProvider.OPENAI.value, "openai")

    def test_llm_config_defaults_to_anthropic(self) -> None:
        from phoenixgithub.config import LLMConfig, LLMProvider
        cfg = LLMConfig()
        self.assertEqual(cfg.provider, LLMProvider.ANTHROPIC)

    def test_llm_config_reads_openai_from_env(self) -> None:
        from phoenixgithub.config import LLMConfig, LLMProvider
        original = os.environ.get("LLM_PROVIDER")
        try:
            os.environ["LLM_PROVIDER"] = "openai"
            cfg = LLMConfig()
            self.assertEqual(cfg.provider, LLMProvider.OPENAI)
        finally:
            if original is None:
                os.environ.pop("LLM_PROVIDER", None)
            else:
                os.environ["LLM_PROVIDER"] = original

    def test_invalid_provider_raises(self) -> None:
        from phoenixgithub.config import LLMProvider
        with self.assertRaises(ValueError):
            LLMProvider("gemini")


class AgentConfigTests(unittest.TestCase):
    def test_max_retries_default(self) -> None:
        from phoenixgithub.config import AgentConfig
        os.environ.pop("MAX_RETRIES", None)
        cfg = AgentConfig()
        self.assertEqual(cfg.max_retries, 2)

    def test_max_retries_from_env(self) -> None:
        from phoenixgithub.config import AgentConfig
        original = os.environ.get("MAX_RETRIES")
        try:
            os.environ["MAX_RETRIES"] = "5"
            cfg = AgentConfig()
            self.assertEqual(cfg.max_retries, 5)
        finally:
            if original is None:
                os.environ.pop("MAX_RETRIES", None)
            else:
                os.environ["MAX_RETRIES"] = original

    def test_resolution_mode_defaults_to_tests(self) -> None:
        from phoenixgithub.config import AgentConfig
        cfg = AgentConfig()
        self.assertEqual(cfg.resolution_mode, "tests")

    def test_resolution_mode_invalid_falls_back(self) -> None:
        from phoenixgithub.config import AgentConfig
        cfg = AgentConfig(resolution_mode="invalid_mode")
        self.assertEqual(cfg.resolution_mode, "tests")

    def test_resolution_mode_valid_values(self) -> None:
        from phoenixgithub.config import AgentConfig
        for mode in ("tests", "reproducer", "both"):
            cfg = AgentConfig(resolution_mode=mode)
            self.assertEqual(cfg.resolution_mode, mode)


class RunContextTests(unittest.TestCase):
    def test_run_context_is_a_dict(self) -> None:
        from phoenixgithub.models import RunContext
        ctx: RunContext = {"run_id": "abc123", "repo": "owner/repo", "issue_number": 7}
        self.assertIsInstance(ctx, dict)
        self.assertEqual(ctx["run_id"], "abc123")

    def test_run_context_supports_update(self) -> None:
        from phoenixgithub.models import RunContext
        ctx: RunContext = {"run_id": "abc"}
        ctx.update({"issue_title": "Bug: something broken", "clone_path": "/tmp/repo"})  # type: ignore[typeddict-item]
        self.assertEqual(ctx.get("issue_title"), "Bug: something broken")


if __name__ == "__main__":
    unittest.main()
