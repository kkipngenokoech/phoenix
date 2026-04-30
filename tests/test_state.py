"""Unit tests for StateManager — thread safety and state transitions."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from phoenixgithub.state import StateManager


def _make_manager(tmp: str) -> StateManager:
    return StateManager(
        state_file=str(Path(tmp) / ".watcher-state.json"),
        workspace_dir=str(Path(tmp) / "workspace"),
    )


class WatcherStateTests(unittest.TestCase):
    def test_fresh_manager_has_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = _make_manager(tmp)
            self.assertEqual(sm.watcher.dispatched, {})
            self.assertEqual(sm.watcher.active_runs, 0)

    def test_mark_dispatched_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = _make_manager(tmp)
            sm.mark_dispatched(42, "run-abc")
            self.assertTrue(sm.is_dispatched(42))
            self.assertEqual(sm.watcher.active_runs, 1)

            # Reload from disk
            sm2 = _make_manager(tmp)
            self.assertTrue(sm2.is_dispatched(42))

    def test_mark_run_finished_decrements_and_clears(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = _make_manager(tmp)
            sm.mark_dispatched(1, "run-1")
            sm.mark_dispatched(2, "run-2")
            sm.mark_run_finished("run-1")
            self.assertFalse(sm.is_dispatched(1))
            self.assertTrue(sm.is_dispatched(2))
            self.assertEqual(sm.watcher.active_runs, 1)

    def test_active_runs_never_goes_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = _make_manager(tmp)
            sm.mark_run_finished("nonexistent-run")
            self.assertEqual(sm.watcher.active_runs, 0)

    def test_clear_dispatched_allows_redispatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = _make_manager(tmp)
            sm.mark_dispatched(7, "run-7")
            sm.clear_dispatched(7)
            self.assertFalse(sm.is_dispatched(7))

    def test_concurrent_writes_do_not_corrupt(self) -> None:
        """Fire concurrent mark_dispatched calls and verify final count is consistent."""
        with tempfile.TemporaryDirectory() as tmp:
            sm = _make_manager(tmp)
            n = 20
            errors: list[Exception] = []

            def dispatch(issue_num: int) -> None:
                try:
                    sm.mark_dispatched(issue_num, f"run-{issue_num}")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=dispatch, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"Concurrent writes raised: {errors}")
            self.assertEqual(sm.watcher.active_runs, n)
            self.assertEqual(len(sm.watcher.dispatched), n)

            # Reload and verify persistence
            sm2 = _make_manager(tmp)
            self.assertEqual(len(sm2.watcher.dispatched), n)


class RunStateTests(unittest.TestCase):
    def test_save_and_load_run(self) -> None:
        from phoenixgithub.models import Run
        with tempfile.TemporaryDirectory() as tmp:
            sm = _make_manager(tmp)
            run = Run(repo="owner/repo", issues=[1])
            sm.save_run(run)
            loaded = sm.load_run(run.run_id)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.run_id, run.run_id)
            self.assertEqual(loaded.repo, "owner/repo")

    def test_load_nonexistent_run_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = _make_manager(tmp)
            self.assertIsNone(sm.load_run("doesnotexist"))

    def test_list_runs_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sm = _make_manager(tmp)
            self.assertEqual(sm.list_runs(), [])

    def test_list_runs_returns_all(self) -> None:
        from phoenixgithub.models import Run
        with tempfile.TemporaryDirectory() as tmp:
            sm = _make_manager(tmp)
            r1, r2 = Run(repo="owner/a", issues=[1]), Run(repo="owner/b", issues=[2])
            sm.save_run(r1)
            sm.save_run(r2)
            runs = sm.list_runs()
            self.assertEqual(len(runs), 2)


if __name__ == "__main__":
    unittest.main()
