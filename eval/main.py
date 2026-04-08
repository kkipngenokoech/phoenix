"""
Single-entry eval runner.

    make eval          # pilot (default)
    make eval TIER=tier1
    make eval TIER=tier2

Starts the Phoenix webhook server (+ ngrok if NGROK_AUTHTOKEN is set),
runs the full evaluation pipeline, then shuts everything down.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HEALTH_URL = "http://localhost:8000/health"
RESULTS_DIR = Path("eval/results")
ALL_STAGES = ["baseline", "issues", "run", "metrics"]


def _wait_for_phoenix(timeout: int = 30) -> bool:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(HEALTH_URL, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def _kill_stale_processes() -> None:
    """Kill any leftover ngrok or Phoenix processes before starting fresh."""
    import signal as _signal
    # Kill ngrok
    result = subprocess.run(["pgrep", "-f", "ngrok"], capture_output=True, text=True)
    for pid in result.stdout.strip().splitlines():
        try:
            os.kill(int(pid), _signal.SIGKILL)
            logger.info(f"Killed stale ngrok process {pid}")
        except ProcessLookupError:
            pass
    # Kill anything on port 8000
    result = subprocess.run(["lsof", "-ti", ":8000"], capture_output=True, text=True)
    for pid in result.stdout.strip().splitlines():
        try:
            os.kill(int(pid), _signal.SIGKILL)
            logger.info(f"Killed stale process on :8000 (pid {pid})")
        except ProcessLookupError:
            pass
    time.sleep(2)


def _start_phoenix() -> subprocess.Popen:
    """Start Phoenix webhook server (+ ngrok if configured)."""
    from dotenv import load_dotenv
    load_dotenv()

    _kill_stale_processes()

    ngrok_token = os.getenv("NGROK_AUTHTOKEN", "")
    ngrok_domain = os.getenv("NGROK_DOMAIN", "")

    if ngrok_token:
        # Start ngrok + Phoenix together (same as `make serve`)
        script = (
            "import subprocess, sys, atexit, signal, os;"
            "from pyngrok import conf, ngrok;"
            f"conf.get_default().auth_token = '{ngrok_token}';"
            "port = 8000;"
            f"domain = '{ngrok_domain}';"
            "tunnel = ngrok.connect(port, bind_tls=True, hostname=domain) if domain else ngrok.connect(port, bind_tls=True);"
            "print(f'  Ngrok:   {tunnel.public_url}');"
            "print(f'  Webhook: {tunnel.public_url}/webhook');"
            "atexit.register(ngrok.kill);"
            "proc = subprocess.Popen([sys.executable, '-m', 'phoenixgithub.cli', 'serve', '--port', str(port)]);"
            "signal.signal(signal.SIGINT, lambda *_: (proc.terminate(), sys.exit(0)));"
            "proc.wait()"
        )
        return subprocess.Popen([sys.executable, "-c", script])
    else:
        # Local only (no ngrok) — webhooks won't fire from GitHub,
        # but useful for testing the pipeline flow locally.
        logger.warning("NGROK_AUTHTOKEN not set — starting Phoenix locally (no public tunnel).")
        return subprocess.Popen(
            [sys.executable, "-m", "phoenixgithub.cli", "serve", "--port", "8000"]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix eval — one command to run everything")
    parser.add_argument("--tier", default="pilot",
                        choices=["pilot", "tier1", "tier2", "stress", "all", "rerun", "rerun2"])
    parser.add_argument("--stages", default=",".join(ALL_STAGES))
    parser.add_argument("--label", default="ai:ready")
    parser.add_argument("--max-per-repo", type=int, default=3,
                        help="Max real issues to mirror per repo (default: 3)")
    parser.add_argument("--workspace", default=os.getenv("EVAL_WORKSPACE", "/tmp/phoenix-eval"))
    parser.add_argument("--issues-file", default=None,
                        help="Override the issues JSON file for the run stage")
    parser.add_argument("--no-serve", action="store_true",
                        help="Skip starting Phoenix (use if it's already running)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all stages even if result files already exist")
    args = parser.parse_args()

    # Load .env before reading any env vars
    from dotenv import load_dotenv
    load_dotenv()

    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        logger.error("GITHUB_TOKEN environment variable is required.")
        sys.exit(1)

    stages = [s.strip() for s in args.stages.split(",")]
    workspace = args.workspace
    Path(workspace).mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Start Phoenix ─────────────────────────────────────────────────────────
    server: subprocess.Popen | None = None
    if not args.no_serve:
        logger.info("Starting Phoenix + ngrok (killing any stale sessions first)...")
        server = _start_phoenix()
        if not _wait_for_phoenix(timeout=30):
            logger.error("Phoenix failed to start within 30s. Check your config.")
            if server:
                server.terminate()
            sys.exit(1)
        logger.info("Phoenix is up.")

    def _shutdown(sig=None, frame=None):
        if server:
            logger.info("Shutting down Phoenix...")
            server.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Select repo set ───────────────────────────────────────────────────────
    from eval.repos import PILOT_REPOS, TIER1_REPOS, TIER2_REPOS, STRESS_REPOS, EVAL_REPOS
    repo_set = {
        "pilot":  PILOT_REPOS,
        "tier1":  TIER1_REPOS,
        "tier2":  TIER2_REPOS,
        "stress": STRESS_REPOS,
        "all":    EVAL_REPOS,
        "rerun":  [],  # issues_rerun.json is pre-built; no repo set needed
        "rerun2": [],  # issues_rerun2.json is pre-built; no repo set needed
    }[args.tier]

    logger.info(f"Tier: {args.tier} | Repos: {len(repo_set)} | Stages: {stages}")

    baselines_file = RESULTS_DIR / f"baselines_{args.tier}.json"
    issues_file    = RESULTS_DIR / f"issues_{args.tier}.json"
    runs_file      = RESULTS_DIR / f"run_results_{args.tier}.json"
    metrics_file   = RESULTS_DIR / f"metrics_{args.tier}.json"
    csv_file       = RESULTS_DIR / f"metrics_{args.tier}.csv"

    try:
        # ── baseline ─────────────────────────────────────────────────────────
        if "baseline" in stages:
            if baselines_file.exists() and not args.force:
                logger.info(f"Skipping baseline — results already exist ({baselines_file}). "
                            "Pass --force to re-run.")
            else:
                logger.info("\n" + "=" * 60)
                logger.info("STAGE: BASELINE  (fork repos + measure)")
                logger.info("=" * 60)
                from eval.baseline import run_baselines
                run_baselines(repo_set, token, workspace, output_file=str(baselines_file))

        # ── issues ────────────────────────────────────────────────────────────
        if "issues" in stages:
            if issues_file.exists() and not args.force:
                logger.info(f"Skipping issues — results already exist ({issues_file}). "
                            "Pass --force to re-run.")
            else:
                logger.info("\n" + "=" * 60)
                logger.info("STAGE: CREATE ISSUES  (trigger Phoenix via ai:ready label)")
                logger.info("=" * 60)
                from eval.issues import generate_all_issues
                generate_all_issues(
                    repo_set, token,
                    label=args.label,
                    max_per_repo=args.max_per_repo,
                    output_file=str(issues_file),
                )

        # ── run ───────────────────────────────────────────────────────────────
        if "run" in stages:
            logger.info("\n" + "=" * 60)
            logger.info("STAGE: WAIT FOR PHOENIX  (SSE-driven, no polling)")
            logger.info("=" * 60)
            active_issues_file = Path(args.issues_file) if args.issues_file else issues_file
            if not active_issues_file.exists():
                logger.error(f"Issues file not found: {active_issues_file}. Run 'issues' stage first.")
                sys.exit(1)

            from eval.issues import CreatedIssue
            from eval.runner import run_eval
            issues_data = json.loads(active_issues_file.read_text())
            issues = [CreatedIssue(**d) for d in issues_data]
            run_eval(issues, token, output_file=str(runs_file))

        # ── metrics ───────────────────────────────────────────────────────────
        if "metrics" in stages:
            logger.info("\n" + "=" * 60)
            logger.info("STAGE: COMPUTE METRICS")
            logger.info("=" * 60)
            if not runs_file.exists():
                logger.error(f"Runs file not found: {runs_file}.")
                sys.exit(1)
            if not baselines_file.exists():
                logger.error(f"Baselines file not found: {baselines_file}.")
                sys.exit(1)

            from eval.baseline import BaselineResult
            from eval.runner import RunResult
            from eval.metrics import compute_all_metrics, export_csv

            runs = [RunResult(**d) for d in json.loads(runs_file.read_text())]
            baselines = [BaselineResult(**d) for d in json.loads(baselines_file.read_text())]
            all_metrics = compute_all_metrics(runs, baselines, workspace,
                                              output_file=str(metrics_file))
            export_csv(all_metrics, output_file=str(csv_file))

        logger.info("\nPipeline complete.")

    finally:
        _shutdown()


if __name__ == "__main__":
    main()
