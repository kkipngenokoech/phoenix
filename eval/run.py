"""
Phoenix Evaluation Pipeline — main entrypoint.

Usage:
    # Run full pilot (fork, baseline, create issues, wait, metrics)
    python -m eval.run pilot

    # Run only specific stages
    python -m eval.run pilot --stages baseline,issues
    python -m eval.run pilot --stages run,metrics

    # Run a specific tier
    python -m eval.run tier1
    python -m eval.run tier2

Stages (in order):
    baseline  — fork repos, measure test pass rate and complexity
    issues    — create synthetic GitHub issues and apply ai:ready label
    run       — poll until Phoenix completes each issue
    metrics   — measure post-PR complexity delta, export CSV

Environment variables required:
    GITHUB_TOKEN       — PAT with repo + fork permissions
    EVAL_WORKSPACE     — local directory for clones (default: /tmp/phoenix-eval)

Phoenix must be running (`make serve`) before the 'run' stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("eval/results")
ALL_STAGES = ["baseline", "issues", "run", "metrics"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix evaluation pipeline")
    parser.add_argument("tier", choices=["pilot", "tier1", "tier2", "stress", "all"],
                        help="Which repo tier to evaluate")
    parser.add_argument("--stages", default=",".join(ALL_STAGES),
                        help=f"Comma-separated stages to run (default: all). Options: {ALL_STAGES}")
    parser.add_argument("--label", default="ai:ready",
                        help="GitHub label to apply to trigger Phoenix (default: ai:ready)")
    parser.add_argument("--max-per-repo", type=int, default=3,
                        help="Max real issues to mirror per repo (default: 3)")
    parser.add_argument("--workspace", default=os.getenv("EVAL_WORKSPACE", "/tmp/phoenix-eval"),
                        help="Local workspace directory for clones")
    args = parser.parse_args()

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

    # Select repo set
    from eval.repos import PILOT_REPOS, TIER1_REPOS, TIER2_REPOS, STRESS_REPOS, EVAL_REPOS
    repo_set = {
        "pilot": PILOT_REPOS,
        "tier1": TIER1_REPOS,
        "tier2": TIER2_REPOS,
        "stress": STRESS_REPOS,
        "all": EVAL_REPOS,
    }[args.tier]

    logger.info(f"Tier: {args.tier} | Repos: {len(repo_set)} | Stages: {stages}")
    logger.info(f"Workspace: {workspace}")

    baselines_file = RESULTS_DIR / f"baselines_{args.tier}.json"
    issues_file = RESULTS_DIR / f"issues_{args.tier}.json"
    runs_file = RESULTS_DIR / f"run_results_{args.tier}.json"
    metrics_file = RESULTS_DIR / f"metrics_{args.tier}.json"
    csv_file = RESULTS_DIR / f"metrics_{args.tier}.csv"

    # ── Stage: baseline ───────────────────────────────────────────────────────
    if "baseline" in stages:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE: BASELINE")
        logger.info("=" * 60)
        from eval.baseline import run_baselines
        run_baselines(repo_set, token, workspace, output_file=str(baselines_file))

    # ── Stage: issues ─────────────────────────────────────────────────────────
    if "issues" in stages:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE: CREATE ISSUES")
        logger.info("=" * 60)
        from eval.issues import generate_all_issues
        generate_all_issues(
            repo_set, token,
            label=args.label,
            max_per_repo=args.max_per_repo,
            output_file=str(issues_file),
        )

    # ── Stage: run ────────────────────────────────────────────────────────────
    if "run" in stages:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE: WAIT FOR PHOENIX RUNS")
        logger.info("=" * 60)
        if not issues_file.exists():
            logger.error(f"Issues file not found: {issues_file}. Run the 'issues' stage first.")
            sys.exit(1)

        from eval.issues import CreatedIssue
        from eval.runner import run_eval
        issues_data = json.loads(issues_file.read_text())
        issues = [CreatedIssue(**d) for d in issues_data]
        run_eval(issues, token, output_file=str(runs_file))

    # ── Stage: metrics ────────────────────────────────────────────────────────
    if "metrics" in stages:
        logger.info("\n" + "=" * 60)
        logger.info("STAGE: COMPUTE METRICS")
        logger.info("=" * 60)
        if not runs_file.exists():
            logger.error(f"Runs file not found: {runs_file}. Run the 'run' stage first.")
            sys.exit(1)
        if not baselines_file.exists():
            logger.error(f"Baselines file not found: {baselines_file}. Run the 'baseline' stage first.")
            sys.exit(1)

        from eval.baseline import BaselineResult
        from eval.runner import RunResult
        from eval.metrics import compute_all_metrics, export_csv

        runs = [RunResult(**d) for d in json.loads(runs_file.read_text())]
        baselines = [BaselineResult(**d) for d in json.loads(baselines_file.read_text())]
        all_metrics = compute_all_metrics(runs, baselines, workspace, output_file=str(metrics_file))
        export_csv(all_metrics, output_file=str(csv_file))

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
