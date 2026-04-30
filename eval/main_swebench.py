"""Entry point for Phoenix SWE-bench evaluation.

Examples:
    # Run 3 instances per repo, lite tier (default)
    python -m eval.main_swebench

    # Run 5 instances, verified tier, specific repos
    python -m eval.main_swebench --tier verified --max 5 --repos requests pytest

    # Dry-run: just load and list instances without running Phoenix
    python -m eval.main_swebench --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phoenix on SWE-bench instances")
    parser.add_argument(
        "--tier",
        choices=["lite", "verified", "full"],
        default="lite",
        help="SWE-bench tier to use (default: lite)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=3,
        metavar="N",
        help="Max instances per repo (default: 3)",
    )
    parser.add_argument(
        "--repos",
        nargs="*",
        metavar="REPO",
        help="Filter to specific repo names, e.g. requests pytest (default: all supported)",
    )
    parser.add_argument(
        "--skip-repos",
        nargs="*",
        metavar="REPO",
        default=[],
        help="Exclude repo names, e.g. astropy matplotlib scikit-learn",
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("EVAL_WORKSPACE", "/tmp/phoenix-swebench"),
        help="Local directory for cloned repos",
    )
    parser.add_argument(
        "--output",
        default="eval/results/swebench_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--sse-url",
        default="http://localhost:8000/eval/stream",
        help="Phoenix SSE stream URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print instances without running Phoenix",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="INSTANCE_ID",
        default=[],
        help="Run only these SWE-bench instance_ids (e.g. pytest-dev__pytest-11143)",
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=None,
        metavar="SEC",
        help="Max seconds to wait per issue for Phoenix (overrides SWEBENCH_MAX_WAIT env; clamped 120–21600)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel Phoenix workers (default: 1). Each worker uses its own clone workspace.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode — re-run all instances even if output file exists.",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        metavar="N",
        help="Cap total instances to run (useful for staged rollouts, e.g. --max-total 50).",
    )
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN", "")
    if not token and not args.dry_run:
        logger.error("GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    from eval.swebench import (
        SUPPORTED_REPOS,
        load_swebench_instances,
        run_swebench_eval,
    )

    # Build repo filter
    repo_filter: set[str] | None = None
    if args.repos:
        # Accept either short names ("requests") or full names ("psf/requests")
        repo_filter = set()
        for name in args.repos:
            if "/" in name:
                repo_filter.add(name)
            else:
                # Match by short name
                matched = [r for r in SUPPORTED_REPOS if r.split("/")[1] == name]
                if matched:
                    repo_filter.update(matched)
                else:
                    logger.warning(f"Unknown repo shortname: {name}")

    instances = load_swebench_instances(
        tier=args.tier,
        repos=repo_filter,
        max_per_repo=args.max,
    )

    if args.skip_repos:
        skip_full: set[str] = set()
        for name in args.skip_repos:
            if "/" in name:
                skip_full.add(name)
            else:
                matched = [r for r in SUPPORTED_REPOS if r.split("/")[1] == name]
                skip_full.update(matched)
        before = len(instances)
        instances = [i for i in instances if i.repo not in skip_full]
        logger.info("Skipped %d instance(s) from: %s", before - len(instances), sorted(skip_full))

    if args.only:
        allow = set(args.only)
        instances = [i for i in instances if i.instance_id in allow]
        if not instances:
            logger.error("No instances match --only %s (check instance_id spelling)", allow)
            sys.exit(1)
        logger.info("Filtered to %d instance(s) via --only", len(instances))

    if args.max_total and len(instances) > args.max_total:
        instances = instances[: args.max_total]
        logger.info("Capped to %d instance(s) via --max-total", len(instances))

    if not instances:
        logger.error("No instances loaded — check repo names and tier")
        sys.exit(1)

    if args.dry_run:
        print(f"\nLoaded {len(instances)} SWE-bench instances:\n")
        for inst in instances:
            ftp_n = len(inst.fail_to_pass)
            print(f"  {inst.instance_id:<50}  base={inst.base_commit[:8]}  ftp={ftp_n}")
        print()
        return

    logger.info(f"Starting SWE-bench eval: {len(instances)} instances")
    logger.info(f"  Workspace: {args.workspace}")
    logger.info(f"  Output:    {args.output}")

    results = run_swebench_eval(
        instances=instances,
        github_token=token,
        workspace=args.workspace,
        output_file=args.output,
        sse_url=args.sse_url,
        max_wait=args.max_wait,
        workers=args.workers,
        resume=not args.no_resume,
    )

    resolved = sum(1 for r in results if r.resolved_oracle is True)
    print(f"\nFinal: {resolved}/{len(results)} oracle-resolved  →  see {args.output}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
