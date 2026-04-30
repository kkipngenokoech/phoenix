"""Re-run oracle evaluation for succeeded instances with wrong/missing oracle results.

Usage:
    python -m eval.reoracle [--results eval/results/swebench_results.json]

Run this after the main eval finishes to fix oracle results for instances where
the format-specific test runner was not available (e.g., Django, sympy).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
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
    parser = argparse.ArgumentParser(description="Re-run oracle for succeeded instances")
    parser.add_argument("--results", default="eval/results/swebench_results.json")
    parser.add_argument("--workspace", default="/tmp/phoenix-swebench")
    parser.add_argument("--repos", nargs="*", help="Limit to specific repos (e.g. django sympy)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset
    from eval.swebench import (
        SWEBenchInstance,
        evaluate_resolution,
        _get_or_create_oracle_venv,
    )

    results_path = Path(args.results)
    data = json.loads(results_path.read_text())

    # Load dataset for instance metadata (test_patch, fail_to_pass, pass_to_pass)
    logger.info("Loading SWE-bench Lite dataset...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    dataset: dict[str, dict] = {r["instance_id"]: r for r in ds}

    oracle_python, oracle_pip = _get_or_create_oracle_venv(args.workspace)

    repo_filter = set(args.repos) if args.repos else None

    to_reprocess = []
    for r in data:
        if r.get("status") != "succeeded":
            continue
        if not r.get("pr_number"):
            continue
        repo_short = r["repo"].split("/")[-1]
        if repo_filter and repo_short not in repo_filter:
            continue
        to_reprocess.append(r)

    logger.info(f"Found {len(to_reprocess)} succeeded instance(s) to re-oracle")

    if args.dry_run:
        for r in to_reprocess:
            print(f"  {r['instance_id']}  pr=#{r['pr_number']}  "
                  f"current_oracle={r.get('resolved_oracle')}  ftp={r.get('fail_to_pass_passed')}/{r.get('fail_to_pass_total')}")
        return

    updated = 0
    for r in to_reprocess:
        iid = r["instance_id"]
        raw = dataset.get(iid)
        if not raw:
            logger.warning(f"  {iid}: not in dataset — skipping")
            continue

        fail_to_pass = json.loads(raw["FAIL_TO_PASS"]) if isinstance(raw["FAIL_TO_PASS"], str) else raw["FAIL_TO_PASS"]
        pass_to_pass = json.loads(raw["PASS_TO_PASS"]) if isinstance(raw["PASS_TO_PASS"], str) else raw["PASS_TO_PASS"]

        instance = SWEBenchInstance(
            instance_id=iid,
            repo=r["repo"],
            base_commit=raw["base_commit"],
            problem_statement=raw["problem_statement"],
            test_patch=raw.get("test_patch", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )

        repo_name = r["repo"].split("/")[1]
        repo_dir = str(Path(args.workspace) / repo_name)
        pr_branch = f"phoenix/issue-{r['issue_number']}"

        logger.info(f"\n  Re-oracle: {iid}  branch={pr_branch}")

        try:
            outcome = evaluate_resolution(
                instance, pr_branch, repo_dir,
                oracle_python=oracle_python,
                oracle_pip=oracle_pip,
            )
        except Exception as e:
            logger.error(f"  {iid}: oracle error: {e}")
            continue

        old_oracle = r.get("resolved_oracle")
        r["resolved_oracle"] = outcome.resolved
        r["cp"] = outcome.cp
        r["fail_to_pass_passed"] = outcome.fail_to_pass_passed
        r["pass_to_pass_broken"] = outcome.pass_to_pass_broken
        r["oracle_ran"] = outcome.skip_reason is None
        r["oracle_skip_reason"] = outcome.skip_reason

        logger.info(
            f"  {iid}: oracle {old_oracle!r} → {outcome.resolved!r}  "
            f"FTP={outcome.fail_to_pass_passed}/{len(fail_to_pass)}"
        )
        updated += 1

        # Save incrementally
        results_path.write_text(json.dumps(data, indent=2))

    logger.info(f"\nRe-oracle complete: updated {updated} instance(s)")

    # Final summary
    resolved = sum(1 for r in data if r.get("resolved_oracle") is True)
    succeeded = sum(1 for r in data if r.get("status") == "succeeded")
    logger.info(f"Oracle resolved: {resolved}/{len(data)} total  ({succeeded} succeeded)")


if __name__ == "__main__":
    main()
