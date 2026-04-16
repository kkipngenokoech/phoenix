#!/usr/bin/env python3
"""Merge SWEBenchResult rows from a patch JSON into a base JSON (match on instance_id)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace rows in base results with rows from patch (same instance_id).",
    )
    parser.add_argument("base", type=Path, help="Main results file to update in place")
    parser.add_argument("patch", type=Path, help="Subset run JSON to merge in")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Write base.json.bak before overwriting",
    )
    args = parser.parse_args()

    if not args.base.is_file():
        print(f"Base file not found: {args.base}", file=sys.stderr)
        return 1
    if not args.patch.is_file():
        print(f"Patch file not found: {args.patch}", file=sys.stderr)
        return 1

    base: list[dict] = json.loads(args.base.read_text())
    patch: list[dict] = json.loads(args.patch.read_text())
    by_id = {r["instance_id"]: r for r in patch}
    base_ids = {r["instance_id"] for r in base}

    out: list[dict] = []
    replaced = 0
    for row in base:
        iid = row["instance_id"]
        if iid in by_id:
            out.append(by_id[iid])
            replaced += 1
        else:
            out.append(row)

    extra = set(by_id) - base_ids
    if extra:
        print(f"Warning: patch instance_ids not in base (ignored): {sorted(extra)}", file=sys.stderr)

    if args.backup:
        bak = args.base.with_suffix(args.base.suffix + ".bak")
        shutil.copy(args.base, bak)
        print(f"Backup: {bak}")

    args.base.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Merged {replaced} row(s) from {args.patch.name} into {args.base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
