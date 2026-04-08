"""Remove all ai:* labels from the 6 rerun issues so Phoenix can re-trigger."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from github import Github, Auth

token = os.getenv("GITHUB_TOKEN", "")
if not token:
    print("GITHUB_TOKEN not set")
    sys.exit(1)

g = Github(auth=Auth.Token(token))
issues_data = json.loads(Path("eval/results/issues_rerun.json").read_text())

for entry in issues_data:
    fork = entry["fork"]
    num = entry["issue_number"]
    repo = g.get_repo(fork)
    issue = repo.get_issue(num)
    ai_labels = [l for l in issue.labels if l.name.startswith("ai:")]
    if ai_labels:
        for lbl in ai_labels:
            issue.remove_from_labels(lbl.name)
            print(f"  Removed {lbl.name} from {fork}#{num}")
    else:
        print(f"  {fork}#{num} — no ai: labels")

print("Done. All ai:* labels removed.")
