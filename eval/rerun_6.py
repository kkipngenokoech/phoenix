"""Re-run just the 6 axios+langchain issues with fixes applied."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

token = os.getenv("GITHUB_TOKEN", "")
if not token:
    logger.error("GITHUB_TOKEN not set")
    sys.exit(1)

from eval.main import _start_phoenix, _wait_for_phoenix
from eval.issues import CreatedIssue
from eval.runner import run_eval

logger.info("Starting Phoenix + ngrok...")
server = _start_phoenix()
if not _wait_for_phoenix(timeout=30):
    logger.error("Phoenix failed to start within 30s")
    server.terminate()
    sys.exit(1)
logger.info("Phoenix is up.")

try:
    issues_data = json.loads(Path("eval/results/issues_rerun.json").read_text())
    issues = [CreatedIssue(**d) for d in issues_data]
    logger.info(f"Loaded {len(issues)} issues to re-run")
    run_eval(issues, token, output_file="eval/results/run_results_rerun.json")
    logger.info("Re-run complete. Results in eval/results/run_results_rerun.json")
finally:
    logger.info("Shutting down Phoenix...")
    server.terminate()
