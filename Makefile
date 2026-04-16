.PHONY: install status watch run-issue labels setup-actions reset-state clean-repo-state clean-workspace-all onboard pre-release release serve serve-local slides eval eval-swebench swebench-dry-run swebench-install enable-fork-issues merge-swebench-retry
PYTHON ?= .venv/bin/python
TIER   ?= pilot
STAGES ?= baseline,issues,run,metrics

install:
	$(PYTHON) -m pip install -e .

status:
	$(PYTHON) -m phoenixgithub.cli status

watch:
	$(PYTHON) -m phoenixgithub.cli watch

run-issue:
	@if [ -z "$(ISSUE)" ]; then \
		echo "Usage: make run-issue ISSUE=<number>"; \
		exit 1; \
	fi
	$(PYTHON) -m phoenixgithub.cli run-issue "$(ISSUE)"

labels:
	$(PYTHON) scripts/create_labels.py

setup-actions:
	$(PYTHON) scripts/install_merge_done_workflow.py

reset-state:
	rm -f .watcher-state.json
	@echo "Watcher state reset (.watcher-state.json removed)"

clean-repo-state:
	@$(PYTHON) scripts/reset_repo_state.py

clean-workspace-all:
	@echo "Removing entire local workspace directory (./workspace)..."
	@rm -rf ./workspace
	@echo "Workspace cleared."

onboard:
	@echo "Onboarding repo from .env (GITHUB_REPO)..."
	@$(MAKE) clean-workspace-all
	@$(MAKE) clean-repo-state
	@$(PYTHON) scripts/create_labels.py
	@$(PYTHON) scripts/install_merge_done_workflow.py
	@$(PYTHON) -m phoenixgithub.cli status
	@echo "Onboarding complete. Next: make watch"

pre-release:
	@$(PYTHON) scripts/pre_release.py $(if $(TAG),--tag $(TAG),)

release:
	@if [ -z "$(TAG)" ]; then \
		echo "Usage: make release TAG=vX.Y.Z [NOTES='Release notes text']"; \
		exit 1; \
	fi
	@command -v gh >/dev/null 2>&1 || { \
		echo "GitHub CLI (gh) is required. Install from https://cli.github.com/"; \
		exit 1; \
	}
	@gh auth status >/dev/null 2>&1 || { \
		echo "GitHub CLI is not authenticated. Run: gh auth login"; \
		exit 1; \
	}
	@$(MAKE) pre-release TAG="$(TAG)"
	@gh release create "$(TAG)" --title "$(TAG)" $(if $(NOTES),--notes "$(NOTES)",--generate-notes)
	@echo "Release $(TAG) created. GitHub Actions will publish to PyPI."

serve:
	@$(PYTHON) -c "\
import subprocess, sys, atexit, signal, os; \
from dotenv import load_dotenv; \
load_dotenv(); \
from pyngrok import conf, ngrok; \
conf.get_default().auth_token = os.getenv('NGROK_AUTHTOKEN', ''); \
port = 8000; \
domain = os.getenv('NGROK_DOMAIN', ''); \
tunnel = ngrok.connect(port, bind_tls=True, hostname=domain) if domain else ngrok.connect(port, bind_tls=True); \
print(f'\n  Ngrok tunnel: {tunnel.public_url}'); \
print(f'  Webhook URL:  {tunnel.public_url}/webhook\n'); \
atexit.register(ngrok.kill); \
proc = subprocess.Popen([sys.executable, '-m', 'phoenixgithub.cli', 'serve', '--port', str(port)]); \
signal.signal(signal.SIGINT, lambda *_: (proc.terminate(), sys.exit(0))); \
proc.wait()"

serve-local:
	$(PYTHON) -m phoenixgithub.cli serve

slides:
	cd slides && python3 -m http.server 8080

# ── Evaluation pipeline ───────────────────────────────────────────────────────
# One command:  make eval           (pilot, 10 repos)
#               make eval TIER=tier1
#               make eval TIER=tier2
#
# Starts Phoenix + ngrok, forks repos, creates issues, waits via SSE, computes
# metrics, then shuts everything down automatically.

eval:
	$(PYTHON) -m eval.main --tier $(TIER) --stages $(STAGES) --workspace workspace $(if $(FORCE),--force,) $(if $(ISSUES_FILE),--issues-file $(ISSUES_FILE),)

# ── SWE-bench evaluation ──────────────────────────────────────────────────────
# Requires: pip install datasets  (run `make swebench-install` first)
#
# Examples:
#   make swebench-dry-run                          # list instances, no Phoenix
#   make eval-swebench                             # 3 instances/repo, lite tier
#   make eval-swebench SWEBENCH_TIER=verified MAX=5
#   make eval-swebench REPOS="requests pytest"     # specific repos only

# # 1. Install the HuggingFace datasets library (one-time)
# make swebench-install

# # 2. Preview what instances would run — no Phoenix invoked
# make swebench-dry-run

# # 3. Run the actual eval (3 instances per repo, lite tier)
# make eval-swebench

# # 4. Override any defaults
# make eval-swebench SWEBENCH_TIER=verified MAX=5
# make eval-swebench REPOS="requests pytest scikit-learn"
# make eval-swebench REPOS="requests pytest" MAX=2 SWEBENCH_OUT=eval/results/my_run.json
#
# Per-issue Phoenix wait (default 2700s from SWEBENCH_MAX_WAIT in eval/runner.py):
#   SWEBENCH_MAX_WAIT=7200 make eval-swebench
#   SWEBENCH_EXTRA='--max-wait 7200' make swe


SWEBENCH_TIER    ?= lite
SWEBENCH_MAX     ?= 3
# Convenience: `make swe MAX=2` sets instance cap (same as SWEBENCH_MAX=2).
ifneq ($(strip $(MAX)),)
SWEBENCH_MAX     := $(MAX)
endif
SWEBENCH_OUT     ?= eval/results/swebench_results.json
SWEBENCH_WORKSPACE ?= /tmp/phoenix-swebench

swebench-install:
	$(PYTHON) -m pip install datasets

# Enable GitHub Issues on your forks (PATCH has_issues); needs admin on each fork.
#   make enable-fork-issues
#   make enable-fork-issues FORK_REPOS="django matplotlib flask"
enable-fork-issues:
	$(PYTHON) scripts/enable_fork_issues.py $(FORK_REPOS)

# Merge a subset SWE-bench JSON into the main results (same instance_id replaces row).
#   make merge-swebench-retry
#   make merge-swebench-retry PATCH=eval/results/other_partial.json
PATCH            ?= eval/results/swebench_sympy_retry.json
merge-swebench-retry:
	$(PYTHON) scripts/merge_swebench_results.py eval/results/swebench_results.json $(PATCH) --backup

swebench-dry-run:
	$(PYTHON) -m eval.main_swebench \
		--tier $(SWEBENCH_TIER) \
		--max $(SWEBENCH_MAX) \
		$(if $(REPOS),--repos $(REPOS),) \
		--dry-run

eval-swebench:
	$(PYTHON) -m eval.main_swebench \
		--tier $(SWEBENCH_TIER) \
		--max $(SWEBENCH_MAX) \
		--workspace $(SWEBENCH_WORKSPACE) \
		--output $(SWEBENCH_OUT) \
		$(if $(REPOS),--repos $(REPOS),)

# ── One-shot SWE-bench command ────────────────────────────────────────────────
# Installs deps, starts Phoenix + ngrok in the background, runs the eval,
# then kills the server when done.
#
#   make swe                                  # 3 instances/repo, lite tier
#   make swe REPOS="requests pytest" MAX=2    # specific repos, 2 each
#   make swe TIER=verified MAX=5              # verified tier, 5 per repo
#   RESOLUTION_MODE=tests|reproducer|both     # how Phoenix gates the test step (default: tests)
#   SWEBENCH_MAX_WAIT=7200                    # seconds per issue (env; default 2700)
#   SWEBENCH_EXTRA='--only pytest-dev__pytest-11143'   # extra args to main_swebench

.PHONY: swe
swe:
	@echo "==> Installing datasets..."
	@$(PYTHON) -m pip install --quiet datasets
	@echo "==> Freeing port 8000 and any stale ngrok/phoenix..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@pkill -f "ngrok http" 2>/dev/null || true
	@rm -f /tmp/phoenix-swe.pid /tmp/ngrok-swe.pid /tmp/phoenix-swe-url /tmp/ngrok-swe.log
	@$(PYTHON) -c "import json,pathlib; f=pathlib.Path('.watcher-state.json'); d=json.loads(f.read_text()) if f.exists() else {}; d.update({'active_runs':0,'dispatched':{}}); f.write_text(json.dumps(d,indent=2))" 2>/dev/null || true
	@sleep 1
	@echo "==> Starting Phoenix server..."
	@$(PYTHON) -m phoenixgithub.cli serve --port 8000 > /tmp/phoenix-swe.log 2>&1 & echo $$! > /tmp/phoenix-swe.pid
	@sleep 3
	@echo "==> Starting ngrok tunnel (CLI — persists independently)..."
	@$(PYTHON) scripts/start_ngrok.py
	@sleep 4
	@echo "==> Extracting tunnel URL from ngrok API..."
	@$(PYTHON) scripts/get_ngrok_url.py
	@echo "==> Running SWE-bench eval..."
	@$(PYTHON) -m eval.main_swebench \
		--tier $(SWEBENCH_TIER) \
		--max $(SWEBENCH_MAX) \
		--workspace $(SWEBENCH_WORKSPACE) \
		--output $(SWEBENCH_OUT) \
		$(if $(REPOS),--repos $(REPOS),) \
		$(SWEBENCH_EXTRA) \
	; STATUS=$$? ; \
	echo "==> Stopping Phoenix + ngrok..." ; \
	kill $$(cat /tmp/phoenix-swe.pid 2>/dev/null) 2>/dev/null ; \
	kill $$(cat /tmp/ngrok-swe.pid 2>/dev/null) 2>/dev/null ; \
	rm -f /tmp/phoenix-swe.pid /tmp/ngrok-swe.pid /tmp/phoenix-swe-url /tmp/ngrok-swe.log /tmp/phoenix-swe.log ; \
	exit $$STATUS


# # Run 1: with reproducer (default)
# make swe REPOS="requests pytest scikit-learn" MAX=3

# # Run 2: ablation — without reproducer
# USE_REPRODUCER=false make swe REPOS="requests pytest scikit-learn" MAX=3

