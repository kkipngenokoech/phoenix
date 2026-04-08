.PHONY: install status watch run-issue labels setup-actions reset-state clean-repo-state clean-workspace-all onboard pre-release release serve serve-local slides eval
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
