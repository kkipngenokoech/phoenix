.PHONY: install status watch run-issue labels setup-actions reset-state clean-repo-state clean-workspace-all onboard serve serve-local swe swe-full swe-one swe-retry-failed
PYTHON ?= .venv/bin/python

# ── Core CLI ──────────────────────────────────────────────────────────────────

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
	@echo "Watcher state reset"

clean-repo-state:
	@$(PYTHON) scripts/reset_repo_state.py

clean-workspace-all:
	@echo "Removing local workspace..."
	@rm -rf ./workspace
	@echo "Done."

onboard:
	@echo "Onboarding from .env (GITHUB_REPO)..."
	@$(MAKE) clean-workspace-all
	@$(MAKE) clean-repo-state
	@$(PYTHON) scripts/create_labels.py
	@$(PYTHON) scripts/install_merge_done_workflow.py
	@$(PYTHON) -m phoenixgithub.cli status
	@echo "Onboarding complete. Next: make watch"

# ── Webhook server ────────────────────────────────────────────────────────────

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

# ── SWE-bench evaluation ──────────────────────────────────────────────────────
# Install deps once:  make swe-install
# Full 300-instance run:   make swe-full
# Single instance re-run:  make swe-one INSTANCE=django__django-11019
# Retry all ai:failed:     make swe-retry-failed

SWEBENCH_TIER      ?= lite
SWEBENCH_MAX       ?= 300
SWEBENCH_MAX_WAIT  ?= 2700
SWEBENCH_OUT       ?= eval/results/swebench_results.json
SWEBENCH_WORKSPACE ?= /tmp/phoenix-swebench
SWEBENCH_EXTRA     ?=
# Space-separated repo short-names to exclude (C-extension repos that can't
# compile from old base commits on newer Python versions):
#   make swe-full SWEBENCH_SKIP="astropy matplotlib scikit-learn"
SWEBENCH_SKIP      ?=

swe-install:
	$(PYTHON) -m pip install datasets

# ── One-shot full run (starts Phoenix + ngrok, runs eval, stops server) ───────
#   make swe-full
#   make swe-full SWEBENCH_MAX_WAIT=7200
#   make swe-full SWEBENCH_EXTRA='--only django__django-11019'

.PHONY: swe-full
swe-full:
	@echo "==> Installing datasets..."
	@$(PYTHON) -m pip install --quiet datasets
	@echo "==> Freeing port 8000 and stale processes..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@pkill -f "ngrok http" 2>/dev/null || true
	@rm -f /tmp/phoenix-swe.pid /tmp/ngrok-swe.pid /tmp/phoenix-swe-url /tmp/ngrok-swe.log
	@$(PYTHON) -c "import json,pathlib; f=pathlib.Path('.watcher-state.json'); d=json.loads(f.read_text()) if f.exists() else {}; d.update({'active_runs':0,'dispatched':{}}); f.write_text(json.dumps(d,indent=2))" 2>/dev/null || true
	@sleep 1
	@echo "==> Starting Phoenix server..."
	@MAX_RETRIES=3 $(PYTHON) -m phoenixgithub.cli serve --port 8000 > /tmp/phoenix-swe.log 2>&1 & echo $$! > /tmp/phoenix-swe.pid
	@sleep 3
	@echo "==> Starting ngrok tunnel..."
	@$(PYTHON) scripts/start_ngrok.py
	@sleep 4
	@echo "==> Extracting tunnel URL..."
	@$(PYTHON) scripts/get_ngrok_url.py
	@echo "==> Running full SWE-bench Lite eval (resume enabled)..."
	@$(PYTHON) -m eval.main_swebench \
		--tier $(SWEBENCH_TIER) \
		--max $(SWEBENCH_MAX) \
		--max-wait $(SWEBENCH_MAX_WAIT) \
		--workspace $(SWEBENCH_WORKSPACE) \
		--output $(SWEBENCH_OUT) \
		$(if $(SWEBENCH_SKIP),--skip-repos $(SWEBENCH_SKIP),) \
		$(SWEBENCH_EXTRA) \
	; STATUS=$$? ; \
	echo "==> Stopping Phoenix + ngrok..." ; \
	kill $$(cat /tmp/phoenix-swe.pid 2>/dev/null) 2>/dev/null ; \
	kill $$(cat /tmp/ngrok-swe.pid 2>/dev/null) 2>/dev/null ; \
	rm -f /tmp/phoenix-swe.pid /tmp/ngrok-swe.pid /tmp/phoenix-swe-url /tmp/ngrok-swe.log /tmp/phoenix-swe.log ; \
	exit $$STATUS

# ── Re-run a single instance (fresh, no resume) ───────────────────────────────
#   make swe-one INSTANCE=astropy__astropy-6938

.PHONY: swe-one
swe-one:
	@if [ -z "$(INSTANCE)" ]; then \
		echo "Usage: make swe-one INSTANCE=<instance_id>"; \
		exit 1; \
	fi
	$(PYTHON) -m eval.main_swebench \
		--tier $(SWEBENCH_TIER) \
		--max $(SWEBENCH_MAX) \
		--workspace $(SWEBENCH_WORKSPACE) \
		--output $(SWEBENCH_OUT) \
		--only $(INSTANCE) \
		--no-resume

# ── Retry all ai:failed instances from results file ───────────────────────────
#   make swe-retry-failed

.PHONY: swe-retry-failed
swe-retry-failed:
	@echo "==> Finding ai:failed instances in $(SWEBENCH_OUT)..."
	@$(PYTHON) -c "\
import json, sys; \
data = json.load(open('$(SWEBENCH_OUT)')); \
ids = [r['instance_id'] for r in data if r.get('phoenix_final_label') == 'ai:failed']; \
print(f'Found {len(ids)} ai:failed instances'); \
[print(i) for i in ids]" | tee /tmp/swe-retry-ids.txt
	@tail -n +2 /tmp/swe-retry-ids.txt | while read INSTANCE; do \
		echo "==> Retrying $$INSTANCE ..."; \
		$(PYTHON) -m eval.main_swebench \
			--tier $(SWEBENCH_TIER) \
			--max $(SWEBENCH_MAX) \
			--workspace $(SWEBENCH_WORKSPACE) \
			--output $(SWEBENCH_OUT) \
			--only $$INSTANCE \
			--no-resume || true; \
	done
	@echo "==> Retry pass complete."
