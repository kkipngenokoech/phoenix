# PhoenixGitHub

> Kipngeno Koech, Muhammad Adam, Baimam Boukar Jean Jacques, Joao Barros.

*PhoenixGitHub is an always-on AI engineering agent that autonomously resolves GitHub issues — from triage to pull request — using a multi-agent LLM pipeline with layered safety controls. It watches labeled issues, plans and implements changes, verifies correctness against a baseline-aware test strategy, and opens a PR for human review.*

---

## Quick Start

```bash
pip install phoenixgithub
phoenixgithub init      # interactive setup wizard
phoenixgithub watch     # start the watcher
```

Add the `ai:ready` label to any issue in your repository. Phoenix picks it up automatically.

<details>
<summary>
 <h3>More details — workflow, CLI, configuration, token permissions</h3>
</summary>

### How it works

PhoenixGitHub turns issue labels into a lightweight development workflow:

- Pick up work from `ai:ready` or `ai:revise`
- Run a structured pipeline: plan → code → test → PR
- Keep issue state synchronized with labels (`ai:in-progress`, `ai:review`, `ai:failed`, `ai:done`)
- Provide guided retry loops when a run fails
- Support interactive first-time setup with `phoenixgithub init`

Designed for teams who want AI automation in normal GitHub workflows, without replacing human approval on merges.

### End-to-End Flow

When an issue enters `ai:ready` or `ai:revise`, PhoenixGitHub:

1. Transitions the issue to `ai:in-progress`
2. Prepares a working branch (`phoenix/issue-<number>`)
3. Builds a plan from issue details and existing code
4. Applies code changes through the Coder agent
5. Runs validation and test checks
6. Commits and pushes results
7. Creates (or reuses) a pull request
8. Transitions the issue to `ai:review` on success, or `ai:failed` on failure

### Label State Machine

```
ai:ready / ai:revise  →  ai:in-progress  →  ai:review  →  ai:done
                                         →  ai:failed   →  ai:revise (optional)
```

AI state labels are mutually exclusive — at most one is active per issue at any time.

### CLI Reference

| Command | Purpose |
|---------|---------|
| `phoenixgithub init` | Interactive setup wizard that creates `.env` |
| `phoenixgithub watch` | Run the daemon and process labeled issues continuously |
| `phoenixgithub run-issue <number>` | One-shot run for a single issue |
| `phoenixgithub status` | Show watcher state and recent runs |
| `phoenixgithub reset-issue <number>` | Clear local dispatch lock for an issue |

### Configuration

Most users should use `phoenixgithub init`. Manual setup is supported via `.env.example`.

**Core**

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub PAT for issue/PR/label operations |
| `GITHUB_REPO` | Repository in `owner/repo` format |
| `LLM_PROVIDER` | Model provider (e.g. `anthropic`) |
| `LLM_MODEL` | Model ID accepted by your endpoint |
| `LLM_API_KEY` | Provider or gateway API key |
| `POLL_INTERVAL` | Watcher poll interval in seconds |
| `MAX_CONCURRENT_RUNS` | Watcher dispatch pressure |

**Agent behavior**

| Variable | Description |
|----------|-------------|
| `TEST_COMMAND` | Command used by the Tester |
| `AUTO_REVISE_ON_TEST_FAILURE` | Auto-relabel to `ai:revise` on failure |
| `AUTO_REVISE_MAX_CYCLES` | Max auto-revise attempts |
| `NO_PROGRESS_ROOT_CAUSE_REPEAT_LIMIT` | Stop repeated root causes sooner |
| `REVISE_INCREMENTAL` | Reuse branch/worktree on revise runs |
| `ALLOW_NO_TESTS` | Treat pytest exit 5 as pass |
| `VALIDATION_PROFILE` | `auto`, `python`, `frontend`, `generic` |

**Tracing**

| Variable | Description |
|----------|-------------|
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | LangSmith API key |
| `LANGCHAIN_PROJECT` | Project name (e.g. `phoenix-owner/repo`) |

### GitHub Token Permissions

For fine-grained PATs:

- Repository contents: read/write
- Issues: read/write
- Pull requests: read/write
- Workflows: read/write (only if installing workflow helpers)
- Metadata: read-only

</details>

---

## Architecture

<img width="800" height="800" alt="image" src="https://github.com/user-attachments/assets/77fa9716-13d4-48cc-88bb-96ed1f171bb2" />


### Agents

| Agent | Responsibility |
|-------|---------------|
| **Planner** | Reads the issue and repository structure, scores file relevance, outputs a structured JSON plan |
| **Coder** | Produces complete file changes based on the plan, with a self-verification step |
| **Tester** | Runs the test suite with baseline comparison to distinguish pre-existing failures from new regressions |
| **Failure Analyst** | Analyzes failing tests, identifies root causes, and feeds structured suggestions back to the Coder |
| **PR Agent** | Opens the pull request and transitions the issue label to `ai:review` |

---

## Results

<details>
<summary>42 real issues · 14 repositories · 100% correctness preservation · 122s mean resolution time (hard tier)</summary>

### Summary by Difficulty

| Tier | Issues | Correctness Preserved | Avg. Time |
|------|--------|-----------------------|-----------|
| Easy | 12 | 12/12 (100%) | < 4 min |
| Medium | 15 | 15/15 (100%) | < 4 min |
| Hard | 15 | 15/15 (100%) | 122 s |
| **Total** | **42** | **42/42 (100%)** | |

### Hard-Tier Resolution Times

| Repository | Min (s) | Max (s) | Mean (s) |
|------------|---------|---------|----------|
| `keycloak/keycloak` | 68 | 97 | 82 |
| `grafana/grafana` | 74 | 79 | 76 |
| `scikit-learn/scikit-learn` | 99 | 163 | 140 |
| `pytest-dev/pytest` | 102 | 198 | 153 |
| `vuejs/core` | 143 | 175 | 159 |
| **Overall** | **68** | **198** | **122** |

### Per-Repository Results

| Repository | Language | Tier | CP | Baseline Broken? |
|------------|----------|------|----|-----------------|
| `andialbrecht/sqlparse` | Python | easy | 3/3 | No |
| `psf/requests` | Python | easy | 3/3 | Yes |
| `axios/axios` | JavaScript | easy | 3/3 | Yes |
| `google/gson` | Java | easy | 3/3 | N/A (Java) |
| `marshmallow-code/marshmallow` | Python | medium | 3/3 | Yes |
| `facebook/docusaurus` | TypeScript | medium | 3/3 | Yes |
| `calcom/cal.com` | TypeScript | medium | 3/3 | Yes |
| `apache/lucene` | Java | medium | 3/3 | N/A (Java) |
| `langchain-ai/langchain` | Python | medium | 3/3 | Yes |
| `pytest-dev/pytest` | Python | hard | 3/3 | Yes |
| `scikit-learn/scikit-learn` | Python | hard | 3/3 | Yes |
| `vuejs/core` | TypeScript | hard | 3/3 | Yes |
| `grafana/grafana` | TypeScript | hard | 3/3 | Yes |
| `keycloak/keycloak` | Java | hard | 3/3 | N/A (Java) |

Of the 11 non-Java repositories, 10 had pre-existing test failures on their default branch. The baseline-aware evaluation strategy correctly classified all 10 as correctness-preserved — Phoenix introduced no new failures in any case.

</details>

---

## Citation

```bibtex
@software{phoenix2026,
  author    = {Kipngeno Koech and Muhammad Adam and
               Baimam Boukar Jean Jacques and Joao Barros},
  title     = {Phoenix: Safe, End-to-End GitHub Issue Resolution via
               Multi-Agent LLM Pipeline with Layered Safety Controls},
  year      = {2026},
  url       = {https://github.com/kkipngenokoech/phoenix}
}
```
