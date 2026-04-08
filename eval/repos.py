"""Registry of evaluation repositories with metadata."""

from dataclasses import dataclass, field
from typing import Literal

Language = Literal["python", "javascript", "typescript", "java"]
Level = Literal["easy", "medium", "hard", "extreme"]
Profile = Literal["python", "frontend", "java", "generic"]


@dataclass
class EvalRepo:
    owner: str
    name: str
    language: Language
    level: Level
    profile: Profile
    purpose: str
    refactoring_tasks: list[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def fork_name(self) -> str:
        return self.name


EVAL_REPOS: list[EvalRepo] = [
    # ── Group 1: Python ───────────────────────────────────────────────────────
    EvalRepo(
        owner="andialbrecht", name="sqlparse",
        language="python", level="easy", profile="python",
        purpose="Naming normalization and dead-code elimination",
        refactoring_tasks=["naming_normalization", "dead_code_elimination"],
    ),
    EvalRepo(
        owner="psf", name="requests",
        language="python", level="easy", profile="python",
        purpose="Rapid iteration on HTTP logic",
        refactoring_tasks=["modularization", "naming_normalization"],
    ),
    EvalRepo(
        owner="marshmallow-code", name="marshmallow",
        language="python", level="medium", profile="python",
        purpose="Testing Failure Analyst Agent with logic-heavy bugs",
        refactoring_tasks=["modularization", "dead_code_elimination"],
    ),
    EvalRepo(
        owner="pytest-dev", name="pytest",
        language="python", level="hard", profile="python",
        purpose="Meta-testing: agent fixing the tool it uses for validation",
        refactoring_tasks=["naming_normalization", "modularization"],
    ),
    EvalRepo(
        owner="scikit-learn", name="scikit-learn",
        language="python", level="hard", profile="python",
        purpose="Strict mathematical and testing requirements",
        refactoring_tasks=["dead_code_elimination", "naming_normalization"],
    ),
    EvalRepo(
        owner="django", name="django",
        language="python", level="extreme", profile="python",
        purpose="Planner Agent ability to scan deep directory trees",
        refactoring_tasks=["modularization", "api_migration"],
    ),
    EvalRepo(
        owner="home-assistant", name="core",
        language="python", level="extreme", profile="python",
        purpose="Integration testing with thousands of virtual devices",
        refactoring_tasks=["dead_code_elimination", "modularization"],
    ),
    EvalRepo(
        owner="ansible", name="ansible",
        language="python", level="extreme", profile="python",
        purpose="Large-scale configuration and edge-case handling",
        refactoring_tasks=["naming_normalization", "modularization"],
    ),

    # ── Group 2: JavaScript / TypeScript ──────────────────────────────────────
    EvalRepo(
        owner="axios", name="axios",
        language="javascript", level="easy", profile="frontend",
        purpose="Coder Agent JSON output accuracy",
        refactoring_tasks=["naming_normalization", "dead_code_elimination"],
    ),
    EvalRepo(
        owner="facebook", name="docusaurus",
        language="typescript", level="medium", profile="frontend",
        purpose="Modularization in a React environment",
        refactoring_tasks=["modularization", "dead_code_elimination"],
    ),
    EvalRepo(
        owner="calcom", name="cal.com",
        language="typescript", level="medium", profile="frontend",
        purpose="Frontend/Backend interaction and scheduling logic",
        refactoring_tasks=["modularization", "api_migration"],
    ),
    EvalRepo(
        owner="vuejs", name="core",
        language="typescript", level="hard", profile="frontend",
        purpose="Deep structural reasoning on framework core logic",
        refactoring_tasks=["modularization", "naming_normalization"],
    ),
    EvalRepo(
        owner="grafana", name="grafana",
        language="typescript", level="hard", profile="frontend",
        purpose="Sophisticated codebase with deep dependency trees",
        refactoring_tasks=["dead_code_elimination", "modularization"],
    ),
    EvalRepo(
        owner="microsoft", name="vscode",
        language="typescript", level="extreme", profile="frontend",
        purpose="Automated label-driven workflows",
        refactoring_tasks=["naming_normalization", "dead_code_elimination"],
    ),

    # ── Group 3: Java ─────────────────────────────────────────────────────────
    EvalRepo(
        owner="google", name="gson",
        language="java", level="easy", profile="java",
        purpose="Cross-language compatibility for JSON handling",
        refactoring_tasks=["naming_normalization", "dead_code_elimination"],
    ),
    EvalRepo(
        owner="apache", name="lucene",
        language="java", level="medium", profile="java",
        purpose="Complex search indexing and directory scanning",
        refactoring_tasks=["modularization", "dead_code_elimination"],
    ),
    EvalRepo(
        owner="keycloak", name="keycloak",
        language="java", level="hard", profile="java",
        purpose="Complex identity management and security logic",
        refactoring_tasks=["modularization", "api_migration"],
    ),

    # ── Group 4: Large-scale / AI ─────────────────────────────────────────────
    EvalRepo(
        owner="langchain-ai", name="langchain",
        language="python", level="medium", profile="python",
        purpose="Self-reasoning: testing the AI on the tools it is built with",
        refactoring_tasks=["modularization", "dead_code_elimination"],
    ),
    EvalRepo(
        owner="getsentry", name="sentry",
        language="python", level="extreme", profile="python",
        purpose="Multi-million line complexity and repo-wide context",
        refactoring_tasks=["dead_code_elimination", "naming_normalization"],
    ),
    EvalRepo(
        owner="pytorch", name="pytorch",
        language="python", level="extreme", profile="python",
        purpose="Highest bar for PR merges and correctness preservation",
        refactoring_tasks=["naming_normalization", "dead_code_elimination"],
    ),
]

# Ordered pilot set — Python + JS repos only (no Java/Maven, no pytest self-test)
PILOT_REPOS = [r for r in EVAL_REPOS if r.full_name in {
    # Python (easy)
    "andialbrecht/sqlparse",
    "psf/requests",
    # Python (medium)
    "marshmallow-code/marshmallow",
    "langchain-ai/langchain",
    # JavaScript (easy)
    "axios/axios",
}]

# Skip extreme repos in initial runs
TIER1_REPOS = [r for r in EVAL_REPOS if r.level in ("easy", "medium")]
TIER2_REPOS = [r for r in EVAL_REPOS if r.level == "hard"]
STRESS_REPOS = [r for r in EVAL_REPOS if r.level == "extreme"]


def get_repo(full_name: str) -> EvalRepo:
    for r in EVAL_REPOS:
        if r.full_name == full_name:
            return r
    raise KeyError(f"Repo not found: {full_name}")
