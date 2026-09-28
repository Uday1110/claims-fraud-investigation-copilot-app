"""Central configuration. Everything comes from environment variables (NFR-01).

A local ``.env`` (git-ignored) is loaded with python-dotenv; ``.env.example`` documents every key.
No secret has a default value in code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
for _k, _v in {"HF_HUB_DISABLE_SYMLINKS_WARNING": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false",
               "DEEPEVAL_TELEMETRY_OPT_OUT": "YES", "ANONYMIZED_TELEMETRY": "False"}.items():
    os.environ.setdefault(_k, _v)
# .env wins over inherited shell variables so the project key is always the one used
load_dotenv(ROOT / ".env", override=True)

DATA_DIR = ROOT / "data"
CLAIMS_DIR = DATA_DIR / "claims"
CORPUS_DIR = DATA_DIR / "policy_corpus"
RUNTIME_DIR = DATA_DIR / "runtime"          # git-ignored: sqlite checkpoints, memory store, faiss index
INDEX_DIR = DATA_DIR / "index"
# tests redirect evidence logs to a temp dir so they never pollute the committed artifacts
LOGS_DIR = Path(os.environ["EVIDENCE_LOGS_DIR"]) if os.getenv("EVIDENCE_LOGS_DIR") else ROOT / "logs"
TRACES_DIR = ROOT / "traces"
REPORTS_DIR = ROOT / "reports"

for _d in (RUNTIME_DIR, INDEX_DIR, LOGS_DIR, TRACES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _flag(name: str, default: bool) -> bool:
    v = _env(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # COPILOT_FORCE_OFFLINE=1 guarantees zero Gemini calls (deterministic fallbacks) even if .env holds a key
    google_api_key: str | None = field(default_factory=lambda: None if _flag("COPILOT_FORCE_OFFLINE", False)
                                       else (_env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY")))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-3.5-flash-lite"))
    judge_model: str = field(default_factory=lambda: _env("GEMINI_JUDGE_MODEL", "gemini-3.5-flash-lite"))
    thinking_level: str = field(default_factory=lambda: _env("GEMINI_THINKING_LEVEL", "low"))

    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "90")))
    llm_max_retries: int = field(default_factory=lambda: int(_env("LLM_MAX_RETRIES", "3")))
    llm_max_concurrency: int = field(default_factory=lambda: int(_env("LLM_MAX_CONCURRENCY", "4")))
    llm_rpm: int = field(default_factory=lambda: int(_env("LLM_RPM", "0")))          # 0 = no spacing
    tool_timeout_s: float = field(default_factory=lambda: float(_env("TOOL_TIMEOUT_S", "20")))
    max_steps: int = field(default_factory=lambda: int(_env("MAX_STEPS", "12")))
    recursion_limit: int = field(default_factory=lambda: int(_env("RECURSION_LIMIT", "40")))
    max_retrieval_rounds: int = field(default_factory=lambda: int(_env("MAX_RETRIEVAL_ROUNDS", "3")))

    price_in_per_1m: float = field(default_factory=lambda: float(_env("PRICE_INPUT_PER_1M", "0.30")))
    price_out_per_1m: float = field(default_factory=lambda: float(_env("PRICE_OUTPUT_PER_1M", "2.50")))
    cost_budget_per_run_usd: float = field(default_factory=lambda: float(_env("COST_BUDGET_PER_RUN_USD", "0.05")))

    phoenix_port: int = field(default_factory=lambda: int(_env("PHOENIX_PORT", "6006")))
    phoenix_project: str = field(default_factory=lambda: _env("PHOENIX_PROJECT", "claims-fraud-copilot"))
    tracing_enabled: bool = field(default_factory=lambda: _flag("TRACING_ENABLED", True))

    # context-engineering knobs (also used by the optimisation benchmark)
    context_compression: bool = field(default_factory=lambda: _flag("CONTEXT_COMPRESSION", True))
    summarize_after_messages: int = field(default_factory=lambda: int(_env("SUMMARIZE_AFTER_MESSAGES", "8")))
    max_context_chars: int = field(default_factory=lambda: int(_env("MAX_CONTEXT_CHARS", "6000")))

    pii_hash_salt: str = field(default_factory=lambda: _env("PII_HASH_SALT", "local-dev-salt"))

    checkpoint_db: Path = field(default_factory=lambda: Path(_env("CHECKPOINT_DB", str(RUNTIME_DIR / "checkpoints.sqlite"))))
    memory_db: Path = field(default_factory=lambda: Path(_env("MEMORY_DB", str(RUNTIME_DIR / "long_term_memory.sqlite"))))


def get_settings() -> Settings:
    return Settings()
