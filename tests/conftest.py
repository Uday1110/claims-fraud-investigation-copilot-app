"""Shared pytest fixtures.

Agent tests run OFFLINE and deterministically: no Gemini calls (LLMClient offline mode uses each node's
deterministic fallback), no Phoenix, and all evidence logs are redirected to a temp directory so the committed
artifacts under logs/ are never modified by a test run (except logs/memory_test.log, which is the committed
output of tests/test_memory_persistence.py by design).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_TMP = Path(tempfile.mkdtemp(prefix="copilot-tests-"))
os.environ["EVIDENCE_LOGS_DIR"] = str(_TMP / "logs")
os.environ["TRACING_ENABLED"] = "false"
os.environ["CHECKPOINT_DB"] = str(_TMP / "ckpt.sqlite")
os.environ["MEMORY_DB"] = str(_TMP / "mem.sqlite")

import pytest  # noqa: E402

from src.agents.base import Deps  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.memory.long_term import LongTermMemory  # noqa: E402


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "ckpt.sqlite"))
    monkeypatch.setenv("MEMORY_DB", str(tmp_path / "mem.sqlite"))
    return get_settings()


@pytest.fixture
def offline_llm(tmp_settings):
    return LLMClient(tmp_settings, offline=True)


def make_deps(settings, toolbox, llm=None) -> Deps:
    return Deps(llm=llm or LLMClient(settings, offline=True), toolbox=toolbox,
                memory=LongTermMemory(settings.memory_db), settings=settings)


@pytest.fixture
def evidence_dir() -> Path:
    return Path(os.environ["EVIDENCE_LOGS_DIR"])
