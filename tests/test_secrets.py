"""Secrets hygiene (NFR-01): no keys committed, .env ignored, config is env-only."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_PATTERNS = [re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), re.compile(r"sk-[A-Za-z0-9]{32,}"),
                re.compile(r"(?i)(api[_-]?key|secret|token)\s*=\s*['\"][A-Za-z0-9_\-]{24,}['\"]")]
SKIP_DIRS = {".venv", ".git", "data/runtime", "data/index", "__pycache__", ".pytest_cache"}


def _repo_files():
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_file() and not any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS) and p.name != ".env" \
                and p.suffix not in (".parquet", ".png", ".sqlite", ".faiss"):
            yield p


def test_gitignore_covers_env():
    ignore = (ROOT / ".gitignore").read_text("utf-8").splitlines()
    assert ".env" in ignore and "!.env.example" in ignore


def test_env_example_has_no_real_key():
    text = (ROOT / ".env.example").read_text("utf-8")
    assert "GOOGLE_API_KEY=your-gemini-api-key-here" in text
    assert not any(p.search(text) for p in KEY_PATTERNS)


def test_no_secret_patterns_in_repository_files():
    hits = [f"{p.relative_to(ROOT)}" for p in _repo_files()
            if any(rx.search(p.read_text("utf-8", errors="ignore")) for rx in KEY_PATTERNS)]
    assert not hits, f"possible secrets committed: {hits}"


def test_env_file_is_not_tracked_by_git():
    if not (ROOT / ".git").exists():
        return
    tracked = subprocess.run(["git", "ls-files", ".env"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert tracked == ""
