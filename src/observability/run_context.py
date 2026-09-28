"""Per-run correlation context (run_id / thread_id / actor) shared by every log writer.

The same ``run_id`` is attached to Phoenix spans (as openinference metadata), to ``logs/tool_calls.jsonl``,
``logs/agent_actions.jsonl`` and ``logs/mcp_transcript.jsonl`` so evidence can be cross-referenced.
"""
from __future__ import annotations

import contextvars
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.guardrails.pii import mask_obj

run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="no-run")
thread_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("thread_id", default="no-thread")
actor_var: contextvars.ContextVar[str] = contextvars.ContextVar("actor", default="system")

_lock = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def current_ids() -> dict[str, str]:
    return {"run_id": run_id_var.get(), "thread_id": thread_id_var.get()}


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one masked JSON object per line. Every evidence log goes through here (NFR-05)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(mask_obj(record), default=str, ensure_ascii=False)
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
