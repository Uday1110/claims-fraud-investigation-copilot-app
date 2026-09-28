"""WRITE: persist context *outside* the LLM window.

* ``scratchpad_note`` -- agents write short structured notes to the state scratchpad (checkpointed) so later
  agents can read a compact trail instead of re-reading raw tool output.
* ``record_evidence`` -- raw (masked, compressed) tool outputs are written to ``state.tool_evidence``; they are
  used for grounding checks and as the evaluation retrieval context, but are not replayed into prompts.
* Long-term writes (episodes / semantic facts) are done by ``src/memory/long_term.py`` from the memory_writer node.
"""
from __future__ import annotations

from typing import Any

from src.guardrails.pii import mask_obj
from src.observability.run_context import now_iso


def scratchpad_note(state: dict, agent: str, note: str, **data: Any) -> list[dict]:
    pad = list(state.get("scratchpad") or [])
    pad.append({"ts": now_iso(), "agent": agent, "note": note[:400], **mask_obj(data)})
    return pad[-20:]


def record_evidence(state: dict, key: str, value: Any) -> dict:
    ev = dict(state.get("tool_evidence") or {})
    ev[key] = mask_obj(value)
    return ev
