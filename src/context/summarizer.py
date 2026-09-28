"""Summarization middleware for the conversation (short-term memory compression).

Runs at the start of every turn (``load_context`` node). When the thread holds more than
``SUMMARIZE_AFTER_MESSAGES`` messages, everything except the most recent 4 is folded into a running
``conversation_summary`` (Gemini, with an extractive fallback) and removed from ``messages`` via
``RemoveMessage`` -- the checkpoint stays small while facts stated earlier remain available.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

KEEP_RECENT = 4


def _extractive(existing: str, msgs: list) -> str:
    lines = [existing] if existing else []
    for m in msgs:
        who = "Investigator" if isinstance(m, HumanMessage) else "Copilot"
        lines.append(f"{who}: {str(m.content)[:240]}")
    return "\n".join(lines)[-3000:]


async def summarize_if_needed(llm, state: dict, threshold: int) -> dict:
    msgs = list(state.get("messages") or [])
    if len(msgs) <= threshold:
        return {}
    old, existing = msgs[:-KEEP_RECENT], state.get("conversation_summary", "")
    transcript = "\n".join(f"{'INVESTIGATOR' if isinstance(m, HumanMessage) else 'COPILOT'}: {str(m.content)[:1500]}"
                           for m in old if isinstance(m, (HumanMessage, AIMessage)))
    summary, _ = await llm.text([
        SystemMessage("Summarise this SIU copilot conversation for future turns. Keep every fact the investigator "
                      "stated, claim ids, recommendations and pending decisions. <= 180 words. Masked ids stay masked."),
        HumanMessage(f"EXISTING SUMMARY:\n{existing or '(none)'}\n\nNEW TURNS:\n{transcript}"),
    ], agent="summarization_middleware", fallback=lambda: _extractive(existing, old), optional=True)
    return {"conversation_summary": summary,
            "messages": [RemoveMessage(id=m.id) for m in old if getattr(m, "id", None)]}
