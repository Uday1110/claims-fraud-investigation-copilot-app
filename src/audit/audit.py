"""Audit middleware (AC-10): machine-generated trail of consequential agent actions.

Writes ``logs/agent_actions.jsonl`` with ``{timestamp, run_id, thread_id, actor, action, tool, decision, ...}``
for: guardrail blocks, access-control refusals, quarantine events, recommendations drafted, human-gate routing,
human sign-off decisions, escalations, clarifications, loop-guard trips and long-term memory writes.
All values are PII-masked by ``append_jsonl``.
"""
from __future__ import annotations

from typing import Any, Optional

from src.config import LOGS_DIR
from src.observability.run_context import actor_var, append_jsonl, current_ids, now_iso

AUDIT_LOG = LOGS_DIR / "agent_actions.jsonl"

CONSEQUENTIAL_ACTIONS = {
    "request_received", "input_blocked", "access_refused", "untrusted_text_quarantined", "intent_classified",
    "tool_invoked", "recommendation_drafted", "output_blocked", "output_repaired", "routed_for_human_signoff",
    "human_signoff_recorded", "escalated_to_human", "clarification_requested", "out_of_scope_declined",
    "loop_guard_tripped", "memory_written", "memory_recalled", "degraded_mode", "answer_returned",
}


def audit(action: str, *, decision: str, tool: Optional[str] = None, actor: Optional[str] = None,
          agent: Optional[str] = None, details: Optional[dict[str, Any]] = None) -> dict:
    if action not in CONSEQUENTIAL_ACTIONS:
        raise ValueError(f"unknown audit action {action!r}")
    rec = {
        "timestamp": now_iso(),
        **current_ids(),
        "actor": actor or actor_var.get(),
        "agent": agent,
        "action": action,
        "tool": tool,
        "decision": decision,
        "details": details or {},
    }
    append_jsonl(AUDIT_LOG, rec)
    return rec
