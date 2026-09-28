"""SELECT: give each agent only the slice of state it needs (least context, least privilege)."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.context.compress import compress_json, trim_to_budget

# which state keys each agent is allowed to see
AGENT_VIEWS: dict[str, tuple[str, ...]] = {
    "intent_router": ("request", "claim_id", "conversation_summary", "memory_context"),
    "history_agent": ("claim_id",),
    "indicator_agent": ("claim_id", "history"),
    "playbook_agent": ("claim_id", "history", "assessment", "request"),
    "case_writer": ("claim_id", "history", "assessment", "scheme_match", "memory_context", "quarantine"),
    "respond": ("request", "claim_id", "history", "assessment", "scheme_match", "case_file", "memory_context",
                "conversation_summary", "human_review", "quarantine"),
}


def select_view(state: dict, agent: str, max_chars: int = 6000) -> dict[str, Any]:
    keys = AGENT_VIEWS.get(agent, ())
    view = {k: state.get(k) for k in keys if state.get(k) not in (None, "", [], {})}
    if "history" in view and agent in ("playbook_agent", "case_writer", "respond"):
        h = view["history"]
        view["history"] = {k: h.get(k) for k in ("claim_id", "line_of_business", "loss_type", "policy_tenure_days",
                                                 "report_lag_days", "claimed_amount", "timeline_summary",
                                                 "linked_entities", "claimant_statement_facts") if h.get(k) is not None}
    if "memory_context" in view:
        view["memory_context"] = select_memories(view["memory_context"], state.get("claim_id"))
    # hard budget per view
    if len(compress_json(view, 10**9)) > max_chars:
        view = {k: (compress_json(v, max_chars // max(1, len(view))) if not isinstance(v, str) else v[: max_chars // 4])
                for k, v in view.items()}
    return view


def select_memories(memory_context: dict, claim_id: str | None, k: int = 4) -> dict:
    """Prefer memories about the current claim, then the most similar ones."""
    eps = memory_context.get("episodes", [])
    if claim_id:
        eps = sorted(eps, key=lambda e: e.get("claim_id") != claim_id)
    facts = memory_context.get("facts", [])
    return {"episodes": eps[:k], "facts": facts[:k]}


def investigator_turns(state: dict, max_chars: int = 3000) -> list[dict]:
    """Recent conversation (investigator + copilot turns only; never claimant text) for follow-ups."""
    msgs = [m for m in state.get("messages", []) if isinstance(m, (HumanMessage, AIMessage))]
    msgs = trim_to_budget(msgs, max_chars)
    return [{"role": "investigator" if isinstance(m, HumanMessage) else "copilot", "text": str(m.content)[:1200]}
            for m in msgs]
