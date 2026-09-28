"""Typed LangGraph state for the copilot.

Structured sub-results (history, assessment, scheme match, case file ...) are stored as the ``model_dump()`` of
the Pydantic models in ``src/schemas.py`` so the SQLite checkpointer can serialise them.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

WORKERS = ("history_agent", "indicator_agent", "playbook_agent")

# intent -> ordered worker plan the supervisor executes before handing off to the terminal node
PLANS: dict[str, list[str]] = {
    "investigate_claim": ["history_agent", "indicator_agent", "playbook_agent", "case_writer"],
    "history_lookup": ["history_agent", "respond"],
    "indicator_check": ["history_agent", "indicator_agent", "respond"],
    "playbook_lookup": ["playbook_agent", "respond"],
    "followup": ["respond"],
    "action_request": ["escalate_human"],
    "ambiguous": ["clarify"],
    "out_of_scope": ["out_of_scope"],
}

# fields cleared at the start of every new user turn (thread-level memory lives in messages + case_file)
PER_TURN_FIELDS = (
    "intent", "plan", "completed", "next", "step_count", "guard", "quarantine", "final_answer",
    "output_check", "errors", "degraded", "human_review", "scratchpad",
)


class Actor(TypedDict, total=False):
    role: str                      # "investigator" | "claimant" | "reviewer"
    id: str                        # e.g. INV-07 (not personal data)
    authorized_claims: list[str]


class CopilotState(TypedDict, total=False):
    # conversation (short-term memory, persisted by the SQLite checkpointer)
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_summary: str
    run_id: str
    thread_id: str

    # request
    actor: Actor
    request: str                   # sanitised investigator request for this turn
    attachments: list[str]         # untrusted claimant-supplied text for this turn (raw, never sent to planner)
    claim_id: Optional[str]

    # guardrails / context
    guard: dict[str, Any]
    quarantine: dict[str, Any]     # QuarantinedFacts + wrapper metadata
    memory_context: dict[str, Any] # recalled long-term memories (prior sessions)
    scratchpad: list[dict[str, Any]]  # "write" context: agent notes (reset per turn)

    # routing
    intent: dict[str, Any]
    plan: list[str]
    completed: list[str]
    next: str
    step_count: int
    max_steps: int

    # worker outputs (structured, validated)
    history: dict[str, Any]
    assessment: dict[str, Any]
    scheme_match: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    tool_evidence: dict[str, Any]  # compressed raw tool outputs used for grounding / eval context
    case_file: dict[str, Any]

    # output
    final_answer: str
    output_check: dict[str, Any]
    human_review: dict[str, Any]
    errors: list[dict[str, Any]]
    degraded: bool
