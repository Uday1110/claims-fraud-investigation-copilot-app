"""LangGraph multi-agent graph for the Claims Fraud-Investigation Copilot.

Topology (typed state: ``src.state.CopilotState``; structured Pydantic output validated at every node boundary)::

    START
      -> input_guard ──(blocked)──────────────────────────────────────────────┐
           │(allowed)                                                          │
      -> load_context   (summarisation middleware + long-term memory recall)   │
      -> intent_router  (Gemini IntentDecision)                                │
      -> supervisor ◄──────────────┐                                           │
           │ conditional edge      │                                           │
           ├─> history_agent ──────┤   workers loop back to the supervisor     │
           ├─> indicator_agent ────┤                                           │
           ├─> playbook_agent ─────┘                                           │
           ├─> case_writer ─┐                                                  │
           ├─> respond ─────┤                                                  │
           ├─> clarify ─────┤  terminals                                       │
           ├─> out_of_scope ┤                                                  │
           ├─> escalate_human                                                  │
           └─> loop_guard ──┤                                 refuse ◄─────────┘
                            ▼                                    │
                       output_guard ◄────────────────────────────┘
                            ▼
                       memory_writer ──(adverse / escalation)──> request_signoff -> human_gate (interrupt) -> END
                            └──────────────(otherwise)────────────────────────────────> END

Checkpointing: ``AsyncSqliteSaver`` (``langgraph-checkpoint-sqlite``) passed to ``build_graph``.
Loop safety: supervisor ``max_steps`` guard + LangGraph ``recursion_limit`` (``Settings.recursion_limit``).
"""
from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from src.agents.base import Deps
from src.agents.finalize import (make_human_gate, make_memory_writer, make_output_guard, make_request_signoff,
                                 route_after_memory)
from src.agents.intake import make_input_guard, make_intent_router, make_load_context, route_after_guard
from src.agents.supervisor import ROUTABLE, TERMINALS, make_supervisor, route_from_supervisor
from src.agents.workers import make_history_agent, make_indicator_agent, make_playbook_agent
from src.agents.writers import (make_case_writer, make_clarify, make_escalate_human, make_loop_guard, make_out_of_scope,
                                make_refuse, make_respond)
from src.state import WORKERS, CopilotState

NODE_FACTORIES = {
    "input_guard": make_input_guard,
    "load_context": make_load_context,
    "intent_router": make_intent_router,
    "supervisor": lambda deps: make_supervisor(),
    "history_agent": make_history_agent,
    "indicator_agent": make_indicator_agent,
    "playbook_agent": make_playbook_agent,
    "case_writer": make_case_writer,
    "respond": make_respond,
    "clarify": make_clarify,
    "out_of_scope": make_out_of_scope,
    "escalate_human": make_escalate_human,
    "loop_guard": make_loop_guard,
    "refuse": make_refuse,
    "output_guard": make_output_guard,
    "memory_writer": make_memory_writer,
    "request_signoff": make_request_signoff,
    "human_gate": make_human_gate,
}


def build_graph(deps: Deps, checkpointer: Optional[Any] = None, overrides: Optional[dict] = None):
    """Compile the copilot graph. ``overrides`` lets tests swap individual nodes (e.g. a runaway worker)."""
    overrides = overrides or {}
    g = StateGraph(CopilotState)
    for name, factory in NODE_FACTORIES.items():
        g.add_node(name, overrides.get(name) or factory(deps))

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_guard, {"load_context": "load_context", "refuse": "refuse"})
    g.add_edge("load_context", "intent_router")
    g.add_edge("intent_router", "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor, {n: n for n in ROUTABLE})
    for w in WORKERS:
        g.add_edge(w, "supervisor")
    for t in (*TERMINALS, "refuse"):
        g.add_edge(t, "output_guard")
    g.add_edge("output_guard", "memory_writer")
    g.add_conditional_edges("memory_writer", route_after_memory, {"request_signoff": "request_signoff", END: END})
    g.add_edge("request_signoff", "human_gate")
    g.add_edge("human_gate", END)
    return g.compile(checkpointer=checkpointer)
