"""Supervisor: deterministic, auditable routing over the intent's plan.

The LLM decides *what the investigator wants* (intent router); the supervisor decides *which worker runs next*
with a pure function, so routing is testable (tests/test_routing.py) and bounded (tests/test_loops.py):

* next = first step of ``PLANS[intent]`` not yet in ``completed``
* a fatal worker error (e.g. claim not found) short-circuits to ``respond`` to explain the gap
* ``step_count`` > ``max_steps`` -> ``loop_guard`` (runaway-loop / cascade guard), independent of LangGraph's
  ``recursion_limit`` which is the second, framework-level backstop.
"""
from __future__ import annotations

from src.agents.base import node_boundary
from src.audit.audit import audit
from src.state import PLANS

TERMINALS = ("case_writer", "respond", "clarify", "out_of_scope", "escalate_human", "loop_guard")
ROUTABLE = ("history_agent", "indicator_agent", "playbook_agent") + TERMINALS
FATAL_MARKERS = ("not found", "invalid claim_id")


def decide_next(state: dict) -> str:
    step = int(state.get("step_count") or 0)
    max_steps = int(state.get("max_steps") or 12)
    if step > max_steps:
        return "loop_guard"
    intent = (state.get("intent") or {}).get("intent", "ambiguous")
    plan = state.get("plan") or PLANS.get(intent, ["clarify"])
    completed = state.get("completed") or []
    errors = state.get("errors") or []
    if any(m in str(e.get("error", "")).lower() for e in errors for m in FATAL_MARKERS) and "respond" not in completed:
        return "respond"
    for stepname in plan:
        if stepname not in completed:
            return stepname if stepname in ROUTABLE else "clarify"
    # plan exhausted without reaching a terminal: never spin -- close out via respond
    return "respond"


def route_from_supervisor(state: dict) -> str:
    """Conditional-edge function used by the graph."""
    return state.get("next") or decide_next(state)


def make_supervisor():
    @node_boundary("supervisor")
    async def supervisor(state: dict) -> dict:
        step = int(state.get("step_count") or 0) + 1
        nxt = decide_next({**state, "step_count": step})
        if nxt == "loop_guard":
            audit("loop_guard_tripped", decision="halted", agent="supervisor",
                  details={"step_count": step, "max_steps": state.get("max_steps")})
        return {"step_count": step, "next": nxt}
    return supervisor
