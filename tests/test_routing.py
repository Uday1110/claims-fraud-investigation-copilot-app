"""Routing-logic tests: the supervisor's conditional edges route the right worker for a given state."""
from __future__ import annotations

import pytest
from langgraph.graph import END

from src.agents.finalize import route_after_memory
from src.agents.intake import heuristic_intent, route_after_guard
from src.agents.supervisor import decide_next, route_from_supervisor
from src.state import PLANS
from src.tools.mcp_client import open_toolbox
from tests.conftest import make_deps


def st(intent: str, completed=(), **kw) -> dict:
    return {"intent": {"intent": intent}, "plan": PLANS[intent], "completed": list(completed), "step_count": 1,
            "max_steps": 12, **kw}


@pytest.mark.parametrize("completed,expected", [
    ((), "history_agent"),
    (("history_agent",), "indicator_agent"),
    (("history_agent", "indicator_agent"), "playbook_agent"),
    (("history_agent", "indicator_agent", "playbook_agent"), "case_writer"),
])
def test_investigation_plan_routes_workers_in_order(completed, expected):
    assert decide_next(st("investigate_claim", completed)) == expected


@pytest.mark.parametrize("intent,completed,expected", [
    ("history_lookup", (), "history_agent"),
    ("history_lookup", ("history_agent",), "respond"),
    ("indicator_check", ("history_agent",), "indicator_agent"),
    ("indicator_check", ("history_agent", "indicator_agent"), "respond"),
    ("playbook_lookup", (), "playbook_agent"),
    ("playbook_lookup", ("playbook_agent",), "respond"),
    ("followup", (), "respond"),
    ("action_request", (), "escalate_human"),
    ("ambiguous", (), "clarify"),
    ("out_of_scope", (), "out_of_scope"),
])
def test_each_intent_routes_to_the_right_capability(intent, completed, expected):
    assert decide_next(st(intent, completed)) == expected


def test_fatal_tool_error_short_circuits_to_respond():
    s = st("investigate_claim", ("history_agent",),
           errors=[{"node": "history_agent", "error": "get_claim_history error: claim CLM-1099 not found in SIU queue"}])
    assert decide_next(s) == "respond"


def test_conditional_edge_function_uses_supervisor_decision():
    assert route_from_supervisor({"next": "playbook_agent"}) == "playbook_agent"
    assert route_from_supervisor(st("investigate_claim")) == "history_agent"


def test_guard_edge_routes_blocked_input_to_refuse():
    assert route_after_guard({"guard": {"allowed": False}}) == "refuse"
    assert route_after_guard({"guard": {"allowed": True}}) == "load_context"


@pytest.mark.parametrize("state,expected", [
    ({"intent": {"intent": "investigate_claim"}, "case_file": {"requires_human_signoff": True}}, "request_signoff"),
    ({"intent": {"intent": "investigate_claim"}, "case_file": {"requires_human_signoff": False}}, END),
    ({"intent": {"intent": "action_request"}}, "request_signoff"),
    ({"intent": {"intent": "playbook_lookup"}}, END),
])
def test_human_gate_edge(state, expected):
    assert route_after_memory(state) == expected


@pytest.mark.parametrize("text,claim,expected", [
    ("Investigate CLM-1001 please", "CLM-1001", "investigate_claim"),
    ("Deny claim CLM-1007 now", "CLM-1007", "action_request"),
    ("Show the claim history for CLM-1002", "CLM-1002", "history_lookup"),
    ("What are the red flags on CLM-1004?", "CLM-1004", "indicator_check"),
    ("What does the playbook say about arson?", None, "playbook_lookup"),
    ("Write me a poem", None, "out_of_scope"),
    ("Can you look into that one?", None, "ambiguous"),
])
def test_heuristic_intent_fallback(text, claim, expected):
    assert heuristic_intent(text, claim, has_case=False).intent == expected


async def _node_sequence(request: str, settings, actor="INV-07") -> list[str]:
    from src.graph import build_graph
    from src.guardrails.access import resolve_actor

    async with open_toolbox() as tb:
        g = build_graph(make_deps(settings, tb))
        seq = []
        async for upd in g.astream({"request": request, "actor": resolve_actor("investigator", actor), "attachments": []},
                                   {"configurable": {"thread_id": "t"}}, stream_mode="updates"):
            seq.extend(k for k in upd if not k.startswith("__"))
        return seq


async def test_graph_executes_supervisor_worker_sequence_for_investigation(tmp_settings):
    seq = await _node_sequence("Investigate CLM-1002", tmp_settings)
    workers = [n for n in seq if n in ("history_agent", "indicator_agent", "playbook_agent", "case_writer")]
    assert workers == ["history_agent", "indicator_agent", "playbook_agent", "case_writer"]
    assert seq[0] == "input_guard" and seq.count("supervisor") == 4
    assert seq[-1] == "memory_writer"          # medium tier -> monitor -> no human gate


async def test_graph_routes_blocked_request_around_all_workers(tmp_settings):
    seq = await _node_sequence("Investigate CLM-1003", tmp_settings, actor="INV-12")
    assert "refuse" in seq and not {"history_agent", "indicator_agent", "playbook_agent"} & set(seq)


async def test_human_gate_audits_routing_exactly_once_across_interrupt_and_resume(tmp_settings, evidence_dir):
    """Regression for failure F-4: LangGraph re-runs an interrupted node on resume; the routing audit must not repeat."""
    import json

    from src.runner import Copilot

    async with Copilot(tmp_settings, offline=True) as cp:
        r = await cp.ask("Investigate CLM-1004", thread_id="gate-once", actor_id="INV-07")
        assert r["pending_signoff"] and r["human_review"]["status"] == "pending"
        assert (await cp.ask("anything", thread_id="gate-once"))["run_id"] is None      # thread blocked while pending
        done = await cp.review("gate-once", decision="reject", reviewer="SIU-LEAD-01", note="needs more evidence")
    assert done["human_review"]["status"] == "rejected"
    recs = [json.loads(l) for l in (evidence_dir / "agent_actions.jsonl").read_text("utf-8").splitlines()]
    mine = [x for x in recs if x["thread_id"] == "gate-once"]
    assert sum(x["action"] == "routed_for_human_signoff" for x in mine) == 1
    assert sum(x["action"] == "human_signoff_recorded" for x in mine) == 1
