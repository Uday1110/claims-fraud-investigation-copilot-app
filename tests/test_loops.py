"""Loop / cascade guard tests: a max-steps guard and LangGraph's recursion limit stop runaway loops."""
from __future__ import annotations

import pytest
from langgraph.errors import GraphRecursionError

from src.agents.base import node_boundary
from src.agents.supervisor import decide_next
from src.graph import build_graph
from src.guardrails.access import resolve_actor
from src.state import PLANS
from src.tools.mcp_client import open_toolbox
from src.tools.rag_tool import agentic_playbook_retrieval
from tests.conftest import make_deps


def test_supervisor_trips_loop_guard_when_step_budget_exhausted():
    s = {"intent": {"intent": "investigate_claim"}, "plan": PLANS["investigate_claim"], "completed": [],
         "step_count": 13, "max_steps": 12}
    assert decide_next(s) == "loop_guard"


def test_exhausted_plan_never_spins():
    s = {"intent": {"intent": "history_lookup"}, "plan": ["history_agent"], "completed": ["history_agent"],
         "step_count": 2, "max_steps": 12}
    assert decide_next(s) == "respond"


@node_boundary("history_agent")
async def runaway_worker(state: dict) -> dict:
    """Pathological worker that never reports completion -> supervisor would re-route to it forever."""
    return {"scratchpad": [{"note": "still working"}]}


async def test_runaway_worker_is_stopped_by_max_steps_guard(tmp_settings, evidence_dir):
    async with open_toolbox() as tb:
        g = build_graph(make_deps(tmp_settings, tb), overrides={"history_agent": runaway_worker})
        out = await g.ainvoke({"request": "Investigate CLM-1001", "actor": resolve_actor("investigator", "INV-07"),
                               "attachments": [], "max_steps": 5},
                              {"configurable": {"thread_id": "loop"}, "recursion_limit": 100})
    assert out["step_count"] == 6                       # 5 allowed steps + the one that trips the guard
    assert "maximum number of agent steps" in out["final_answer"]
    assert "loop_guard_tripped" in (evidence_dir / "agent_actions.jsonl").read_text("utf-8")


async def test_recursion_limit_is_the_framework_backstop(tmp_settings):
    async with open_toolbox() as tb:
        g = build_graph(make_deps(tmp_settings, tb), overrides={"history_agent": runaway_worker})
        with pytest.raises(GraphRecursionError):
            await g.ainvoke({"request": "Investigate CLM-1001", "actor": resolve_actor("investigator", "INV-07"),
                             "attachments": [], "max_steps": 10_000},
                            {"configurable": {"thread_id": "rec"}, "recursion_limit": 15})


class NeverSatisfiedLLM:
    """LLM stub whose relevance grader always asks for another retrieval round."""
    offline = False
    calls = 0

    async def structured(self, schema, messages, *, agent, fallback=None, optional=False):
        from src.tools.rag_tool import RetrievalJudgement, RetrievalPlan

        self.calls += 1
        if schema is RetrievalPlan:
            return RetrievalPlan(query="staged collision", rationale="t"), {}
        return RetrievalJudgement(sufficient=False, rewritten_query=f"retry {self.calls}", matched=False,
                                  match_confidence=0.1), {}


async def test_agentic_rag_retrieval_loop_is_bounded(tmp_settings):
    async with open_toolbox() as tb:
        out = await agentic_playbook_retrieval(NeverSatisfiedLLM(), tb, {"indicators": []}, agent="test", max_rounds=3)
    assert out["scheme_match"]["retrieval_rounds"] == 3
    assert len(out["queries"]) == 3
