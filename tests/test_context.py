"""Context-engineering tests: write / select / compress / isolate / summarize."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from src.context.compress import compress_tool_output, trim_to_budget
from src.context.isolate import isolated_messages
from src.context.select import AGENT_VIEWS, select_view
from src.context.summarizer import summarize_if_needed
from src.context.write import record_evidence, scratchpad_note


def test_write_scratchpad_and_evidence_are_masked():
    pad = scratchpad_note({}, "history_agent", "note", ref="POL-12345678")
    assert pad[0]["ref"] == "POL-****5678"
    ev = record_evidence({}, "get_claim_history", {"p": "CLT-123456"})
    assert ev["get_claim_history"]["p"] == "CLT-****56"


def test_select_gives_each_agent_only_its_view():
    state = {"claim_id": "CLM-1", "history": {"claim_id": "CLM-1", "timeline_summary": "t", "prior_claims": [1] * 50},
             "assessment": {"score": 1}, "messages": ["secret"], "quarantine": {"stated_events": ["x"]}}
    v = select_view(state, "indicator_agent")
    assert set(v) <= set(AGENT_VIEWS["indicator_agent"]) and "messages" not in v and "quarantine" not in v
    v2 = select_view(state, "case_writer")
    assert "prior_claims" not in v2["history"]          # compressed history for downstream writers


def test_compress_tool_output_allow_list_and_list_caps():
    data = {"claim_id": "c", "score": 0, "indicators": [{"i": n} for n in range(20)], "debug_blob": "x" * 999}
    out = compress_tool_output("score_red_flags", data)
    assert "debug_blob" not in out and len(out["indicators"]) == 7 and out["indicators"][-1] == {"omitted": 14}
    assert out["score"] == 0


def test_trim_to_budget_keeps_recent():
    msgs = [HumanMessage("a" * 1000, id=str(i)) for i in range(10)]
    kept = trim_to_budget(msgs, max_chars=2500, keep_last=2)
    assert kept[-1].id == "9" and len(kept) == 2


def test_isolated_messages_have_no_shared_transcript():
    msgs = isolated_messages("history_agent", "role", {"claim_id": "CLM-1"}, "task")
    assert isinstance(msgs[0], SystemMessage) and len(msgs) == 2 and "UNTRUSTED" in msgs[0].content


async def test_summarization_middleware_folds_old_turns(offline_llm):
    msgs = []
    for i in range(6):
        msgs += [HumanMessage(f"investigator fact {i}", id=f"h{i}"), AIMessage(f"answer {i}", id=f"a{i}")]
    out = await summarize_if_needed(offline_llm, {"messages": msgs}, threshold=8)
    assert "investigator fact 0" in out["conversation_summary"]
    assert all(isinstance(m, RemoveMessage) for m in out["messages"]) and len(out["messages"]) == 8
    assert await summarize_if_needed(offline_llm, {"messages": msgs[:4]}, threshold=8) == {}
