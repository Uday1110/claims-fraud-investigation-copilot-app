"""Exit path: output_guard -> memory_writer -> (human_gate) -> END."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import interrupt

from src.agents.base import Deps, node_boundary
from src.agents.writers import render_case_file
from src.audit.audit import audit
from src.guardrails.access import can_sign_off
from src.guardrails.output_guard import check_case_file, guard_output
from src.memory.long_term import _FACTY, extract_facts
from src.observability.run_context import current_ids


def needs_signoff(state: dict) -> bool:
    intent = (state.get("intent") or {}).get("intent")
    if intent == "action_request":
        return True
    return intent == "investigate_claim" and bool((state.get("case_file") or {}).get("requires_human_signoff")) \
        and (state.get("human_review") or {}).get("status") in (None, "pending", "not_required")


# ============================================================================================ output guard
def make_output_guard(deps: Deps):
    @node_boundary("output_guard")
    async def output_guard(state: dict) -> dict:
        out: dict = {}
        violations: list[str] = []
        intent = (state.get("intent") or {}).get("intent")
        answer = state.get("final_answer") or "No answer produced."
        if intent == "investigate_claim" and state.get("case_file"):
            a = state.get("assessment") or {}
            fired = {i["code"] for i in a.get("supporting_indicators", [])}
            chunk_ids = {c["chunk_id"] for c in state.get("retrieved_chunks") or []}
            cf, v = check_case_file(state["case_file"], fired_codes=fired, retrieved_chunk_ids=chunk_ids,
                                    tier=a.get("likelihood", "low"))
            violations += v
            if v:
                out["case_file"] = cf
                answer = render_case_file(cf, state.get("scheme_match"))
        g = guard_output(answer, {**state, **out})
        violations += g["violations"]
        if g["blocked"]:
            audit("output_blocked", decision="blocked", agent="output_guard", details={"violations": violations})
        elif violations:
            audit("output_repaired", decision="repaired", agent="output_guard", details={"violations": violations})
        audit("answer_returned", decision=f"risk_tier:{g['risk_tier']}", agent="output_guard",
              details={"intent": intent, "chars": len(g["text"]), "degraded": bool(state.get("degraded"))})
        out.update({
            "final_answer": g["text"],
            "output_check": {"passed": not g["blocked"], "risk_tier": g["risk_tier"], "violations": violations,
                             "repaired": bool(violations) and not g["blocked"]},
            "messages": [AIMessage(g["text"])],
        })
        return out
    return output_guard


# ============================================================================================ memory writer
def make_memory_writer(deps: Deps):
    @node_boundary("memory_writer")
    async def memory_writer(state: dict) -> dict:
        actor = (state.get("actor") or {}).get("id", "anon")
        intent = (state.get("intent") or {}).get("intent")
        claim_id = state.get("claim_id")
        ids = current_ids()
        written = {"episode": False, "facts": 0, "method": None}
        cf = state.get("case_file") or {}
        if intent == "investigate_claim" and cf.get("claim_id") == claim_id:
            sm = state.get("scheme_match") or {}
            deps.memory.remember_episode(actor, claim_id, {
                "recommended_action": cf["recommended_action"], "likelihood": cf["fraud_likelihood"],
                "score": cf["fraud_score"], "scheme": sm.get("scheme_id"), "title": cf["title"],
                "summary": cf["executive_summary"][:500],
                "indicators": [s.split(":")[0] for s in cf["supporting_indicators"]][:10],
            }, status="pending_signoff" if cf.get("requires_human_signoff") else "completed",
                thread_id=ids["thread_id"], run_id=ids["run_id"])
            written["episode"] = True
        # semantic memory: durable facts the investigator stated this turn (LangMem manager)
        last_human = next((m for m in reversed(state.get("messages") or []) if isinstance(m, HumanMessage)), None)
        turn_text = str(last_human.content) if last_human else ""
        if (state.get("guard") or {}).get("allowed") and _FACTY.search(turn_text):
            facts, method = await extract_facts(deps.llm, [turn_text], claim_id)
            written["facts"] = deps.memory.remember_facts(actor, facts, claim_id=claim_id, source=method)
            written["method"] = method
        if written["episode"] or written["facts"]:
            audit("memory_written", decision="persisted", agent="memory_writer",
                  details={"claim_id": claim_id, **written})
        return {}
    return memory_writer


def route_after_memory(state: dict) -> str:
    return "request_signoff" if needs_signoff(state) else END


# ============================================================================================ human gate
def make_request_signoff(deps: Deps):
    """Runs exactly once per sign-off: audits the routing and marks the review pending (checkpointed).

    Kept separate from ``human_gate`` because LangGraph re-executes an interrupted node from its start on resume;
    anything side-effecting before ``interrupt()`` would otherwise run twice (see docs/failure-analysis.md, F-4).
    """
    @node_boundary("request_signoff")
    async def request_signoff(state: dict) -> dict:
        cf = state.get("case_file") or {}
        intent = (state.get("intent") or {}).get("intent")
        requested = (state.get("human_review") or {}).get("requested_action") if intent == "action_request" else cf.get("recommended_action")
        audit("routed_for_human_signoff", decision="awaiting_reviewer", agent="request_signoff",
              details={"claim_id": state.get("claim_id"), "requested_action": requested})
        return {"human_review": {"status": "pending", "requested_action": requested,
                                 "decision_note": (state.get("human_review") or {}).get("decision_note")}}
    return request_signoff


def make_human_gate(deps: Deps):
    @node_boundary("human_gate")
    async def human_gate(state: dict) -> dict:
        cf = state.get("case_file") or {}
        requested = (state.get("human_review") or {}).get("requested_action")
        claim_id = state.get("claim_id")
        payload = {"type": "human_signoff_required", "claim_id": claim_id, "requested_action": requested,
                   "fraud_likelihood": cf.get("fraud_likelihood"), "fraud_score": cf.get("fraud_score"),
                   "summary": (cf.get("executive_summary") or state.get("final_answer", ""))[:400],
                   "resume_with": {"decision": "approve|reject", "reviewer": "SIU-LEAD-01", "note": "..."}}
        note = None
        for _ in range(3):
            decision = interrupt(payload)            # graph pauses here; state is checkpointed
            decision = decision if isinstance(decision, dict) else {"decision": str(decision)}
            reviewer = decision.get("reviewer", "")
            if can_sign_off(reviewer) and decision.get("decision") in ("approve", "reject"):
                break
            note = f"reviewer {reviewer!r} not authorised to sign off or invalid decision"
            audit("access_refused", decision="signoff_refused", agent="human_gate", actor=reviewer or "unknown",
                  details={"reason": note})
            payload = {**payload, "error": note}
        else:
            return {"human_review": {"status": "pending", "requested_action": requested, "decision_note": note}}
        status = "approved" if decision["decision"] == "approve" else "rejected"
        audit("human_signoff_recorded", decision=status, agent="human_gate", actor=reviewer,
              details={"claim_id": claim_id, "requested_action": requested, "note": (decision.get("note") or "")[:200]})
        actor = (state.get("actor") or {}).get("id", "anon")
        if claim_id:
            deps.memory.update_episode_status(actor, claim_id, f"signoff_{status}", {"reviewer": reviewer})
        verdict = (f"Human sign-off recorded: {reviewer} {status} the recommendation '{requested}' for {claim_id}.")
        return {"human_review": {"status": status, "reviewer": reviewer, "decision_note": decision.get("note"),
                                 "requested_action": requested},
                "final_answer": (state.get("final_answer") or "") + f"\n\n---\n{verdict}",
                "messages": [AIMessage(verdict)]}
    return human_gate
