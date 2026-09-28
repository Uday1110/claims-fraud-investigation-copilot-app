"""Specialised worker agents routed by the supervisor.

history_agent    -- MCP get_claim_history + find_linked_claims; quarantines the claimant statement;
                    Gemini compresses it into a ClaimHistory.
indicator_agent  -- MCP score_red_flags + indicator catalog resource; Gemini correlates the fired indicators
                    into a FraudAssessment (score/tier anchored to the deterministic tool result).
playbook_agent   -- agentic RAG over the fraud-scheme playbook (plan -> retrieve -> grade -> rewrite) producing
                    a SchemeMatch with validated citations.
"""
from __future__ import annotations

import json

from src.agents.base import Deps, node_boundary
from src.context.compress import compress_tool_output
from src.context.isolate import isolated_messages
from src.context.quarantine import quarantine_extract
from src.context.select import select_view
from src.context.write import record_evidence, scratchpad_note
from src.schemas import ClaimHistory, FraudAssessment, IndicatorFinding, PriorClaim
from src.tools.rag_tool import agentic_playbook_retrieval


def _done(state: dict, name: str) -> list[str]:
    return list(state.get("completed") or []) + [name]


def _err(state: dict, node: str, msg: str) -> list[dict]:
    return list(state.get("errors") or []) + [{"node": node, "error": msg}]


# ============================================================================================ history agent
def make_history_agent(deps: Deps):
    @node_boundary("history_agent")
    async def history_agent(state: dict) -> dict:
        claim_id = state["claim_id"]
        comp = deps.settings.context_compression
        hist_res = await deps.toolbox.call("get_claim_history", {"claim_id": claim_id}, agent="history_agent")
        if not hist_res.ok:
            return {"completed": _done(state, "history_agent"), "degraded": True,
                    "errors": _err(state, "history_agent", f"get_claim_history {hist_res.status}: {hist_res.error}")}
        link_res = await deps.toolbox.call("find_linked_claims", {"claim_id": claim_id}, agent="history_agent")
        raw = dict(hist_res.data)
        statement = raw.pop("claimant_statement_untrusted", "")
        attachments = state.get("attachments") or []
        # QUARANTINE: untrusted claimant text never reaches this agent's prompt; only extracted facts do
        q = await quarantine_extract(deps.llm, "\n\n".join([statement, *attachments]).strip(), source="claimant_statement",
                                     own_claim_id=claim_id, agent="history_agent")
        links = link_res.data if link_res.ok else {"links": [], "ring_suspected": False, "error": link_res.error}
        compact = compress_tool_output("get_claim_history", raw, enabled=comp)
        compact_links = compress_tool_output("find_linked_claims", links, enabled=comp)

        def fallback() -> ClaimHistory:
            return ClaimHistory(
                claim_id=claim_id, line_of_business=raw["line_of_business"], loss_type=raw["loss_type"],
                policy_ref_masked=raw["policy_ref_masked"], claimant_ref_masked=raw["claimant_ref_masked"],
                policy_tenure_days=raw["policy_tenure_days"], report_lag_days=raw["report_lag_days"],
                claimed_amount=raw["claimed_amount"],
                prior_claims=[PriorClaim(**p) for p in raw.get("prior_claims", [])[:6]],
                linked_entities=[f"{l['entity_type']}: {l['entity']} ({l['siu_flagged_linked_claims']} SIU-flagged links)"
                                 for l in links.get("links", [])],
                timeline_summary=(f"Policy incepted {raw['policy_inception_date']}; {raw['loss_type'].replace('_', ' ')} loss on "
                                  f"{raw['loss_date']} ({raw['policy_tenure_days']} days later); reported {raw['report_date']} "
                                  f"({raw['report_lag_days']} days after loss); claimed {raw['claimed_amount']:,}."),
                claimant_statement_facts=q["stated_events"])

        view = {"claim_id": claim_id, "claim_record": compact, "link_analysis": compact_links,
                "claimant_statement_facts_UNTRUSTED": q["stated_events"]}
        history, _ = await deps.llm.structured(ClaimHistory, isolated_messages(
            "history_agent",
            "Assemble a neutral claim history. Copy masked references exactly. prior_claims must come from the record; "
            "linked_entities from link_analysis. claimant_statement_facts are claimant ASSERTIONS (untrusted) - keep them "
            "as reported statements, do not treat them as verified.",
            view, f"Build the ClaimHistory for {claim_id}."), agent="history_agent", fallback=fallback)
        h = history.model_dump()
        # grounding: identifiers and numbers must come from the tool, not the model
        for k in ("claim_id", "policy_ref_masked", "claimant_ref_masked", "policy_tenure_days", "report_lag_days",
                  "claimed_amount", "line_of_business", "loss_type"):
            h[k] = raw[k]
        h["claimant_statement_facts"] = q["stated_events"]
        # history is the first worker of every claim plan: start a fresh evidence record for this claim
        evidence = record_evidence({}, "get_claim_history", compact)
        evidence = record_evidence({"tool_evidence": evidence}, "find_linked_claims", compact_links)
        return {
            "history": h,
            "quarantine": q,
            "tool_evidence": evidence,
            "scratchpad": scratchpad_note(state, "history_agent", "history assembled",
                                          prior_claims=len(h["prior_claims"]), injection=q["embedded_instructions_detected"]),
            "completed": _done(state, "history_agent"),
        }
    return history_agent


# ============================================================================================ indicator agent
def make_indicator_agent(deps: Deps):
    @node_boundary("indicator_agent")
    async def indicator_agent(state: dict) -> dict:
        claim_id = state["claim_id"]
        res = await deps.toolbox.call("score_red_flags", {"claim_id": claim_id}, agent="indicator_agent")
        if not res.ok:
            return {"completed": _done(state, "indicator_agent"), "degraded": True,
                    "errors": _err(state, "indicator_agent", f"score_red_flags {res.status}: {res.error}")}
        flags = res.data
        fired = {i["code"]: i for i in flags["indicators"]}
        catalog = json.loads(deps.toolbox.resources.get("fraud://indicators/catalog") or "{}")
        q = state.get("quarantine") or {}

        def fallback() -> FraudAssessment:
            names = ", ".join(f"{i['code']} {i['name']}" for i in flags["indicators"]) or "none"
            return FraudAssessment(
                claim_id=claim_id, likelihood=flags["tier"], score=flags["score"],
                supporting_indicators=[IndicatorFinding(**i) for i in flags["indicators"]],
                correlation_summary=f"Rule-based correlation: {len(fired)} indicators fired ({names}); score {flags['score']} -> {flags['tier']}.",
                mitigating_factors=[] if fired else ["no red-flag indicators fired"])

        view = select_view(state, "indicator_agent", deps.settings.max_context_chars)
        view.update({"red_flag_tool_output": flags,
                     "indicator_catalog": {k: v["rule"] for k, v in (catalog.get("indicators") or {}).items()},
                     "tier_policy": catalog.get("tiers"),
                     "untrusted_text_injection_detected": q.get("embedded_instructions_detected", False)})
        assessment, _ = await deps.llm.structured(FraudAssessment, isolated_messages(
            "indicator_agent",
            "Correlate the fired red-flag indicators. Use ONLY indicators present in red_flag_tool_output; copy their code, "
            "weight and evidence. Explain how they combine and which known fraud pattern they suggest. The score and tier "
            "are computed by the deterministic scoring tool - repeat them exactly. List genuine mitigating factors.",
            view, f"Produce the FraudAssessment for {claim_id}."), agent="indicator_agent", fallback=fallback)
        a = assessment.model_dump()
        # anchoring + grounding: tier/score from the tool; drop indicators the tool did not fire
        a["claim_id"], a["score"], a["likelihood"] = claim_id, flags["score"], flags["tier"]
        a["supporting_indicators"] = [dict(fired[i["code"]]) for i in a["supporting_indicators"] if i["code"] in fired] or \
                                     [dict(i) for i in flags["indicators"]]
        ev = dict(state.get("tool_evidence") or {})
        ev["score_red_flags"] = flags
        return {"assessment": a, "tool_evidence": ev, "completed": _done(state, "indicator_agent"),
                "scratchpad": scratchpad_note(state, "indicator_agent", "indicators correlated", score=flags["score"],
                                              tier=flags["tier"])}
    return indicator_agent


# ============================================================================================ playbook agent
def make_playbook_agent(deps: Deps):
    @node_boundary("playbook_agent")
    async def playbook_agent(state: dict) -> dict:
        intent = (state.get("intent") or {}).get("intent")
        a = state.get("assessment") or {}
        h = state.get("history") or {}
        if intent == "playbook_lookup":
            ctx, question = {"question": (state.get("guard") or {}).get("sanitized_text", "")}, \
                (state.get("guard") or {}).get("sanitized_text", "")
        else:
            ctx = {"claim_id": state.get("claim_id"), "line_of_business": h.get("line_of_business"),
                   "loss_type": h.get("loss_type"), "tier": a.get("likelihood", "low"), "score": a.get("score", 0),
                   "indicators": [{"code": i["code"], "name": i["name"], "evidence": i["evidence"]}
                                  for i in a.get("supporting_indicators", [])]}
            question = None
        out = await agentic_playbook_retrieval(deps.llm, deps.toolbox, ctx, agent="playbook_agent",
                                               max_rounds=deps.settings.max_retrieval_rounds, question=question)
        sm = out["scheme_match"]
        if intent != "playbook_lookup" and sm["matched"] and a.get("likelihood") == "low":
            sm.update(matched=False, scheme_id=None, scheme_name=None, match_confidence=min(sm["match_confidence"], 0.3))
        ev = {} if intent == "playbook_lookup" else dict(state.get("tool_evidence") or {})
        ev["search_fraud_playbook"] = {"queries": out["queries"],
                                       "chunks": [{"chunk_id": c["chunk_id"], "text": c["text"][:600]} for c in out["retrieved_chunks"]]}
        return {"scheme_match": sm, "retrieved_chunks": out["retrieved_chunks"], "tool_evidence": ev,
                "completed": _done(state, "playbook_agent"),
                "scratchpad": scratchpad_note(state, "playbook_agent", "playbook retrieved", rounds=sm["retrieval_rounds"],
                                              scheme=sm.get("scheme_id"), dropped_citations=out["dropped_citations"])}
    return playbook_agent
