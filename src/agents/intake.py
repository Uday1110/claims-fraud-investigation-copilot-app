"""Entry nodes: input_guard -> load_context -> intent_router."""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import Deps, node_boundary
from src.audit.audit import audit
from src.context.select import select_view
from src.context.summarizer import summarize_if_needed
from src.guardrails.access import referenced_claims
from src.guardrails.input_guard import REFUSALS, check_input
from src.schemas import IntentDecision
from src.state import PER_TURN_FIELDS, PLANS

NEEDS_CLAIM = {"investigate_claim", "history_lookup", "indicator_check", "action_request"}


# --------------------------------------------------------------------------------------------- input guard
def make_input_guard(deps: Deps):
    @node_boundary("input_guard")
    async def input_guard(state: dict) -> dict:
        request = state.get("request", "")
        actor = state.get("actor") or {}
        reset = {k: None for k in PER_TURN_FIELDS}
        reset.update({"completed": [], "plan": [], "step_count": 0, "errors": [], "scratchpad": [],
                      "degraded": False, "max_steps": state.get("max_steps") or deps.settings.max_steps})
        audit("request_received", decision="evaluating", agent="input_guard",
              details={"chars": len(request), "attachments": len(state.get("attachments") or []), "role": actor.get("role")})
        g = check_input(request, actor)
        out = {**reset, "guard": g.model_dump(), "messages": [HumanMessage(g.sanitized_text or request[:200])]}
        if not g.allowed:
            action = "access_refused" if g.category == "cross_claimant_access" else "input_blocked"
            audit(action, decision="refused", agent="input_guard", details={"category": g.category, "reasons": g.reasons})
            out["final_answer"] = REFUSALS.get(g.category, "Request refused by input guardrail.")
            out["intent"] = {"intent": "out_of_scope", "claim_id": None, "confidence": 1.0,
                             "rationale": f"blocked by input guardrail: {g.category}"}
        return out
    return input_guard


def route_after_guard(state: dict) -> str:
    return "load_context" if (state.get("guard") or {}).get("allowed") else "refuse"


# --------------------------------------------------------------------------------------------- context
def make_load_context(deps: Deps):
    @node_boundary("load_context")
    async def load_context(state: dict) -> dict:
        out: dict = {}
        # summarisation middleware (short-term memory compression)
        out.update(await summarize_if_needed(deps.llm, state, deps.settings.summarize_after_messages))
        refs = referenced_claims(state.get("guard", {}).get("sanitized_text", ""))
        claim_id = refs[0] if refs else state.get("claim_id")
        out["claim_id"] = claim_id
        actor = state.get("actor") or {}
        mem = deps.memory.recall(actor.get("id", "anon"), state.get("request", ""), claim_id)
        out["memory_context"] = mem
        if mem["episodes"] or mem["facts"]:
            audit("memory_recalled", decision="loaded", agent="load_context",
                  details={"episodes": [e["claim_id"] for e in mem["episodes"]], "facts": len(mem["facts"])})
        return out
    return load_context


# --------------------------------------------------------------------------------------------- intent router
_KW = [
    ("action_request", re.compile(r"\b(deny|reject|decline|approve|pay|authori[sz]e payment|close the claim|settle|escalate)\b", re.I)),
    ("out_of_scope", re.compile(r"\b(poem|joke|weather|recipe|stock|bitcoin|football score|write me a song|translate)\b", re.I)),
    ("followup", re.compile(r"\b(last time|previous(ly)?|earlier|again|we discussed|you said|i (told|mentioned)|remind me|where did we land|what did)\b", re.I)),
    ("playbook_lookup", re.compile(r"\b(playbook|scheme|how (do|should) (we|i) investigate|what (is|are) (a |the )?.*(fraud|scheme))\b", re.I)),
    ("history_lookup", re.compile(r"\b(history|prior claims|timeline)\b", re.I)),
    ("indicator_check", re.compile(r"\b(red.?flags?|indicators?|score|likelihood)\b", re.I)),
    ("investigate_claim", re.compile(r"\b(investigat\w*|case file|assess|review|analy[sz]e|look into|compile)\b", re.I)),
]


def heuristic_intent(request: str, claim_id: str | None, has_case: bool) -> IntentDecision:
    """Deterministic fallback classifier (used when Gemini is unavailable). Claim-aware rule order."""
    kw = dict(_KW)

    def d(intent: str, why: str, conf: float = 0.6) -> IntentDecision:
        return IntentDecision(intent=intent, claim_id=claim_id, confidence=conf, rationale=f"keyword rule: {why}")

    for intent in ("action_request", "out_of_scope"):
        if kw[intent].search(request):
            return d(intent, intent)
    if kw["followup"].search(request) and (has_case or claim_id):
        return d("followup", "followup")
    if claim_id:
        for intent in ("history_lookup", "indicator_check", "investigate_claim"):
            if kw[intent].search(request):
                return d(intent, intent)
        return d("investigate_claim", "claim id only", 0.5)
    if kw["playbook_lookup"].search(request) or re.search(r"\b(red.?flags?|indicators?|investigate a suspected)\b", request, re.I):
        return d("playbook_lookup", "scheme question without claim")
    return IntentDecision(intent="ambiguous", claim_id=None, confidence=0.4,
                          clarifying_question="Which claim (CLM-NNNN) should I look at, and do you want a full case file, "
                                              "the claim history, or the red-flag check?",
                          rationale="no recognisable task")


INTENT_SYSTEM = """You classify requests sent to an insurance SIU (special investigation unit) fraud copilot.
Intents:
- investigate_claim: build a full investigation case file for a flagged claim (e.g. "Investigate CLM-1001", "compile the case", "does CLM-1004 match a fraud scheme?")
- history_lookup: only the claim/claimant history or timeline
- indicator_check: only the red flags / fraud score / likelihood
- playbook_lookup: a question about a fraud scheme or investigation procedure, not about a specific claim's facts
- followup: a question about a case already discussed in this conversation or a previous session ("what did we recommend last time?", "given what I told you ...")
- action_request: asks the copilot to deny, reject, approve, pay, settle or escalate a claim (the copilot cannot do this; a human must)
- ambiguous: an SIU task but unclear which claim or what output -> ask a clarifying question
- out_of_scope: not an SIU / claims-fraud task
Return claim_id exactly as written (CLM-NNNN) if present, else the thread's current claim if the request clearly refers to it."""


def make_intent_router(deps: Deps):
    @node_boundary("intent_router")
    async def intent_router(state: dict) -> dict:
        request = (state.get("guard") or {}).get("sanitized_text") or state.get("request", "")
        claim_id = state.get("claim_id")
        has_case = bool(state.get("case_file"))
        view = select_view(state, "intent_router", deps.settings.max_context_chars)
        view["thread_has_case_file_for"] = (state.get("case_file") or {}).get("claim_id")
        decision, meta = await deps.llm.structured(IntentDecision, [
            SystemMessage(INTENT_SYSTEM),
            HumanMessage(f"REQUEST: {request}\nTHREAD CONTEXT: {view}"),
        ], agent="intent_router", fallback=lambda: heuristic_intent(request, claim_id, has_case))

        d = decision.model_dump()
        if d.get("claim_id"):
            d["claim_id"] = d["claim_id"].upper()
            refs = referenced_claims(request)
            if refs and d["claim_id"] not in refs:        # never trust an LLM-invented claim id
                d["claim_id"] = refs[0]
        d["claim_id"] = d.get("claim_id") or claim_id
        if d["intent"] in NEEDS_CLAIM and not d["claim_id"]:
            d.update(intent="ambiguous", clarifying_question=d.get("clarifying_question") or
                     "Which claim (CLM-NNNN) should I work on?")
        if d["intent"] == "followup" and not (has_case or d["claim_id"] or (state.get("memory_context") or {}).get("episodes")):
            d.update(intent="ambiguous", clarifying_question="I don't have an earlier case in context. Which claim do you mean?")
        allowed = set((state.get("actor") or {}).get("authorized_claims") or [])
        if d["claim_id"] and d["claim_id"] not in allowed:
            audit("access_refused", decision="refused", agent="intent_router", details={"claim_id": d["claim_id"]})
            d.update(intent="out_of_scope", rationale="claim not in actor's authorised queue")
            return {"intent": d, "plan": ["out_of_scope"], "claim_id": None,
                    "final_answer": "I can't access that claim: it is not assigned to your queue."}
        audit("intent_classified", decision=d["intent"], agent="intent_router",
              details={"claim_id": d["claim_id"], "confidence": d["confidence"], "degraded": meta.get("degraded", False)})
        return {"intent": d, "plan": PLANS[d["intent"]], "claim_id": d["claim_id"],
                "degraded": bool(state.get("degraded")) or meta.get("degraded", False)}
    return intent_router
