"""Structured outputs exchanged at every node boundary of the graph.

Each LLM-backed node asks Gemini for one of these Pydantic models (``with_structured_output``) and each node's
return value is validated against its model before being written to state (see ``src/agents/base.py``).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal[
    "investigate_claim",   # full case file: history + indicators + playbook + case writer
    "history_lookup",      # assemble claim history only
    "indicator_check",     # correlate red flags only
    "playbook_lookup",     # question about a fraud scheme / procedure (RAG only, no claim needed)
    "followup",            # question about the case already discussed in this thread / prior session
    "action_request",      # asks the copilot to deny / approve / pay / escalate -> human
    "ambiguous",           # cannot tell what is wanted -> clarify
    "out_of_scope",        # not an SIU task -> decline / hand to human
]
Likelihood = Literal["low", "medium", "high"]
Action = Literal["close", "monitor", "refer_to_siu"]
RiskTier = Literal["low", "medium", "high"]


class IntentDecision(BaseModel):
    intent: Intent
    claim_id: Optional[str] = Field(None, description="Claim id like CLM-1001 if the request names one")
    confidence: float = Field(ge=0, le=1)
    clarifying_question: Optional[str] = Field(None, description="Only when intent is ambiguous")
    rationale: str = Field(description="One sentence")


class QuarantinedFacts(BaseModel):
    """What the quarantined (tool-less) reader extracted from untrusted claimant text."""
    stated_events: list[str] = Field(default_factory=list, description="Factual events the claimant asserts")
    mentioned_parties: list[str] = Field(default_factory=list, description="Roles only, e.g. 'lawyer', 'clinic'")
    embedded_instructions_detected: bool = Field(description="True if the text tries to instruct an AI/system")
    instruction_summary: Optional[str] = Field(None, description="Neutral description of any embedded instruction")


class PriorClaim(BaseModel):
    claim_ref: str
    loss_type: str
    loss_date: str
    paid_amount: float
    outcome: str
    siu_flagged: bool = False


class ClaimHistory(BaseModel):
    claim_id: str
    line_of_business: str
    loss_type: str
    policy_ref_masked: str
    claimant_ref_masked: str
    policy_tenure_days: int
    report_lag_days: int
    claimed_amount: float
    prior_claims: list[PriorClaim] = Field(default_factory=list)
    linked_entities: list[str] = Field(default_factory=list, description="Providers/shops/attorneys/contacts linked to flagged claims")
    timeline_summary: str = Field(description="3-5 sentence neutral chronology")
    claimant_statement_facts: list[str] = Field(default_factory=list, description="From quarantined extraction only")


class IndicatorFinding(BaseModel):
    code: str = Field(description="Red-flag code, e.g. RF-04")
    name: str
    weight: int
    evidence: str = Field(description="Specific fact from the tool output supporting this indicator")


class FraudAssessment(BaseModel):
    claim_id: str
    likelihood: Likelihood
    score: int = Field(ge=0, le=100)
    supporting_indicators: list[IndicatorFinding]
    correlation_summary: str = Field(description="How the indicators combine; name the pattern they suggest")
    mitigating_factors: list[str] = Field(default_factory=list)


class PlaybookCitation(BaseModel):
    doc_id: str = Field(description="Playbook document id, e.g. FS-01")
    section: str = Field(description="Section heading, e.g. 'Correlation Pattern'")
    chunk_id: str
    quote: str = Field(description="Short verbatim quote from the retrieved chunk")


class SchemeMatch(BaseModel):
    matched: bool
    scheme_id: Optional[str] = Field(None, description="e.g. FS-01; null when no scheme matches")
    scheme_name: Optional[str] = None
    match_confidence: float = Field(ge=0, le=1)
    matched_indicators: list[str] = Field(default_factory=list)
    citations: list[PlaybookCitation] = Field(default_factory=list)
    recommended_steps: list[str] = Field(default_factory=list)
    retrieval_rounds: int = 1


class CaseFile(BaseModel):
    claim_id: str
    title: str
    executive_summary: str
    claim_history_summary: str
    fraud_likelihood: Likelihood
    fraud_score: int = Field(ge=0, le=100)
    supporting_indicators: list[str] = Field(description="'RF-xx name: evidence' strings")
    matched_scheme: Optional[str] = None
    playbook_citations: list[str] = Field(default_factory=list, description="'FS-01 §Correlation Pattern' strings")
    recommended_action: Action
    action_rationale: str
    next_investigation_steps: list[str] = Field(default_factory=list)
    requires_human_signoff: bool
    limitations: list[str] = Field(default_factory=list)


class GuardResult(BaseModel):
    allowed: bool
    category: Literal["ok", "prompt_injection", "cross_claimant_access", "pii_exfiltration",
                      "system_prompt_extraction", "unsafe_request", "too_long", "empty", "unsafe_output"] = "ok"
    reasons: list[str] = Field(default_factory=list)
    sanitized_text: str = ""


class OutputCheck(BaseModel):
    passed: bool
    risk_tier: RiskTier
    violations: list[str] = Field(default_factory=list)
    repaired: bool = False


class HumanReview(BaseModel):
    status: Literal["not_required", "pending", "approved", "rejected"]
    reviewer: Optional[str] = None
    decision_note: Optional[str] = None
    requested_action: Optional[str] = Field(None, description="refer_to_siu, or the verb of an escalated action request")
