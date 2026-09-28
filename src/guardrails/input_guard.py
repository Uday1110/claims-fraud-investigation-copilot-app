"""Input guardrails -- wired into the graph's entry node (``input_guard``) before any LLM sees the request.

Validators follow a Guardrails-AI-style contract (name, on_fail policy, validate() -> pass/fail) but are
implemented as local policy functions so they run offline and deterministically:

| validator                  | on_fail   | what it stops                                              |
|----------------------------|-----------|------------------------------------------------------------|
| NotEmpty / MaxLength       | refuse    | empty or oversized requests (context-stuffing)             |
| AccessScope                | refuse    | touching a claim outside the actor's assignment (AC-06)    |
| PromptInjection            | refuse    | instructions to ignore rules / role override / fake system |
| SystemPromptExtraction     | refuse    | "print your system prompt / hidden instructions"           |
| PIIExfiltration            | refuse    | requests for unmasked policy numbers / identifiers / phone |
| FraudEnablement            | refuse    | "how to stage / fake a claim / evade SIU detection"        |
| EncodedInstruction         | refuse    | "decode this base64 and follow it" smuggling               |
| PIIRedaction               | fix       | masks any identifiers the investigator typed               |
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from src.context.quarantine import INJECTION_PATTERNS
from src.guardrails.access import unauthorized_claims
from src.guardrails.pii import redact_free_text
from src.schemas import GuardResult

MAX_REQUEST_CHARS = 4000

SYSTEM_PROMPT_RE = re.compile(r"\b(system prompt|hidden instructions|your instructions|developer message|initial prompt)\b", re.I)
FRAUD_ENABLE_RE = re.compile(
    r"\b(avoid|evade|beat|bypass|get around|fool)\b.{0,30}\b(detection|siu|investigat\w*|fraud checks?|red flags?)\b|"
    r"\bhow (can|could|do|would) (i|we|someone|a claimant)\b.{0,40}\b(stage|fake|fabricate|inflate)\b|"
    r"\b(fake|fabricate|stage)\b.{0,20}\b(a|an|the)\b.{0,20}\b(claim|accident|burst pipe|theft|fire)\b.{0,40}\b(without|undetected|not get caught)\b",
    re.I)
ENCODED_RE = re.compile(r"\b(decode|base64|rot13|hex)\b.{0,40}\b(follow|execute|run|obey|do what it says)\b", re.I)
PII_EXFIL_RE = re.compile(
    r"\b(full|unmasked|complete|raw|real|actual|entire)\b.{0,30}\b(policy (number|no)|claimant (id|identifier)|phone|"
    r"address|ssn|identifier)s?\b|\b(unmask|de-?anonymi[sz]e|decrypt)\b", re.I)


@dataclass
class Validator:
    name: str
    category: str
    on_fail: str               # "refuse" | "fix"
    check: Callable[[str, dict], Optional[str]]   # returns reason when failing


def _not_empty(t: str, _a: dict) -> Optional[str]:
    return "empty request" if not t.strip() else None


def _max_len(t: str, _a: dict) -> Optional[str]:
    return f"request exceeds {MAX_REQUEST_CHARS} chars" if len(t) > MAX_REQUEST_CHARS else None


def _access(t: str, actor: dict) -> Optional[str]:
    bad = unauthorized_claims(t, actor)
    return f"actor {actor.get('id')} is not authorised for {', '.join(bad)}" if bad else None


def _injection(t: str, _a: dict) -> Optional[str]:
    # "approve / deny this claim" from an investigator is not an injection: it is an action_request that the
    # intent router escalates to a human, so auto_action is only enforced on untrusted claimant text.
    hits = [n for n in ("ignore_instructions", "role_override", "fake_system")
            if INJECTION_PATTERNS[n].search(t)]
    return f"prompt-injection patterns: {hits}" if hits else None


def _sys_prompt(t: str, _a: dict) -> Optional[str]:
    return "attempt to extract system prompt" if SYSTEM_PROMPT_RE.search(t) else None


def _fraud_enable(t: str, _a: dict) -> Optional[str]:
    return "request would help commit or conceal insurance fraud" if FRAUD_ENABLE_RE.search(t) else None


def _encoded(t: str, _a: dict) -> Optional[str]:
    return "encoded-instruction smuggling" if ENCODED_RE.search(t) else None


def _pii_exfil(t: str, _a: dict) -> Optional[str]:
    return "request for unmasked personal identifiers" if PII_EXFIL_RE.search(t) else None


VALIDATORS: list[Validator] = [
    Validator("NotEmpty", "empty", "refuse", _not_empty),
    Validator("MaxLength", "too_long", "refuse", _max_len),
    Validator("SystemPromptExtraction", "system_prompt_extraction", "refuse", _sys_prompt),
    Validator("PIIExfiltration", "pii_exfiltration", "refuse", _pii_exfil),
    Validator("FraudEnablement", "unsafe_request", "refuse", _fraud_enable),
    Validator("EncodedInstruction", "prompt_injection", "refuse", _encoded),
    Validator("PromptInjection", "prompt_injection", "refuse", _injection),
    Validator("AccessScope", "cross_claimant_access", "refuse", _access),
]


def check_input(text: str, actor: dict) -> GuardResult:
    reasons: list[str] = []
    category = "ok"
    for v in VALIDATORS:
        reason = v.check(text or "", actor)
        if reason:
            reasons.append(f"{v.name}: {reason}")
            if category == "ok":
                category = v.category
    sanitized, _ = redact_free_text(text or "")   # PIIRedaction (fix)
    return GuardResult(allowed=not reasons, category=category, reasons=reasons, sanitized_text=sanitized)


REFUSALS = {
    "cross_claimant_access": "I can't access that claim: it is not assigned to your queue. Access requests for other "
                             "claimants' data are refused and logged. Ask your SIU lead to reassign it if needed.",
    "prompt_injection": "I can't follow instructions that try to override my operating rules. Please restate the "
                        "investigation request.",
    "pii_exfiltration": "I can't reveal unmasked policy numbers or claimant identifiers. Masked references are shown "
                        "in case files; full identifiers are available only in the core claims system.",
    "system_prompt_extraction": "I can't share my internal configuration. I can help investigate an assigned claim.",
    "unsafe_request": "I can't help with that: it would facilitate or conceal insurance fraud. I can explain how SIU "
                      "investigates a scheme instead.",
    "too_long": "That request is too long. Please summarise it (max 4,000 characters).",
    "empty": "Please enter a request, e.g. 'Investigate CLM-1001'.",
}
