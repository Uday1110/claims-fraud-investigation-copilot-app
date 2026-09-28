"""Output guardrails -- wired into the graph's ``output_guard`` node; every answer passes through here.

Checks (and repairs where safe):
* PIILeak                 -> mask any raw policy number / claimant id / phone / email (repair)
* AutoAdverseAction       -> the copilot may *recommend*; text claiming a claim was denied / paid / closed is
                             rewritten and the case is forced to human sign-off (repair)
* PolicyConsistency       -> recommended action must match the GL-01 tier policy (repair to policy)
* HumanSignoffRequired    -> refer_to_siu always requires human sign-off (repair)
* GroundedIndicators      -> indicator codes cited must be ones the red-flag tool actually fired (drop others)
* ResolvableCitations     -> playbook citations must resolve to chunks retrieved in this run (drop others)
* SystemLeak              -> block answers that echo system/role prompts
"""
from __future__ import annotations

import re
from typing import Any

from src.guardrails.output_risk import classify_output_risk
from src.guardrails.pii import contains_raw_pii, mask_text

TIER_ACTION = {"low": "close", "medium": "monitor", "high": "refer_to_siu"}
ADVERSE_RE = re.compile(
    r"\b(claim (has been|is|was) (denied|rejected|declined)|we (have )?(denied|rejected)|payment (has been |is )?"
    r"(stopped|withheld|suspended)|(approved|authori[sz]ed) for payment|claim (is|has been) (approved|paid))\b", re.I)
SYSTEM_LEAK_RE = re.compile(r"(ROLE \((history|indicator|playbook|case_writer|intent)|You are a QUARANTINED reader)", re.I)
RF_RE = re.compile(r"RF-\d{2}")


def check_case_file(cf: dict, *, fired_codes: set[str], retrieved_chunk_ids: set[str], tier: str) -> tuple[dict, list[str]]:
    violations: list[str] = []
    cf = dict(cf)
    policy_action = TIER_ACTION.get(tier)
    if policy_action and cf.get("recommended_action") != policy_action:
        violations.append(f"PolicyConsistency: action {cf.get('recommended_action')} != GL-01 policy {policy_action} for tier {tier}")
        cf["recommended_action"] = policy_action
        cf["action_rationale"] = (cf.get("action_rationale", "") + f" [Adjusted to GL-01 policy: {tier} -> {policy_action}.]").strip()
    if cf.get("fraud_likelihood") != tier:
        violations.append(f"PolicyConsistency: likelihood {cf.get('fraud_likelihood')} != tool tier {tier}")
        cf["fraud_likelihood"] = tier
    if cf.get("recommended_action") == "refer_to_siu" and not cf.get("requires_human_signoff"):
        violations.append("HumanSignoffRequired: refer_to_siu without sign-off flag")
    cf["requires_human_signoff"] = cf.get("recommended_action") == "refer_to_siu" or bool(cf.get("requires_human_signoff"))

    kept = []
    for s in cf.get("supporting_indicators", []):
        codes = set(RF_RE.findall(s))
        if codes and not codes <= fired_codes:
            violations.append(f"GroundedIndicators: dropped ungrounded indicator '{s[:60]}'")
            continue
        kept.append(s)
    cf["supporting_indicators"] = kept

    cits = []
    for c in cf.get("playbook_citations", []):
        cid = c.split(" ")[0] if " " in c else c
        doc = cid.split("#")[0]
        if any(r.startswith(doc) for r in retrieved_chunk_ids):
            cits.append(c)
        else:
            violations.append(f"ResolvableCitations: dropped unresolvable citation '{c}'")
    cf["playbook_citations"] = cits
    return cf, violations


def check_text(text: str) -> tuple[str, list[str], bool]:
    """Returns (safe_text, violations, blocked)."""
    violations = []
    if SYSTEM_LEAK_RE.search(text):
        return ("I can't share internal configuration details.", ["SystemLeak: blocked"], True)
    leaks = contains_raw_pii(text)
    if leaks:
        violations.append(f"PIILeak: masked {leaks}")
        text = mask_text(text)
    if ADVERSE_RE.search(text):
        violations.append("AutoAdverseAction: rewritten to recommendation pending human sign-off")
        text = ADVERSE_RE.sub("[adverse action requires human SIU sign-off - copilot recommendation only]", text)
    return text, violations, False


def guard_output(answer: str, state: dict[str, Any]) -> dict[str, Any]:
    safe, violations, blocked = check_text(answer)
    risk = classify_output_risk(state)
    return {"text": safe, "violations": violations, "blocked": blocked, "risk_tier": risk["tier"], "risk": risk}
