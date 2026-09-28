"""Output-risk classification (see docs/output-risk.md).

| tier   | outputs                                                                      | gate                                  |
|--------|------------------------------------------------------------------------------|---------------------------------------|
| high   | refer_to_siu case files; any deny / suspend / escalate request               | human-in-the-loop interrupt (sign-off)|
| medium | monitor case files; indicator / history answers exposing claim facts         | output guard + audit, masked ids      |
| low    | close recommendations, playbook answers, clarifications, refusals            | output guard                          |
"""
from __future__ import annotations

from typing import Any


def classify_output_risk(state: dict[str, Any]) -> dict[str, Any]:
    intent = (state.get("intent") or {}).get("intent")
    cf = state.get("case_file") or {}
    action = cf.get("recommended_action")
    if intent == "action_request" or (intent == "investigate_claim" and action == "refer_to_siu"):
        return {"tier": "high", "gate": "human_signoff", "reason": "adverse / escalation recommendation"}
    if (intent == "investigate_claim" and action == "monitor") or intent in ("history_lookup", "indicator_check", "followup"):
        return {"tier": "medium", "gate": "output_guard+audit", "reason": "claim-specific facts or monitoring action"}
    return {"tier": "low", "gate": "output_guard", "reason": "no adverse action; general or closing output"}
