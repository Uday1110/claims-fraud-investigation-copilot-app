"""Synthetic claims data access + red-flag indicator engine used by the MCP server.

All identifiers leaving this module are masked (data minimisation at source): the LLM, the logs and the traces
only ever see ``POL-****1234`` / ``CLT-****56`` style references.
"""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_DIR = ROOT / "data" / "claims"
CORPUS_DIR = ROOT / "data" / "policy_corpus"

CLAIM_ID_RE = re.compile(r"^CLM-\d{4}$")
INJECTION_RE = re.compile(
    r"(ignore (all )?(previous|prior) instructions|system note to ai|you are now|disregard .*instructions|"
    r"mark this claim as approved|print the full policy number)", re.I)

# Red-flag indicator catalog (exposed as MCP resource fraud://indicators/catalog)
INDICATORS: dict[str, dict] = {
    "RF-01": {"name": "NEW_POLICY_LOSS", "weight": 15, "rule": "loss within 60 days of policy inception"},
    "RF-02": {"name": "LATE_REPORTING", "weight": 8, "rule": "reported more than 14 days after loss"},
    "RF-03": {"name": "EARLY_ATTORNEY", "weight": 12, "rule": "attorney retained within 3 days of loss"},
    "RF-04": {"name": "MULTI_SOFT_TISSUE", "weight": 15, "rule": "3+ soft-tissue injury claimants in one loss"},
    "RF-05": {"name": "NO_POLICE_REPORT", "weight": 8, "rule": "auto collision/theft without police report"},
    "RF-06": {"name": "CLAIM_ESTIMATE_GAP", "weight": 10, "rule": "claimed amount >= 1.8x independent estimate"},
    "RF-07": {"name": "LINKED_ENTITY_RING", "weight": 14, "rule": "provider/shop/attorney/contractor linked to 3+ SIU-flagged claims"},
    "RF-08": {"name": "FREQUENT_CLAIMANT", "weight": 10, "rule": "2+ prior claims in the last 36 months"},
    "RF-09": {"name": "COVERAGE_INCREASE", "weight": 18, "rule": "coverage increased >=20% within 90 days before loss"},
    "RF-10": {"name": "FINANCIAL_DISTRESS", "weight": 12, "rule": "3+ late premiums in 12 months or negative vehicle equity"},
    "RF-11": {"name": "BILLING_ANOMALY", "weight": 16, "rule": "billing on closed dates / upcoded items (16 per anomaly type)"},
    "RF-12": {"name": "UNWITNESSED_INJURY", "weight": 8, "rule": "unwitnessed workplace injury"},
    "RF-13": {"name": "DUPLICATE_CLAIM", "weight": 45, "rule": "cross-carrier index match on claimant + loss date"},
    "RF-14": {"name": "EVIDENCE_INCONSISTENCY", "weight": 14, "rule": "adjuster evidence contradicts account (14 per note, max 28)"},
    "RF-15": {"name": "NO_WITNESSES", "weight": 4, "rule": "auto/property loss with no independent witness"},
    "RF-16": {"name": "TEXT_INJECTION_ATTEMPT", "weight": 6, "rule": "claimant-supplied text contains instructions aimed at the AI"},
    "RF-17": {"name": "SHARED_CONTACT", "weight": 10, "rule": "claimant phone shared with another claimant in the book"},
    "RF-18": {"name": "LARGE_LOSS", "weight": 5, "rule": "claimed amount >= 100,000"},
}
TIERS = {"low": (0, 19), "medium": (20, 44), "high": (45, 100)}
EVIDENCE_KEYWORDS = ("predate", "inconsistent", "accelerant", "moved out", "undamaged", "social post",
                     "both keys", "recovered burned")


def tier_for(score: int) -> str:
    return "high" if score >= 45 else "medium" if score >= 20 else "low"


def mask_policy(p: str) -> str:
    return f"POL-****{p[-4:]}"


def mask_claimant(c: str) -> str:
    return f"CLT-****{c[-2:]}"


@lru_cache(maxsize=1)
def _load() -> tuple[dict, list, list]:
    flagged = {c["claim_id"]: c for c in json.loads((CLAIMS_DIR / "flagged_claims.json").read_text("utf-8"))}
    hist = json.loads((CLAIMS_DIR / "historical_claims.json").read_text("utf-8"))
    ext = json.loads((CLAIMS_DIR / "external_index.json").read_text("utf-8"))
    return flagged, hist, ext


class ClaimNotFound(ValueError):
    pass


def get_claim(claim_id: str) -> dict:
    if not isinstance(claim_id, str) or not CLAIM_ID_RE.match(claim_id):
        raise ValueError(f"invalid claim_id format: expected CLM-NNNN, got {claim_id!r}")
    flagged, _, _ = _load()
    if claim_id not in flagged:
        raise ClaimNotFound(f"claim {claim_id} not found in SIU queue")
    return flagged[claim_id]


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def claim_history(claim_id: str) -> dict:
    c = get_claim(claim_id)
    _, hist, _ = _load()
    prior = [h for h in hist if h["claimant_id"] == c["claimant_id"]]
    return {
        "claim_id": c["claim_id"],
        "policy_ref_masked": mask_policy(c["policy_number"]),
        "claimant_ref_masked": mask_claimant(c["claimant_id"]),
        "line_of_business": c["line_of_business"],
        "loss_type": c["loss_type"],
        "loss_date": c["loss_date"],
        "report_date": c["report_date"],
        "policy_inception_date": c["policy_inception_date"],
        "policy_tenure_days": _days(c["policy_inception_date"], c["loss_date"]),
        "report_lag_days": _days(c["loss_date"], c["report_date"]),
        "claimed_amount": c["claimed_amount"],
        "independent_estimate": c.get("estimate_amount"),
        "passengers": c.get("passengers", 0),
        "soft_tissue_injuries": c.get("soft_tissue_injuries", 0),
        "attorney_involved": bool(c.get("attorney")),
        "attorney_retained_days_after_loss": c.get("attorney_retained_days_after_loss"),
        "police_report": c.get("police_report"),
        "witnesses": c.get("witnesses"),
        "provider": c.get("provider"),
        "repair_shop": c.get("repair_shop"),
        "contractor": c.get("contractor"),
        "coverage_change": {"date": c.get("coverage_change_date"), "pct": c.get("coverage_change_pct", 0)},
        "late_premium_payments_12m": c.get("late_premium_payments_12m", 0),
        "adjuster_evidence_notes": c.get("evidence_notes", []),
        "intake_flags": c.get("intake_flags", []),
        "prior_claims": [{
            "claim_ref": h["claim_id"], "loss_type": h["loss_type"], "loss_date": h["loss_date"],
            "paid_amount": h["paid_amount"], "outcome": h["outcome"], "siu_flagged": h["siu_flagged"],
        } for h in sorted(prior, key=lambda h: h["loss_date"])],
        # untrusted: returned verbatim ONLY inside an explicit quarantine envelope; the caller must quarantine it
        "claimant_statement_untrusted": c.get("claimant_statement", ""),
    }


def linked_claims(claim_id: str) -> dict:
    c = get_claim(claim_id)
    _, hist, _ = _load()
    links = []
    for field in ("provider", "repair_shop", "attorney", "contractor"):
        v = c.get(field)
        if not v:
            continue
        hits = [h for h in hist if h.get(field) == v]
        if hits:
            links.append({
                "entity_type": field, "entity": v, "linked_claims": len(hits),
                "siu_flagged_linked_claims": sum(1 for h in hits if h["siu_flagged"]),
                "sample_refs": [h["claim_id"] for h in hits[:5]],
            })
    phone_hits = [h for h in hist if h.get("claimant_phone") == c["claimant_phone"] and h["claimant_id"] != c["claimant_id"]]
    if phone_hits:
        links.append({"entity_type": "shared_contact_phone", "entity": "[PHONE-REDACTED]",
                      "linked_claims": len(phone_hits),
                      "siu_flagged_linked_claims": sum(1 for h in phone_hits if h["siu_flagged"]),
                      "sample_refs": [h["claim_id"] for h in phone_hits[:5]]})
    return {"claim_id": claim_id, "links": links, "ring_suspected": any(l["siu_flagged_linked_claims"] >= 3 for l in links)}


def red_flags(claim_id: str) -> dict:
    c = get_claim(claim_id)
    _, hist, ext = _load()
    h = claim_history(claim_id)
    fired: list[dict] = []

    def fire(code: str, evidence: str, weight: int | None = None):
        meta = INDICATORS[code]
        fired.append({"code": code, "name": meta["name"], "weight": weight or meta["weight"], "evidence": evidence})

    if h["policy_tenure_days"] <= 60:
        fire("RF-01", f"loss {h['policy_tenure_days']} days after policy inception")
    if h["report_lag_days"] > 14:
        fire("RF-02", f"reported {h['report_lag_days']} days after loss")
    if c.get("attorney_retained_days_after_loss") is not None and c["attorney_retained_days_after_loss"] <= 3:
        fire("RF-03", f"attorney retained {c['attorney_retained_days_after_loss']} day(s) after loss")
    if c.get("soft_tissue_injuries", 0) >= 3:
        fire("RF-04", f"{c['soft_tissue_injuries']} soft-tissue injury claimants from one impact")
    if c["line_of_business"] == "auto" and not c.get("police_report") and ("collision" in c["loss_type"] or "theft" in c["loss_type"]):
        fire("RF-05", "no police report filed")
    est = c.get("estimate_amount")
    if est and c["claimed_amount"] >= 1.8 * est:
        fire("RF-06", f"claimed {c['claimed_amount']:,} vs independent estimate {est:,} ({c['claimed_amount'] / est:.1f}x)")
    ring = [l for l in linked_claims(claim_id)["links"] if l["siu_flagged_linked_claims"] >= 3 and l["entity_type"] != "shared_contact_phone"]
    if ring:
        l = max(ring, key=lambda x: x["siu_flagged_linked_claims"])
        fire("RF-07", f"{l['entity_type']} '{l['entity']}' linked to {l['siu_flagged_linked_claims']} SIU-flagged claims")
    recent_prior = [p for p in h["prior_claims"] if 0 < _days(p["loss_date"], c["loss_date"]) <= 3 * 365]
    if len(recent_prior) >= 2:
        fire("RF-08", f"{len(recent_prior)} prior claims in the last 36 months")
    cc = c.get("coverage_change_date")
    if cc and c.get("coverage_change_pct", 0) >= 20 and 0 <= _days(cc, c["loss_date"]) <= 90:
        fire("RF-09", f"coverage increased {c['coverage_change_pct']}% {_days(cc, c['loss_date'])} days before loss")
    if c.get("late_premium_payments_12m", 0) >= 3 or c.get("loan_balance_exceeds_value"):
        bits = []
        if c.get("late_premium_payments_12m", 0) >= 3:
            bits.append(f"{c['late_premium_payments_12m']} late premium payments in 12 months")
        if c.get("loan_balance_exceeds_value"):
            bits.append("loan balance exceeds vehicle value")
        fire("RF-10", "; ".join(bits))
    anomalies = []
    if c.get("billing_on_closed_dates", 0) > 0:
        anomalies.append(f"{c['billing_on_closed_dates']} services billed on clinic-closed dates")
    if c.get("upcoded_line_items", 0) > 0:
        anomalies.append(f"{c['upcoded_line_items']} upcoded line items")
    if anomalies:
        fire("RF-11", "; ".join(anomalies), weight=16 * len(anomalies))
    if c.get("injury_unwitnessed"):
        fire("RF-12", "workplace injury unwitnessed")
    dup = [e for e in ext if e["claimant_id"] == c["claimant_id"] and e["loss_date"] == c["loss_date"]]
    if dup:
        fire("RF-13", f"cross-carrier index match {dup[0]['index_ref']} ({dup[0]['carrier']}, status {dup[0]['status']})")
    notes = [n for n in c.get("evidence_notes", []) if any(k in n.lower() for k in EVIDENCE_KEYWORDS)]
    if notes:
        fire("RF-14", " | ".join(notes[:2]), weight=14 * min(2, len(notes)))
    if c["line_of_business"] in ("auto", "property") and c.get("witnesses", 0) == 0:
        fire("RF-15", "no independent witnesses")
    if INJECTION_RE.search(c.get("claimant_statement", "")):
        fire("RF-16", "claimant statement contains instructions addressed to an AI system (quarantined, not obeyed)")
    if any(l["entity_type"] == "shared_contact_phone" for l in linked_claims(claim_id)["links"]):
        fire("RF-17", "claimant phone number shared with another claimant in the book")
    if c["claimed_amount"] >= 100_000:
        fire("RF-18", f"claimed amount {c['claimed_amount']:,}")

    score = min(100, sum(f["weight"] for f in fired))
    return {"claim_id": claim_id, "score": score, "tier": tier_for(score), "indicators": fired,
            "tier_thresholds": {k: list(v) for k, v in TIERS.items()}}


def scheme_index() -> list[dict]:
    out = []
    for p in sorted(CORPUS_DIR.glob("*.md")):
        head = p.read_text("utf-8").split("---")[1]
        meta = dict(line.split(":", 1) for line in head.strip().splitlines() if ":" in line)
        out.append({"doc_id": meta["doc_id"].strip(), "title": meta["title"].strip(),
                    "line_of_business": meta["line_of_business"].strip(), "file": p.name})
    return out
