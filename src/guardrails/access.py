"""Claim-level access control: an actor may only touch claims assigned to them (AC-06)."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from src.config import CLAIMS_DIR

CLAIM_REF_RE = re.compile(r"\bCLM-\d{3,6}\b", re.I)


@lru_cache(maxsize=1)
def _assignments() -> dict:
    return json.loads((CLAIMS_DIR / "case_assignments.json").read_text("utf-8"))


def resolve_actor(role: str, actor_id: str) -> dict:
    """Build the actor record (role, id, authorized_claims) from the synthetic assignment table."""
    a = _assignments()
    if role in ("investigator", "reviewer"):
        rec = a["investigators"].get(actor_id)
        if not rec:
            return {"role": role, "id": actor_id, "authorized_claims": [], "can_sign_off": False}
        return {"role": role, "id": actor_id, "authorized_claims": list(rec["authorized_claims"]),
                "can_sign_off": bool(rec.get("can_sign_off"))}
    if role == "claimant":
        # claimant-portal actors are identified by their own claim reference only
        own = a["claimant_portal"].get(actor_id)
        return {"role": "claimant", "id": actor_id, "authorized_claims": [own["claim_id"]] if own else [],
                "can_sign_off": False}
    return {"role": role, "id": actor_id, "authorized_claims": [], "can_sign_off": False}


def can_sign_off(reviewer_id: str) -> bool:
    return bool(_assignments()["investigators"].get(reviewer_id, {}).get("can_sign_off"))


def referenced_claims(text: str) -> list[str]:
    return sorted({m.upper() for m in CLAIM_REF_RE.findall(text or "")})


def unauthorized_claims(text: str, actor: dict) -> list[str]:
    allowed = set(actor.get("authorized_claims") or [])
    return [c for c in referenced_claims(text) if c not in allowed]
