"""PII detection and masking (AC-06, NFR-05).

Two layers:
1. Deterministic regex recognisers for the insurer's own identifier formats (policy numbers, claimant ids,
   phone numbers, e-mail addresses). These ALWAYS run and are the control relied on by every log writer.
2. Microsoft Presidio (``presidio-analyzer``) with custom pattern recognisers for the same identifiers plus
   its built-in PERSON / EMAIL / PHONE entities, used on free text when available.

``mask_text`` / ``mask_obj`` are applied by the tool-log middleware, the audit trail, the MCP transcript,
the output guardrail and the MCP server itself, so raw identifiers never reach answers, logs or traces.
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import Any

from src.config import get_settings

log = logging.getLogger(__name__)

POLICY_RE = re.compile(r"\bPOL-(\d{4,})(\d{4})\b")
CLAIMANT_RE = re.compile(r"\bCLT-(\d{2,})(\d{2})\b")
PHONE_RE = re.compile(r"(?<!\w)\+?\d{1,3}[-\s.]\d{3}[-\s.]\d{3}[-\s.]\d{4}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
NAME_RE = re.compile(r"\bSynthetic Claimant [A-Z]\d{1,2}\b")

# Already-masked tokens (so the leak scanner does not flag them)
MASKED_POLICY_RE = re.compile(r"\bPOL-\*{4}\d{4}\b")
MASKED_CLAIMANT_RE = re.compile(r"\bCLT-\*{4}\d{2}\b")


def mask_policy(p: str) -> str:
    m = POLICY_RE.fullmatch(p)
    return f"POL-****{m.group(2)}" if m else "POL-****"


def mask_claimant(c: str) -> str:
    m = CLAIMANT_RE.fullmatch(c)
    return f"CLT-****{m.group(2)}" if m else "CLT-****"


def pseudonym(value: str, prefix: str = "ID") -> str:
    """Stable salted pseudonym, used where correlation across records is needed without the raw value."""
    salt = get_settings().pii_hash_salt
    return f"{prefix}#{hashlib.sha256((salt + value).encode()).hexdigest()[:10]}"


def regex_mask(text: str) -> str:
    text = POLICY_RE.sub(lambda m: f"POL-****{m.group(2)}", text)
    text = CLAIMANT_RE.sub(lambda m: f"CLT-****{m.group(2)}", text)
    text = EMAIL_RE.sub("[EMAIL-REDACTED]", text)
    text = PHONE_RE.sub("[PHONE-REDACTED]", text)
    text = NAME_RE.sub("[NAME-REDACTED]", text)
    return text


@lru_cache(maxsize=1)
def _presidio():
    """Build a Presidio analyzer with the small spaCy model + our identifier recognisers (or None)."""
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        })
        engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
        engine.registry.add_recognizer(PatternRecognizer(
            supported_entity="POLICY_NUMBER", patterns=[Pattern("policy", r"\bPOL-\d{8}\b", 0.95)]))
        engine.registry.add_recognizer(PatternRecognizer(
            supported_entity="CLAIMANT_ID", patterns=[Pattern("claimant", r"\bCLT-\d{6}\b", 0.95)]))
        return engine
    except Exception as exc:  # pragma: no cover - optional dependency path
        log.warning("Presidio unavailable, regex PII masking only: %s", exc)
        return None


PRESIDIO_ENTITIES = ["POLICY_NUMBER", "CLAIMANT_ID", "PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON"]


def detect_pii(text: str) -> list[dict]:
    """Return detected PII spans (entity type + offsets; never the raw value)."""
    found: list[dict] = []
    for name, rx in (("POLICY_NUMBER", POLICY_RE), ("CLAIMANT_ID", CLAIMANT_RE),
                     ("PHONE_NUMBER", PHONE_RE), ("EMAIL_ADDRESS", EMAIL_RE)):
        for m in rx.finditer(text):
            found.append({"entity": name, "start": m.start(), "end": m.end(), "source": "regex"})
    engine = _presidio()
    if engine is not None:
        try:
            for r in engine.analyze(text=text, language="en", entities=PRESIDIO_ENTITIES, score_threshold=0.6):
                if not any(f["start"] <= r.start < f["end"] for f in found):
                    found.append({"entity": r.entity_type, "start": r.start, "end": r.end, "source": "presidio"})
        except Exception as exc:  # pragma: no cover
            log.warning("presidio analyze failed: %s", exc)
    return sorted(found, key=lambda f: f["start"])


def redact_free_text(text: str) -> tuple[str, list[str]]:
    """Presidio + regex redaction for untrusted free text. Returns (redacted_text, entity_types_found)."""
    spans = detect_pii(text)
    out, last, kinds = [], 0, []
    for s in spans:
        if s["start"] < last:
            continue
        raw = text[s["start"]:s["end"]]
        if s["entity"] == "POLICY_NUMBER" and POLICY_RE.fullmatch(raw):
            rep = mask_policy(raw)
        elif s["entity"] == "CLAIMANT_ID" and CLAIMANT_RE.fullmatch(raw):
            rep = mask_claimant(raw)
        elif s["entity"] == "PERSON" and raw.lower() in {"claimant", "police", "i", "we"}:
            rep = raw
        else:
            rep = f"[{s['entity']}-REDACTED]"
        out.append(text[last:s["start"]])
        out.append(rep)
        kinds.append(s["entity"])
        last = s["end"]
    out.append(text[last:])
    return regex_mask("".join(out)), sorted(set(kinds))


def mask_text(text: str) -> str:
    """Fast deterministic masking used on every log line."""
    return regex_mask(text) if isinstance(text, str) else text


def mask_obj(obj: Any) -> Any:
    """Recursively mask any str inside dict/list structures."""
    if isinstance(obj, str):
        return mask_text(obj)
    if isinstance(obj, dict):
        return {k: mask_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [mask_obj(v) for v in obj]
    return obj


def contains_raw_pii(text: str) -> list[str]:
    """Leak scanner used by the output guardrail and by tests over committed logs/traces."""
    hits = []
    if POLICY_RE.search(text):
        hits.append("POLICY_NUMBER")
    if CLAIMANT_RE.search(text):
        hits.append("CLAIMANT_ID")
    if PHONE_RE.search(text):
        hits.append("PHONE_NUMBER")
    if EMAIL_RE.search(text):
        hits.append("EMAIL_ADDRESS")
    if NAME_RE.search(text):
        hits.append("CLAIMANT_NAME")
    return hits
