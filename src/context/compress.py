"""COMPRESS: shrink tool outputs and conversation history before they enter an LLM context window."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AnyMessage

# per-tool field allow-lists: keep what the downstream agents need, drop the rest
_KEEP = {
    "get_claim_history": ["claim_id", "policy_ref_masked", "claimant_ref_masked", "line_of_business", "loss_type",
                          "loss_date", "report_date", "policy_inception_date", "policy_tenure_days", "report_lag_days",
                          "claimed_amount", "independent_estimate", "passengers", "soft_tissue_injuries",
                          "attorney_involved", "attorney_retained_days_after_loss", "police_report", "witnesses",
                          "provider", "repair_shop", "contractor", "coverage_change", "late_premium_payments_12m",
                          "adjuster_evidence_notes", "intake_flags", "prior_claims"],
    "score_red_flags": ["claim_id", "score", "tier", "indicators"],
    "find_linked_claims": ["claim_id", "links", "ring_suspected"],
}


def compress_tool_output(tool_name: str, data: Any, *, max_list: int = 6, enabled: bool = True) -> Any:
    if not enabled or not isinstance(data, dict):
        return data
    keep = _KEEP.get(tool_name)
    out = {k: data[k] for k in keep if k in data} if keep else dict(data)
    for k, v in list(out.items()):
        if isinstance(v, list) and len(v) > max_list:
            out[k] = v[:max_list] + [{"omitted": len(v) - max_list}]
        if v in (None, [], {}, 0) and k not in ("score", "witnesses", "police_report"):
            out.pop(k)
    return out


def compress_json(obj: Any, max_chars: int) -> str:
    s = json.dumps(obj, default=str, separators=(",", ":"))
    return s if len(s) <= max_chars else s[: max_chars - 20] + '..."[truncated]"'


def message_chars(messages: list[AnyMessage]) -> int:
    return sum(len(str(m.content)) for m in messages)


def trim_to_budget(messages: list[AnyMessage], max_chars: int, keep_last: int = 4) -> list[AnyMessage]:
    """Keep the most recent messages that fit the character budget (always keep the last ``keep_last``)."""
    kept: list[AnyMessage] = []
    total = 0
    for i, m in enumerate(reversed(messages)):
        size = len(str(m.content))
        if i >= keep_last and total + size > max_chars:
            break
        kept.append(m)
        total += size
    return list(reversed(kept))
