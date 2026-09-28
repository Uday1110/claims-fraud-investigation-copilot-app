"""Runtime cost & token governance.

``CostMeter`` accumulates Gemini token usage per run (from ``AIMessage.usage_metadata``) and enforces a per-run
budget (``COST_BUDGET_PER_RUN_USD``). When a run is over budget, *optional* LLM calls (e.g. extra retrieval
grading rounds, memory extraction) are skipped and replaced with their deterministic fallback -- the budget
guard is audited as ``degraded_mode``. Per-run totals are appended to ``logs/run_costs.jsonl``.

The authoritative cost/latency evidence is still Phoenix-derived (``scripts/golden_signals.py``); this meter is
the in-path control that keeps a single run inside budget.
"""
from __future__ import annotations

import contextvars
from dataclasses import asdict, dataclass, field

from src.config import LOGS_DIR, get_settings
from src.observability.run_context import append_jsonl, current_ids, now_iso

RUN_COST_LOG = LOGS_DIR / "run_costs.jsonl"


@dataclass
class CostMeter:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    llm_calls: int = 0
    skipped_optional_calls: int = 0
    by_agent: dict = field(default_factory=dict)

    def add(self, usage: dict | None, agent: str) -> None:
        usage = usage or {}
        i, o = int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)
        r = int((usage.get("output_token_details") or {}).get("reasoning", 0) or 0)
        self.input_tokens += i
        self.output_tokens += o
        self.reasoning_tokens += r
        self.llm_calls += 1
        a = self.by_agent.setdefault(agent, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        a["calls"] += 1
        a["input_tokens"] += i
        a["output_tokens"] += o

    @property
    def cost_usd(self) -> float:
        s = get_settings()
        return round(self.input_tokens / 1e6 * s.price_in_per_1m + self.output_tokens / 1e6 * s.price_out_per_1m, 6)

    def over_budget(self) -> bool:
        return self.cost_usd >= get_settings().cost_budget_per_run_usd

    def flush(self, extra: dict | None = None) -> dict:
        rec = {"timestamp": now_iso(), **current_ids(), **asdict(self), "cost_usd": self.cost_usd,
               "budget_usd": get_settings().cost_budget_per_run_usd, **(extra or {})}
        append_jsonl(RUN_COST_LOG, rec)
        return rec


meter_var: contextvars.ContextVar[CostMeter] = contextvars.ContextVar("cost_meter")


def current_meter() -> CostMeter:
    try:
        return meter_var.get()
    except LookupError:
        m = CostMeter()
        meter_var.set(m)
        return m
