"""Measured before/after optimisation (good-to-have) -> reports/optimization/*.json + docs/optimization-note data

Runs the same benchmark investigations twice inside the traced process:
    baseline   GEMINI_THINKING_LEVEL=high, CONTEXT_COMPRESSION=false, MAX_RETRIEVAL_ROUNDS=3
    optimized  GEMINI_THINKING_LEVEL=low,  CONTEXT_COMPRESSION=true,  MAX_RETRIEVAL_ROUNDS=2
Each phase's runs are tagged (bench-baseline / bench-optimized); after the Phoenix export the spans are split by
tag and a Phoenix-derived golden-signals report is produced for each, plus a comparison.
"""
from __future__ import annotations

import asyncio
import json
import os
import time

import pandas as pd

from src.config import REPORTS_DIR, TRACES_DIR

BENCH_CLAIMS = ["CLM-1001", "CLM-1005", "CLM-1008"]
PHASES = {
    "bench-baseline": {"GEMINI_THINKING_LEVEL": "high", "CONTEXT_COMPRESSION": "false", "MAX_RETRIEVAL_ROUNDS": "3"},
    "bench-optimized": {"GEMINI_THINKING_LEVEL": "low", "CONTEXT_COMPRESSION": "true", "MAX_RETRIEVAL_ROUNDS": "2"},
}
OUT_DIR = REPORTS_DIR / "optimization"


async def run_phase(tag: str, env: dict, concurrency: int = 3) -> list[dict]:
    from src.runner import Copilot

    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        sem = asyncio.Semaphore(concurrency)
        async with Copilot() as cp:
            async def one(c):
                async with sem:
                    r = await cp.ask(f"Investigate flagged claim {c} and draft the investigation case file.",
                                     thread_id=f"{tag}-{c}-{int(time.time())}", actor_id="INV-07", tags=[tag])
                    return {"claim": c, "run_id": r["run_id"], "action": (r["case_file"] or {}).get("recommended_action"),
                            "degraded": r["degraded"], "cost_usd": r["cost"]["cost_usd"]}
            return await asyncio.gather(*(one(c) for c in BENCH_CLAIMS))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def split_and_report() -> dict:
    """After export: per-phase span files + Phoenix-derived golden signals + comparison."""
    from scripts.golden_signals import build
    from scripts.span_analytics import _first_tag

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(TRACES_DIR / "phoenix_spans.parquet")
    tags = df["attributes.tag.tags"].map(_first_tag) if "attributes.tag.tags" in df.columns else pd.Series([None] * len(df))
    trace_tag = pd.Series(tags.values, index=df["context.trace_id"]).dropna()
    trace_tag = trace_tag[~trace_tag.index.duplicated()]
    reports = {}
    for tag in PHASES:
        traces = set(trace_tag[trace_tag == tag].index)
        sub = df[df["context.trace_id"].isin(traces)]
        if sub.empty:
            continue
        p = TRACES_DIR / f"{tag.replace('-', '_')}_spans.parquet"
        sub.to_parquet(p, index=False)
        rep = build(p, eval_path=REPORTS_DIR / "__none__.json")
        rep["phase_config"] = PHASES[tag]
        (OUT_DIR / f"golden_signals_{tag.split('-')[1]}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        reports[tag] = rep
    if len(reports) == 2:
        b, o = reports["bench-baseline"], reports["bench-optimized"]

        def delta(get):
            bv, ov = get(b), get(o)
            return {"baseline": bv, "optimized": ov,
                    "change_pct": round((ov - bv) / bv * 100, 1) if bv else None}
        cmp = {
            "generated_by": "python -m scripts.optimization_benchmark (via scripts/regenerate_evidence.py)",
            "benchmark_claims": BENCH_CLAIMS, "phases": PHASES,
            "run_latency_p50_ms": delta(lambda r: r["latency"]["end_to_end_run"]["p50_ms"]),
            "run_latency_p95_ms": delta(lambda r: r["latency"]["end_to_end_run"]["p95_ms"]),
            "thinking_p50_ms": delta(lambda r: r["latency"]["thinking_llm"]["p50_ms"]),
            "prompt_tokens": delta(lambda r: r["tokens"]["prompt_tokens"]),
            "completion_tokens": delta(lambda r: r["tokens"]["completion_tokens"]),
            "reasoning_tokens": delta(lambda r: r["tokens"]["reasoning_tokens"]),
            "cost_total_usd": delta(lambda r: r["cost"]["total_usd"]),
            "llm_calls": delta(lambda r: r["traffic"]["llm_calls"]),
        }
        (OUT_DIR / "comparison.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
        reports["comparison"] = cmp
    return reports
