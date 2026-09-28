"""Phoenix-derived golden-signals report -> reports/golden_signals.json   (AC-09)

Reads the Phoenix span export (``traces/phoenix_spans.parquet``, produced by ``get_spans_dataframe()``) and computes:
* latency  - end-to-end per run, and p50/p95 by span category: thinking (LLM), acting (agent nodes), tool;
             per agent node and per tool
* traffic  - runs, spans, LLM calls, tool calls, runs per tag
* errors   - error-span rate, LLM / tool error spans, degraded runs (from the eval report)
* saturation / budget - tokens in/out (+ reasoning), cost = tokens x price, cost per run vs budget
* quality  - accuracy + hallucination rate imported from reports/eval_report.json
* SLO evaluation for each signal

Usage:  python -m scripts.golden_signals [--spans traces/phoenix_spans.parquet] [--out reports/golden_signals.json]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.span_analytics import (SPANS, TOK_IN, TOK_OUT, TOK_REASON, drop_bench, latency_block, load_spans, pct,
                                    per_run)
from src.config import REPORTS_DIR, get_settings

SLO = {"run_p95_ms": 240_000, "tool_p95_ms": 5_000, "error_span_rate": 0.05, "cost_per_run_p95_usd": None,
       "accuracy_min": 0.90, "hallucination_rate_max": 0.10}


def build(spans_path: Path = SPANS, eval_path: Path = REPORTS_DIR / "eval_report.json") -> dict:
    s = get_settings()
    df = load_spans(spans_path)
    if spans_path == SPANS:
        df = drop_bench(df)
    runs = per_run(df)
    think, act, tool = (df[df["category"] == c] for c in ("thinking", "acting", "tool"))
    ev = json.loads(eval_path.read_text("utf-8"))["summary"] if eval_path.exists() else {}
    total_in, total_out = int(df[TOK_IN].sum()), int(df[TOK_OUT].sum())
    total_cost = round(float(df["cost_usd"].sum()), 6)
    budget = s.cost_budget_per_run_usd
    report = {
        "generated_by": "python -m scripts.golden_signals",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"phoenix_spans": str(spans_path.name), "spans": int(len(df)),
                   "trace_window": [str(df["start_time"].min()), str(df["end_time"].max())],
                   "eval_report": "reports/eval_report.json" if ev else None},
        "model": s.gemini_model,
        "pricing_usd_per_1m_tokens": {"input": s.price_in_per_1m, "output": s.price_out_per_1m,
                                      "note": "configurable via PRICE_INPUT_PER_1M / PRICE_OUTPUT_PER_1M"},
        "latency": {
            "end_to_end_run": latency_block(runs["run_latency_ms"]),
            "thinking_llm": latency_block(think["latency_ms"]),
            "acting_agent_nodes": latency_block(act["latency_ms"]),
            "tool": latency_block(tool["latency_ms"]),
            "per_run_thinking_ms": latency_block(runs["thinking_ms"]),
            "per_run_tool_ms": latency_block(runs["tool_ms"]),
            "by_agent_node": {n: latency_block(g["latency_ms"]) for n, g in act.groupby("name")},
            "by_tool": {n: latency_block(g["latency_ms"]) for n, g in tool.groupby("name")},
        },
        "traffic": {
            "runs": int(len(runs)), "spans": int(len(df)), "llm_calls": int(len(think)), "tool_calls": int(len(tool)),
            "agent_node_executions": int(len(act)),
            "runs_by_tag": {str(k): int(v) for k, v in runs["run_tag"].fillna("untagged").value_counts().items()},
            "llm_calls_per_run": {"p50": pct(runs["llm_calls"], 50), "p95": pct(runs["llm_calls"], 95)},
        },
        "errors": {
            "error_spans": int(df["is_error"].sum()),
            "error_span_rate": round(float(df["is_error"].mean()), 4),
            "llm_error_spans": int(think["is_error"].sum()), "tool_error_spans": int(tool["is_error"].sum()),
            "runs_with_error_spans": int((runs["error_spans"] > 0).sum()),
            "degraded_eval_runs": ev.get("degraded_runs"),
        },
        "tokens": {
            "prompt_tokens": total_in, "completion_tokens": total_out, "reasoning_tokens": int(df[TOK_REASON].sum()),
            "total_tokens": total_in + total_out,
            "per_run_total": {"p50": pct(runs["total_tokens"], 50), "p95": pct(runs["total_tokens"], 95)},
            "by_agent": {str(k): {"llm_calls": int(g.shape[0]), "prompt_tokens": int(g[TOK_IN].sum()),
                                  "completion_tokens": int(g[TOK_OUT].sum()), "reasoning_tokens": int(g[TOK_REASON].sum()),
                                  "cost_usd": round(float(g["cost_usd"].sum()), 6),
                                  "p50_ms": pct(g["latency_ms"], 50), "p95_ms": pct(g["latency_ms"], 95)}
                         for k, g in think.groupby("agent")},
        },
        "cost": {
            "total_usd": total_cost,
            "per_run_usd": {"mean": round(float(runs["cost_usd"].mean()), 6) if len(runs) else None,
                            "p50": round(float(runs["cost_usd"].median()), 6) if len(runs) else None,
                            "p95": round(float(runs["cost_usd"].quantile(0.95)), 6) if len(runs) else None,
                            "max": round(float(runs["cost_usd"].max()), 6) if len(runs) else None},
            "budget_per_run_usd": budget,
            "runs_over_budget": int((runs["cost_usd"] > budget).sum()),
        },
        "quality": {
            "accuracy": ev.get("accuracy"), "hallucination_rate": ev.get("hallucination_rate"),
            "hallucination_score_mean": ev.get("hallucination_score_mean"), "faithfulness_mean": ev.get("faithfulness_mean"),
            "answer_relevancy_mean": ev.get("answer_relevancy_mean"), "case_correctness_mean": ev.get("case_correctness_mean"),
            "investigation_action_accuracy": ev.get("investigation_action_accuracy"),
            "safety_pass_rate": ev.get("safety_pass_rate"),
        },
    }
    lat, err, q = report["latency"], report["errors"], report["quality"]
    report["slo"] = {
        "run_p95_latency": {"target_ms": SLO["run_p95_ms"], "actual_ms": lat["end_to_end_run"]["p95_ms"],
                            "met": (lat["end_to_end_run"]["p95_ms"] or 0) <= SLO["run_p95_ms"]},
        "tool_p95_latency": {"target_ms": SLO["tool_p95_ms"], "actual_ms": lat["tool"]["p95_ms"],
                             "met": (lat["tool"]["p95_ms"] or 0) <= SLO["tool_p95_ms"]},
        "error_span_rate": {"target_max": SLO["error_span_rate"], "actual": err["error_span_rate"],
                            "met": err["error_span_rate"] <= SLO["error_span_rate"]},
        "cost_per_run_p95": {"target_max_usd": budget, "actual_usd": report["cost"]["per_run_usd"]["p95"],
                             "met": (report["cost"]["per_run_usd"]["p95"] or 0) <= budget},
        "accuracy": {"target_min": SLO["accuracy_min"], "actual": q["accuracy"],
                     "met": q["accuracy"] is not None and q["accuracy"] >= SLO["accuracy_min"]},
        "hallucination_rate": {"target_max": SLO["hallucination_rate_max"], "actual": q["hallucination_rate"],
                               "met": q["hallucination_rate"] is not None and q["hallucination_rate"] <= SLO["hallucination_rate_max"]},
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spans", type=Path, default=SPANS)
    ap.add_argument("--eval", type=Path, default=REPORTS_DIR / "eval_report.json")
    ap.add_argument("--out", type=Path, default=REPORTS_DIR / "golden_signals.json")
    a = ap.parse_args()
    rep = build(a.spans, a.eval)
    a.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("slo",)}, indent=2))


if __name__ == "__main__":
    main()
