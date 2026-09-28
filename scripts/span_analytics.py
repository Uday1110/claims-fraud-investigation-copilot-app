"""Shared Phoenix-span analytics used by golden_signals.py and build_dashboard.py.

Input: the committed Phoenix export ``traces/phoenix_spans.parquet`` (``get_spans_dataframe()``).
Span categories used for latency:
    thinking -> LLM spans (Gemini calls, incl. reasoning tokens)
    acting   -> CHAIN spans of agent graph nodes (supervisor, workers, writers, guards)
    tool     -> TOOL spans (MCP tools + RAG tool)
    run      -> root span of each turn (``copilot_turn``) = end-to-end latency
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TRACES_DIR, get_settings
from src.graph import NODE_FACTORIES

SPANS = TRACES_DIR / "phoenix_spans.parquet"
AGENT_NODES = set(NODE_FACTORIES)
TOK_IN = "attributes.llm.token_count.prompt"
TOK_OUT = "attributes.llm.token_count.completion"
TOK_REASON = "attributes.llm.token_count.completion_details.reasoning"


def load_spans(path: Path = SPANS) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in ("start_time", "end_time"):
        df[c] = pd.to_datetime(df[c], utc=True, format="mixed")
    df["latency_ms"] = (df["end_time"] - df["start_time"]).dt.total_seconds() * 1000
    for c in (TOK_IN, TOK_OUT, TOK_REASON):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else 0.0

    def cat(row) -> str:
        kind = str(row.get("span_kind", "")).upper()
        if kind == "LLM":
            return "thinking"
        if kind == "TOOL":
            return "tool"
        if row["name"] == "copilot_turn" and (pd.isna(row.get("parent_id")) or row.get("parent_id") in (None, "None", "")):
            return "run"
        if kind == "CHAIN" and row["name"] in AGENT_NODES:
            return "acting"
        return "other"

    df["category"] = df.apply(cat, axis=1)
    # Attribute each LLM span to the agent that made the call (fix for failure F-6: every LLM span is named
    # "ChatGoogleGenerativeAI"; the calling agent is the nearest ancestor named "<agent>:<Schema>" or an agent node).
    by_id = df.set_index("context.span_id")[["name", "parent_id"]].to_dict("index")

    def agent_of(span_id: str) -> str:
        cur, hops = by_id.get(span_id), 0
        while cur and hops < 12:
            name = str(cur["name"])
            if ":" in name and name.split(":")[0]:
                return name.split(":")[0]
            if name in AGENT_NODES:
                return name
            cur, hops = by_id.get(cur["parent_id"]), hops + 1
        return "unattributed"

    df["agent"] = [agent_of(sid) if c == "thinking" else None for sid, c in zip(df["context.span_id"], df["category"])]
    s = get_settings()
    df["cost_usd"] = df[TOK_IN] / 1e6 * s.price_in_per_1m + df[TOK_OUT] / 1e6 * s.price_out_per_1m
    df["is_error"] = df["status_code"].astype(str).str.upper().eq("ERROR")
    df["run_tag"] = df.get("attributes.tag.tags", pd.Series([None] * len(df))).map(_first_tag)
    return df


def _first_tag(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        tags = json.loads(v) if isinstance(v, str) else list(v)
    except (TypeError, ValueError):
        return None
    for t in tags:
        if t in ("demo", "eval", "redteam", "redteam-offline", "human_review", "bench-baseline", "bench-optimized"):
            return t
    return None


BENCH_TAGS = ("bench-baseline", "bench-optimized")


def drop_bench(df: pd.DataFrame) -> pd.DataFrame:
    """Headline signals exclude the optimisation-benchmark traces (they have their own reports)."""
    bench_traces = set(df.loc[df["run_tag"].isin(BENCH_TAGS), "context.trace_id"])
    return df[~df["context.trace_id"].isin(bench_traces)]


def pct(series: pd.Series, q: float) -> float | None:
    s = series.dropna()
    return round(float(np.percentile(s, q)), 1) if len(s) else None


def latency_block(series: pd.Series) -> dict:
    return {"count": int(series.dropna().shape[0]), "p50_ms": pct(series, 50), "p95_ms": pct(series, 95),
            "max_ms": round(float(series.max()), 1) if len(series.dropna()) else None,
            "mean_ms": round(float(series.mean()), 1) if len(series.dropna()) else None}


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run (turn): latency, tokens, cost, llm/tool call counts, errors."""
    runs = df[df["category"] == "run"][["context.trace_id", "run_id", "latency_ms", "run_tag"]].rename(
        columns={"latency_ms": "run_latency_ms"})
    g = df.groupby("context.trace_id")
    agg = pd.DataFrame({
        "prompt_tokens": g[TOK_IN].sum(), "completion_tokens": g[TOK_OUT].sum(), "reasoning_tokens": g[TOK_REASON].sum(),
        "cost_usd": g["cost_usd"].sum(),
        "llm_calls": g["category"].apply(lambda c: int((c == "thinking").sum())),
        "tool_calls": g["category"].apply(lambda c: int((c == "tool").sum())),
        "error_spans": g["is_error"].sum(),
        "thinking_ms": df[df["category"] == "thinking"].groupby("context.trace_id")["latency_ms"].sum(),
        "tool_ms": df[df["category"] == "tool"].groupby("context.trace_id")["latency_ms"].sum(),
    }).fillna(0)
    out = runs.merge(agg, left_on="context.trace_id", right_index=True, how="left")
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out.sort_values("run_latency_ms", ascending=False)
