"""Cost / latency / token dashboard (AC-09) -> reports/dashboard_data.csv + reports/dashboard.png

1. ``dashboard_data.csv`` is the span-level table exported from Phoenix (``get_spans_dataframe()`` columns:
   run_id, trace/span ids, span name, kind, category, latency, tokens, cost, status).
2. ``dashboard.png`` is drawn ONLY from that CSV (re-read from disk), so the picture and its data file cannot drift.
3. ``phoenix_ui_screenshot()`` captures the live Phoenix UI (localhost:6006) with Playwright while the in-process
   server is running -> reports/phoenix_ui.png (called by scripts/regenerate_evidence.py).

Usage:  python -m scripts.build_dashboard
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.span_analytics import SPANS, TOK_IN, TOK_OUT, TOK_REASON, drop_bench, load_spans  # noqa: E402
from src.config import REPORTS_DIR, get_settings  # noqa: E402

CSV = REPORTS_DIR / "dashboard_data.csv"
PNG = REPORTS_DIR / "dashboard.png"

# reference categorical slots (dataviz skill, light mode) + text/surface tokens
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def export_csv(spans: Path = SPANS, out: Path = CSV) -> pd.DataFrame:
    df = drop_bench(load_spans(spans))
    cols = {"run_id": "run_id", "context.trace_id": "trace_id", "context.span_id": "span_id", "parent_id": "parent_id",
            "name": "span_name", "span_kind": "span_kind", "category": "category", "run_tag": "run_tag",
            "start_time": "start_time", "latency_ms": "latency_ms", TOK_IN: "prompt_tokens", TOK_OUT: "completion_tokens",
            TOK_REASON: "reasoning_tokens", "cost_usd": "cost_usd", "status_code": "status_code"}
    table = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    table["latency_ms"] = table["latency_ms"].round(1)
    table["cost_usd"] = table["cost_usd"].round(8)
    table.sort_values("start_time").to_csv(out, index=False)
    return table


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _hbar_pair(ax, labels, a, b, la, lb, unit="s"):
    y = np.arange(len(labels))
    h = 0.38
    ax.barh(y + h / 2 + 0.01, a, height=h, color=S1, label=la)
    ax.barh(y - h / 2 - 0.01, b, height=h, color=S2, label=lb)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for yi, v in zip(y, b):
        ax.text(v, yi - h / 2, f" {v:.1f}{unit}", va="center", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)


def draw(csv: Path = CSV, out: Path = PNG, eval_report: Path = REPORTS_DIR / "eval_report.json") -> None:
    t = pd.read_csv(csv)                      # the dashboard is drawn from the committed data file only
    s = get_settings()
    runs = t[t["category"] == "run"]
    per_trace = t.groupby("trace_id").agg(prompt=("prompt_tokens", "sum"), completion=("completion_tokens", "sum"),
                                         cost=("cost_usd", "sum"))
    per_trace = per_trace.loc[per_trace.index.intersection(runs["trace_id"])]
    ev = json.loads(eval_report.read_text("utf-8"))["summary"] if eval_report.exists() else {}

    fig = plt.figure(figsize=(16, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.22, 1, 1], hspace=0.45, wspace=0.42)

    # --- KPI strip (stat tiles, not charts)
    kpis = [("Runs traced", f"{len(runs)}"),
            ("Run latency p95", f"{np.percentile(runs['latency_ms'], 95) / 1000:.1f}s" if len(runs) else "-"),
            ("Tokens in / out", f"{t['prompt_tokens'].sum() / 1000:.1f}k / {t['completion_tokens'].sum() / 1000:.1f}k"),
            ("Est. cost (placeholder prices)", f"${t['cost_usd'].sum():.4f}"),
            ("Eval accuracy", f"{ev.get('accuracy', 0) * 100:.0f}%" if ev.get("accuracy") is not None else "-"),
            ("Hallucination rate", f"{ev.get('hallucination_rate') * 100:.0f}%" if ev.get("hallucination_rate") is not None else "-")]
    kax = fig.add_subplot(gs[0, :])
    kax.axis("off")
    for i, (label, val) in enumerate(kpis):
        x = i / len(kpis)
        kax.text(x + 0.005, 0.72, label, fontsize=10, color=INK2, transform=kax.transAxes)
        kax.text(x + 0.005, 0.12, val, fontsize=15, color=INK, fontweight="bold", transform=kax.transAxes)

    # --- latency by span category (p50 vs p95)
    ax = fig.add_subplot(gs[1, 0])
    cats = [c for c in ("run", "thinking", "acting", "tool") if (t["category"] == c).any()]
    names = {"run": "end-to-end run", "thinking": "thinking (LLM)", "acting": "acting (agent node)", "tool": "tool call"}
    p50 = [t.loc[t["category"] == c, "latency_ms"].quantile(0.5) / 1000 for c in cats]
    p95 = [t.loc[t["category"] == c, "latency_ms"].quantile(0.95) / 1000 for c in cats]
    _hbar_pair(ax, [names[c] for c in cats], p50, p95, "p50", "p95")
    _style(ax, "Latency by span type (seconds)")

    # --- p95 per agent node
    ax = fig.add_subplot(gs[1, 1])
    act = t[t["category"] == "acting"].groupby("span_name")["latency_ms"].quantile(0.95).sort_values(ascending=False).head(9) / 1000
    ax.barh(act.index, act.values, color=S1, height=0.6)
    ax.invert_yaxis()
    for i, v in enumerate(act.values):
        ax.text(v, i, f" {v:.1f}s", va="center", fontsize=8, color=INK2)
    _style(ax, "Agent node latency p95 (seconds)")

    # --- tool latency
    ax = fig.add_subplot(gs[1, 2])
    tl = t[t["category"] == "tool"].groupby("span_name")["latency_ms"]
    if len(tl.size()):
        labels = list(tl.size().index)
        _hbar_pair(ax, labels, list(tl.quantile(0.5).values), list(tl.quantile(0.95).values), "p50", "p95", unit="ms")
    _style(ax, "Tool call latency (ms)")

    # --- tokens per run (stacked prompt / completion), top 20 runs by tokens
    ax = fig.add_subplot(gs[2, 0:2])
    pt = per_trace.assign(total=per_trace["prompt"] + per_trace["completion"]).sort_values("total", ascending=False).head(24)
    x = np.arange(len(pt))
    ax.bar(x, pt["prompt"], color=S1, width=0.7, label="prompt tokens")
    ax.bar(x, pt["completion"], bottom=pt["prompt"] + pt["total"].max() * 0.004, color=S2, width=0.7, label="completion tokens (incl. reasoning)")
    run_ids = runs.set_index("trace_id")["run_id"].reindex(pt.index).fillna("?").astype(str).str[:6]
    ax.set_xticks(x, run_ids, rotation=60, fontsize=7)
    ax.set_xlabel("run_id (top runs by tokens)", fontsize=8.5, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    _style(ax, "Tokens per run")
    ax.grid(axis="x", visible=False)

    # --- cost per run distribution vs budget
    ax = fig.add_subplot(gs[2, 2])
    costs = per_trace["cost"].sort_values().values * 1000
    ax.hist(costs, bins=min(20, max(5, len(costs) // 2)), color=S3, edgecolor=SURFACE, linewidth=2)
    budget = s.cost_budget_per_run_usd * 1000
    ax.axvline(budget, color=INK2, linestyle="--", linewidth=1.2)
    ax.text(budget, ax.get_ylim()[1] * 0.92, f"  budget ${s.cost_budget_per_run_usd:.2f}", fontsize=8, color=INK2)
    ax.set_xlabel("cost per run (USD x 1e-3)", fontsize=8.5, color=INK2)
    ax.set_ylabel("runs", fontsize=8.5, color=INK2)
    _style(ax, "Cost per run")
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    fig.suptitle(f"Claims Fraud Copilot - Phoenix latency / cost / token dashboard   (model {s.gemini_model}; source {csv.name})",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def phoenix_ui_screenshot(base_url: str, project: str, out: Path = REPORTS_DIR / "phoenix_ui.png") -> str | None:
    """Screenshot the live Phoenix project page (traces + latency/cost/token columns)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    import httpx

    projects = httpx.get(f"{base_url}/v1/projects", timeout=30).json().get("data", [])
    pid = next((p["id"] for p in projects if p.get("name") == project), None)
    url = f"{base_url}/projects/{pid}" if pid else f"{base_url}/projects"
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1800, "height": 1100})
        page.goto(url, wait_until="networkidle", timeout=90_000)
        page.wait_for_timeout(6000)          # let the traces table + latency/token/cost columns render
        page.screenshot(path=str(out), full_page=False)
        b.close()
    return str(out)


def main() -> None:
    export_csv()
    draw()
    print(f"wrote {CSV} and {PNG}")


if __name__ == "__main__":
    main()
