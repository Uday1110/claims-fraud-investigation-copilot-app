"""ONE command to regenerate every evidence artifact (NFR-02, NFR-06, Reproducibility Rule).

    python -m scripts.regenerate_evidence            # full run (Gemini + Phoenix + DeepEval judge)
    python -m scripts.regenerate_evidence --offline  # deterministic fallbacks, no Gemini (mechanics check only)

Pipeline (all inside ONE traced process so every run lands in the same Phoenix project):
  1. reset evidence logs + runtime stores (fresh checkpoints / long-term memory)
  2. start Arize Phoenix in-process and instrument LangChain/LangGraph      (src/observability/tracing.py)
  3. demo session from committed sample inputs                              -> reports/demo_run.md|json
  4. golden-set evaluation (copilot runs + DeepEval LLM-as-judge)           -> reports/eval_report.json
  5. red-team attack set                                                    -> reports/redteam_results.json
  6. optimisation benchmark, baseline vs optimized (optional --benchmark)   -> reports/optimization/
  7. export Phoenix spans                                                   -> traces/phoenix_spans.parquet|jsonl
  8. Phoenix-derived golden signals                                         -> reports/golden_signals.json
  9. dashboard data + chart + live Phoenix UI screenshot                    -> reports/dashboard_data.csv|.png, phoenix_ui.png
 10. failure candidates, PII redaction sample, evidence manifest            -> reports/*.json
Logs written along the way by committed middleware: logs/tool_calls.jsonl, logs/agent_actions.jsonl,
logs/mcp_transcript.jsonl, logs/run_costs.jsonl.  (logs/memory_test.log is produced by `python -m pytest`.)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

from src.config import LOGS_DIR, REPORTS_DIR, ROOT, RUNTIME_DIR, TRACES_DIR, get_settings

EVIDENCE_LOGS = ["tool_calls.jsonl", "agent_actions.jsonl", "mcp_transcript.jsonl", "run_costs.jsonl"]


def step(msg: str) -> None:
    print(f"\n=== [{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def reset() -> None:
    for name in EVIDENCE_LOGS:
        (LOGS_DIR / name).unlink(missing_ok=True)
    s = get_settings()
    for p in (s.checkpoint_db, s.memory_db):
        for suffix in ("", "-wal", "-shm", "-journal"):
            (p.parent / (p.name + suffix)).unlink(missing_ok=True)
    for p in TRACES_DIR.glob("*"):
        if p.is_file() and p.name != ".gitkeep":
            p.unlink()
    if (REPORTS_DIR / "optimization").exists():
        shutil.rmtree(REPORTS_DIR / "optimization")


def manifest() -> None:
    arts = sorted([*LOGS_DIR.glob("*.jsonl"), *LOGS_DIR.glob("*.log"), *TRACES_DIR.glob("*.*"), *REPORTS_DIR.rglob("*.*")])
    producers = {
        "tool_calls.jsonl": "src/tools/tool_logging.py (ToolBox middleware)",
        "agent_actions.jsonl": "src/audit/audit.py (audit middleware)",
        "mcp_transcript.jsonl": "src/tools/mcp_client.py + src/tools/tool_logging.py",
        "run_costs.jsonl": "src/observability/cost.py",
        "memory_test.log": "tests/test_memory_persistence.py",
        "phoenix_spans.parquet": "src/observability/tracing.py::export_spans",
        "phoenix_spans.jsonl": "src/observability/tracing.py::export_spans",
        "golden_signals.json": "scripts/golden_signals.py",
        "dashboard_data.csv": "scripts/build_dashboard.py::export_csv",
        "dashboard.png": "scripts/build_dashboard.py::draw",
        "phoenix_ui.png": "scripts/build_dashboard.py::phoenix_ui_screenshot",
        "eval_report.json": "evals/run_eval.py",
        "redteam_results.json": "scripts/run_redteam.py",
        "failure_candidates.json": "scripts/find_failures.py",
        "pii_redaction_sample.json": "scripts/pii_redaction_sample.py",
        "demo_run.md": "src/cli.py::run_demo", "demo_run.json": "src/cli.py::run_demo",
    }
    rows = []
    for a in arts:
        if a.name == "evidence_manifest.json":
            continue
        rel = a.relative_to(ROOT).as_posix()
        prod = producers.get(a.name) or ("scripts/optimization_benchmark.py" if "optimization" in rel or "bench_" in a.name else None)
        rows.append({"artifact": rel, "bytes": a.stat().st_size, "sha256": hashlib.sha256(a.read_bytes()).hexdigest()[:16],
                     "produced_by": prod})
    (REPORTS_DIR / "evidence_manifest.json").write_text(json.dumps({
        "generated_by": "python -m scripts.regenerate_evidence", "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": get_settings().gemini_model, "artifacts": rows}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--benchmark", action="store_true", help="also run the baseline-vs-optimized benchmark")
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("EVAL_CONCURRENCY", "3")))
    ap.add_argument("--skip-demo", action="store_true")
    ap.add_argument("--skip-redteam", action="store_true")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if a.offline:
        os.environ["GOOGLE_API_KEY"] = ""
        os.environ["GEMINI_API_KEY"] = ""
    t0 = time.perf_counter()
    step("1/10 reset evidence logs and runtime stores")
    reset()

    step("2/10 start Phoenix (in-process) + instrument LangChain/LangGraph")
    from src.observability.tracing import export_spans, init_tracing

    base_url = init_tracing()
    print("Phoenix UI:", base_url)

    if not a.skip_demo:
        step("3/10 demo session (data/sample_inputs/demo_session.json)")
        from src.cli import run_demo

        res = asyncio.run(run_demo(offline=a.offline, transcript=REPORTS_DIR / "demo_run.md"))
        (REPORTS_DIR / "demo_run.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    step("4/10 golden-set evaluation (DeepEval, Gemini judge)")
    from evals.run_eval import evaluate

    rep = asyncio.run(evaluate(a.concurrency, judge=not (a.no_judge or a.offline)))
    print(json.dumps(rep["summary"], indent=2))

    if not a.skip_redteam:
        step("5/10 red-team attack set")
        from scripts.run_redteam import run as run_redteam

        rt = asyncio.run(run_redteam(a.concurrency))
        print(f"red team defended {rt['defended']}/{rt['attacks']}")

    if a.benchmark:
        step("6/10 optimisation benchmark (baseline vs optimized)")
        from scripts.optimization_benchmark import PHASES, run_phase

        for tag, env in PHASES.items():
            print(tag, asyncio.run(run_phase(tag, env)))

    step("7/10 export Phoenix spans -> traces/")
    time.sleep(3)
    print(export_spans())

    step("8/10 golden signals (Phoenix-derived)")
    from scripts.golden_signals import build

    gs = build()
    (REPORTS_DIR / "golden_signals.json").write_text(json.dumps(gs, indent=2), encoding="utf-8")
    print(json.dumps(gs["slo"], indent=2))
    if a.benchmark:
        from scripts.optimization_benchmark import split_and_report

        print(json.dumps(split_and_report().get("comparison", {}), indent=2))

    step("9/10 dashboard data + chart + Phoenix UI screenshot")
    from scripts.build_dashboard import draw, export_csv, phoenix_ui_screenshot

    export_csv()
    draw()
    try:
        print("screenshot:", phoenix_ui_screenshot(base_url, get_settings().phoenix_project))
    except Exception as exc:  # noqa: BLE001 - screenshot is best-effort; data + chart are the scored artifacts
        print("phoenix screenshot skipped:", exc)

    step("10/10 failure candidates, PII sample, manifest")
    from scripts import find_failures, pii_redaction_sample

    find_failures.main()
    pii_redaction_sample.main()
    manifest()
    print(f"\nDONE in {time.perf_counter() - t0:.0f}s. Runtime state in {RUNTIME_DIR} (git-ignored).")


if __name__ == "__main__":
    main()
