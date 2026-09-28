"""Mine REAL failures from the committed evidence -> reports/failure_candidates.json

Sources: Phoenix spans with status ERROR, tool-log records with status != ok, audit events for degraded mode /
loop guard / output repairs / refusals, and failed eval checks. Each candidate carries the run_id + span_id (or the
tool-log / audit line number) that docs/failure-analysis.md cites.

Usage:  python -m scripts.find_failures
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import LOGS_DIR, REPORTS_DIR, TRACES_DIR


def main() -> None:
    out: dict = {"generated_by": "python -m scripts.find_failures"}
    spans = pd.read_parquet(TRACES_DIR / "phoenix_spans.parquet")
    err = spans[spans["status_code"].astype(str).str.upper() == "ERROR"]
    out["error_spans"] = [{
        "run_id": r.get("run_id"), "trace_id": r["context.trace_id"], "span_id": r["context.span_id"], "name": r["name"],
        "span_kind": r.get("span_kind"), "status_message": str(r.get("status_message"))[:400]}
        for _, r in err.iterrows()]
    tool = [json.loads(l) for l in (LOGS_DIR / "tool_calls.jsonl").read_text("utf-8").splitlines() if l.strip()]
    out["tool_failures"] = [{"line": i + 1, **{k: r.get(k) for k in ("timestamp", "run_id", "agent", "tool_name", "args",
                                                                     "status", "error", "latency_ms", "attempts")}}
                            for i, r in enumerate(tool) if r["status"] != "ok"]
    audit = [json.loads(l) for l in (LOGS_DIR / "agent_actions.jsonl").read_text("utf-8").splitlines() if l.strip()]
    interesting = {"degraded_mode", "loop_guard_tripped", "output_repaired", "output_blocked"}
    out["audit_events"] = [{"line": i + 1, **{k: r.get(k) for k in ("timestamp", "run_id", "agent", "action", "decision", "details")}}
                           for i, r in enumerate(audit) if r["action"] in interesting]
    ev = REPORTS_DIR / "eval_report.json"
    if ev.exists():
        rep = json.loads(ev.read_text("utf-8"))
        out["eval_failures"] = [{"id": c["id"], "run_id": c["run_id"], "failed_checks": [k for k, v in c["checks"].items() if not v],
                                 "judge_failed": [k for k, v in (c.get("judge") or {}).items() if v.get("passed") is False]}
                                for c in rep["cases"] if not c["passed"] or any(v.get("passed") is False for v in (c.get("judge") or {}).values())]
    (REPORTS_DIR / "failure_candidates.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print({k: len(v) for k, v in out.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
