"""Bonus FastAPI endpoint: SSE streaming of graph progress + human review, offline mode.

Also writes reports/api_demo_run.log (a demonstrated local run of the API).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(l.split(": ", 1) for l in block.splitlines() if ": " in l)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_streaming_investigation_and_review(monkeypatch):
    monkeypatch.setenv("COPILOT_OFFLINE", "true")
    monkeypatch.setenv("API_TRACING", "false")
    from src.api.app import app

    log = []
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        r = c.post("/investigate", json={"request": "Investigate CLM-1003", "actor_id": "INV-07", "thread_id": "api-t1"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        ev = _events(r.text)
        nodes = [d["node"] for e, d in ev if e == "node"]
        final = ev[-1][1]
        assert ev[0][0] == "start" and ev[-1][0] == "final"
        assert nodes[:3] == ["input_guard", "load_context", "intent_router"] and "case_writer" in nodes
        assert final["pending_signoff"] is True and final["case_file"]["recommended_action"] == "refer_to_siu"
        log.append(f"POST /investigate CLM-1003 -> events={[e for e, _ in ev]}")
        log.append(f"  nodes streamed: {nodes}")
        log.append(f"  final: intent={final['intent']} action={final['case_file']['recommended_action']} pending_signoff={final['pending_signoff']}")
        rv = c.post("/review", json={"thread_id": "api-t1", "decision": "approve", "reviewer": "SIU-LEAD-01"})
        assert rv.status_code == 200 and rv.json()["human_review"]["status"] == "approved"
        log.append(f"POST /review api-t1 approve SIU-LEAD-01 -> {rv.json()['human_review']}")
        assert c.post("/review", json={"thread_id": "api-t1", "decision": "approve", "reviewer": "SIU-LEAD-01"}).status_code == 409
        blocked = _events(c.post("/investigate", json={"request": "Investigate CLM-1003", "actor_id": "INV-12"}).text)[-1][1]
        assert "not assigned" in blocked["answer"]
        log.append(f"POST /investigate as INV-12 -> refused: {blocked['answer'][:80]}")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "api_demo_run.log").write_text("# FastAPI streaming demo (tests/test_api.py, offline mode)\n" +
                                                       "\n".join(log) + "\n", encoding="utf-8")
