"""OPTIONAL bonus: async FastAPI streaming endpoint.

    uvicorn src.api.app:app --port 8000
    curl -N -X POST localhost:8000/investigate -H "content-type: application/json" \
         -d '{"request": "Investigate CLM-1001", "actor_id": "INV-07"}'

Endpoints
    GET  /health                      liveness + model + tracing status
    POST /investigate                 Server-Sent Events: start -> node (one per graph node) -> final
    POST /review                      record a human sign-off for a thread paused at the human gate
Same graph, guardrails, audit trail and Phoenix tracing as the CLI (one Copilot per process, opened in lifespan).
Set COPILOT_OFFLINE=true to serve deterministic fallbacks without Gemini.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import get_settings
from src.runner import Copilot

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    offline = os.getenv("COPILOT_OFFLINE", "").lower() in ("1", "true", "yes")
    if os.getenv("API_TRACING", "true").lower() == "true" and get_settings().tracing_enabled:
        from src.observability.tracing import init_tracing

        STATE["phoenix"] = init_tracing()
    async with Copilot(offline=True if offline else None) as cp:
        STATE["copilot"] = cp
        yield
    STATE.clear()


app = FastAPI(title="Claims Fraud-Investigation Copilot", version="1.0", lifespan=lifespan)


class InvestigateRequest(BaseModel):
    request: str = Field(min_length=1, max_length=4000)
    actor_id: str = "INV-07"
    role: Literal["investigator", "claimant", "reviewer"] = "investigator"
    thread_id: Optional[str] = None
    attachments: list[str] = Field(default_factory=list, description="untrusted claimant text (quarantined)")


class ReviewRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]
    reviewer: str
    note: str = ""


@app.get("/health")
async def health() -> dict:
    cp: Copilot = STATE.get("copilot")
    return {"status": "ok" if cp else "starting", "model": get_settings().gemini_model,
            "offline": bool(cp and cp.deps.llm.offline), "phoenix": STATE.get("phoenix")}


@app.post("/investigate")
async def investigate(req: InvestigateRequest) -> StreamingResponse:
    cp: Copilot = STATE["copilot"]

    async def events():
        async for ev in cp.astream(req.request, thread_id=req.thread_id, actor_id=req.actor_id, actor_role=req.role,
                                   attachments=req.attachments):
            yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'], default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/review")
async def review(req: ReviewRequest) -> dict:
    cp: Copilot = STATE["copilot"]
    if not await cp.pending_signoff(req.thread_id):
        raise HTTPException(409, f"thread {req.thread_id} has no pending sign-off")
    r = await cp.review(req.thread_id, decision=req.decision, reviewer=req.reviewer, note=req.note)
    return {"run_id": r["run_id"], "human_review": r["human_review"], "answer": r["answer"]}
