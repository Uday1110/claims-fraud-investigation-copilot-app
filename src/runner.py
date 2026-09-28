"""Copilot runtime: owns the MCP session, checkpointer, long-term memory and graph for one process.

    async with Copilot() as cp:
        r = await cp.ask("Investigate CLM-1001", thread_id="t1", actor_id="INV-07")
        if r["pending_signoff"]:
            await cp.review("t1", decision="approve", reviewer="SIU-LEAD-01")

Every turn gets a fresh ``run_id`` that is attached to Phoenix spans (openinference ``metadata.run_id`` +
``session.id`` = thread) and to every JSONL evidence log.
"""
from __future__ import annotations

import asyncio
import contextlib
import uuid
from contextlib import AsyncExitStack
from typing import Any, Optional

from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from src.agents.base import Deps
from src.audit.audit import audit
from src.config import Settings, get_settings
from src.graph import build_graph
from src.guardrails.access import resolve_actor
from src.guardrails.pii import mask_text
from src.llm import LLMClient
from src.memory.long_term import LongTermMemory
from src.memory.short_term import open_checkpointer
from src.observability.cost import CostMeter, meter_var
from src.observability.run_context import actor_var, run_id_var, thread_id_var
from src.observability.tracing import trace_attributes
from src.tools.mcp_client import open_toolbox

RUN_TIMEOUT_S = 600


class Copilot:
    def __init__(self, settings: Optional[Settings] = None, *, offline: Optional[bool] = None, toolbox=None):
        self.settings = settings or get_settings()
        self.offline = offline
        self._external_toolbox = toolbox
        self._stack = AsyncExitStack()
        self.graph = None
        self.deps: Optional[Deps] = None

    @staticmethod
    def warm_up() -> None:
        """Load the heavy local models at start-up, not inside the first user request.

        Fix for failure F-1 (docs/failure-analysis.md): lazy loading of spaCy/Presidio and the Sentence-Transformers
        model made the first run's input_guard take ~45 s and its first playbook search ~10 s.
        """
        from src.guardrails.pii import detect_pii
        from src.tools.rag_tool import embed, get_index

        detect_pii("warm-up POL-00000000")
        get_index()
        embed(["warm-up"])

    async def __aenter__(self) -> "Copilot":
        await asyncio.to_thread(self.warm_up)
        toolbox =self._external_toolbox or await self._stack.enter_async_context(open_toolbox())
        saver = await self._stack.enter_async_context(open_checkpointer(self.settings.checkpoint_db))
        self.deps = Deps(llm=LLMClient(self.settings, offline=self.offline), toolbox=toolbox,
                         memory=LongTermMemory(self.settings.memory_db), settings=self.settings)
        self.graph = build_graph(self.deps, checkpointer=saver)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    def _config(self, thread_id: str, run_id: str, actor_id: str, tags: list[str]) -> dict:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": self.settings.recursion_limit,
                "run_name": "copilot_turn", "tags": ["claims-fraud-copilot", *tags],
                "metadata": {"run_id": run_id, "thread_id": thread_id, "actor": actor_id}}

    async def _invoke(self, payload: Any, *, thread_id: str, actor_id: str, tags: list[str]) -> dict:
        run_id = uuid.uuid4().hex[:16]
        run_id_var.set(run_id)
        thread_id_var.set(thread_id)
        actor_var.set(actor_id)
        meter = CostMeter()
        meter_var.set(meter)
        config = self._config(thread_id, run_id, actor_id, tags)
        error = None
        with trace_attributes(session_id=thread_id, user_id=actor_id, run_id=run_id, tags=tags):
            try:
                await asyncio.wait_for(self.graph.ainvoke(payload, config), timeout=RUN_TIMEOUT_S)
            except GraphRecursionError as exc:          # framework-level runaway backstop
                error = f"GraphRecursionError: {exc}"
                audit("loop_guard_tripped", decision="recursion_limit", agent="runner",
                      details={"recursion_limit": self.settings.recursion_limit})
            except asyncio.TimeoutError:
                error = f"run exceeded {RUN_TIMEOUT_S}s"
                audit("degraded_mode", decision="run_timeout", agent="runner", details={"timeout_s": RUN_TIMEOUT_S})
        snap = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        values = dict(snap.values or {})
        interrupts = [i.value for t in (snap.tasks or []) for i in (getattr(t, "interrupts", None) or [])]
        cost = meter.flush({"intent": (values.get("intent") or {}).get("intent"), "tags": tags})
        return {
            "run_id": run_id, "thread_id": thread_id,
            "answer": values.get("final_answer") if not error else f"Run stopped safely: {error}",
            "intent": (values.get("intent") or {}).get("intent"),
            "claim_id": values.get("claim_id"),
            "case_file": values.get("case_file"),
            "assessment": values.get("assessment"),
            "scheme_match": values.get("scheme_match"),
            "history": values.get("history"),
            "quarantine": values.get("quarantine"),
            "tool_evidence": values.get("tool_evidence"),
            "retrieved_chunks": values.get("retrieved_chunks"),
            "memory_context": values.get("memory_context"),
            "output_check": values.get("output_check"),
            "human_review": values.get("human_review"),
            "guard": values.get("guard"),
            "degraded": bool(values.get("degraded")) or bool(error),
            "errors": values.get("errors") or ([{"node": "runner", "error": error}] if error else []),
            "pending_signoff": bool(interrupts),
            "interrupt": interrupts[0] if interrupts else None,
            "cost": cost,
        }

    async def ask(self, request: str, *, thread_id: Optional[str] = None, actor_id: str = "INV-07",
                  actor_role: str = "investigator", attachments: Optional[list[str]] = None,
                  tags: Optional[list[str]] = None) -> dict:
        thread_id = thread_id or f"thread-{uuid.uuid4().hex[:10]}"
        pending = await self.pending_signoff(thread_id)
        if pending:
            # the human gate blocks the thread: a new turn would silently discard the pending decision
            return {"run_id": None, "thread_id": thread_id, "intent": None, "claim_id": pending.get("claim_id"),
                    "answer": (f"Thread {thread_id} is awaiting human SIU sign-off for {pending.get('claim_id')} "
                               f"({pending.get('requested_action')}). Record the decision first: python -m src.cli review "
                               f"--thread {thread_id} --decision approve|reject --reviewer <SIU lead>."),
                    "pending_signoff": True, "interrupt": pending, "degraded": False, "errors": [],
                    "output_check": {"risk_tier": "high"}, "cost": {"cost_usd": 0.0}}
        actor = resolve_actor(actor_role, actor_id)
        # ingress PII middleware: identifiers are masked before the payload reaches the graph, so they can never
        # appear in checkpoints, Phoenix span inputs or logs (NFR-05). Detection-based guards still see the intent.
        payload = {"request": mask_text(request), "actor": actor,
                   "attachments": [mask_text(a) for a in (attachments or [])], "thread_id": thread_id}
        return await self._invoke(payload, thread_id=thread_id, actor_id=actor_id, tags=tags or [])

    async def astream(self, request: str, *, thread_id: Optional[str] = None, actor_id: str = "INV-07",
                      actor_role: str = "investigator", attachments: Optional[list[str]] = None):
        """Async generator of per-node progress events, then the final result (used by the FastAPI endpoint)."""
        thread_id = thread_id or f"api-{uuid.uuid4().hex[:10]}"
        if await self.pending_signoff(thread_id):
            yield {"event": "final", "data": await self.ask(request, thread_id=thread_id, actor_id=actor_id)}
            return
        run_id = uuid.uuid4().hex[:16]
        run_id_var.set(run_id)
        thread_id_var.set(thread_id)
        actor_var.set(actor_id)
        meter = CostMeter()
        meter_var.set(meter)
        payload = {"request": mask_text(request), "actor": resolve_actor(actor_role, actor_id),
                   "attachments": [mask_text(a) for a in (attachments or [])], "thread_id": thread_id}
        config = self._config(thread_id, run_id, actor_id, ["api"])
        yield {"event": "start", "data": {"run_id": run_id, "thread_id": thread_id}}
        with trace_attributes(session_id=thread_id, user_id=actor_id, run_id=run_id, tags=["api"]):
            async for upd in self.graph.astream(payload, config, stream_mode="updates"):
                for node, out in upd.items():
                    if node.startswith("__"):
                        continue
                    yield {"event": "node", "data": {"node": node, "keys": sorted((out or {}).keys())}}
        snap = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        v = snap.values or {}
        interrupts = [i.value for t in (snap.tasks or []) for i in (getattr(t, "interrupts", None) or [])]
        cost = meter.flush({"intent": (v.get("intent") or {}).get("intent"), "tags": ["api"]})
        yield {"event": "final", "data": {"run_id": run_id, "thread_id": thread_id, "answer": v.get("final_answer"),
                                          "intent": (v.get("intent") or {}).get("intent"), "case_file": v.get("case_file"),
                                          "output_check": v.get("output_check"), "pending_signoff": bool(interrupts),
                                          "interrupt": interrupts[0] if interrupts else None, "cost_usd": cost["cost_usd"]}}

    async def pending_signoff(self, thread_id: str) -> Optional[dict]:
        snap = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        for t in snap.tasks or []:
            for i in getattr(t, "interrupts", None) or []:
                return i.value
        return None

    async def review(self, thread_id: str, *, decision: str, reviewer: str, note: str = "",
                     tags: Optional[list[str]] = None) -> dict:
        """Resume a run paused at the human gate with a reviewer's sign-off decision."""
        return await self._invoke(Command(resume={"decision": decision, "reviewer": reviewer, "note": note}),
                                  thread_id=thread_id, actor_id=reviewer, tags=(tags or []) + ["human_review"])


@contextlib.asynccontextmanager
async def copilot_session(**kw):
    async with Copilot(**kw) as cp:
        yield cp
