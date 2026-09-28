"""Arize Phoenix instrumentation (mandated) -- local, in-process.

``init_tracing()`` is CALLED on the run path by the CLI, the API and the evidence-regeneration script:

1. launches the Phoenix server in-process (``phoenix.launch_app``; UI at http://localhost:6006),
2. registers an OpenTelemetry tracer provider pointed at it (``phoenix.otel.register``),
3. instruments LangChain/LangGraph with ``openinference-instrumentation-langchain`` so every graph node (CHAIN),
   every Gemini call (LLM, with token counts incl. reasoning tokens) and every tool call (TOOL; MCP + RAG) becomes
   a span.

``trace_attributes()`` attaches ``session.id`` (= thread), ``user.id`` (= actor) and ``metadata.run_id`` to every
span of a turn, so a span can be cited as (run_id, span_id). ``export_spans()`` writes the committed trace export.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator, Optional

from src.config import RUNTIME_DIR, TRACES_DIR, get_settings

log = logging.getLogger(__name__)
_STATE: dict = {"session": None, "provider": None, "instrumented": False, "base_url": None}


def init_tracing(*, launch: bool = True, project: Optional[str] = None, fresh: bool = True) -> Optional[str]:
    """Start Phoenix + instrument LangChain. Returns the Phoenix base URL (or None if tracing disabled).

    ``fresh=False`` re-opens the existing local Phoenix store (data/runtime/phoenix) instead of wiping it, e.g. to
    export the spans of a run whose process stopped before its export step (scripts/finalize_offline.py).
    """
    s = get_settings()
    if not s.tracing_enabled:
        return None
    if _STATE["instrumented"]:
        return _STATE["base_url"]
    # Phoenix logs emoji; a cp1252 Windows console would crash its server thread -> force UTF-8 streams
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PHOENIX_PORT"] = str(s.phoenix_port)
    os.environ.setdefault("PHOENIX_PROJECT_NAME", project or s.phoenix_project)
    # fresh, project-local Phoenix store per process (git-ignored runtime dir)
    work = RUNTIME_DIR / "phoenix"
    if fresh:
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    os.environ["PHOENIX_WORKING_DIR"] = str(work)
    base_url = f"http://127.0.0.1:{s.phoenix_port}"
    if launch:
        import phoenix as px

        import time as _time

        import phoenix.server.thread_server as _ts

        # Phoenix's thread server hard-codes a 5 s start-up limit (``time() + 5``); a cold start that has to create
        # and migrate a fresh SQLite DB takes longer on Windows, and a failed start cannot be retried in-process.
        # ``time`` is used only for that limit in this module, so a 12x slower clock = 60 s start-up budget.
        _ts.time = lambda: _time.time() / 12.0
        _STATE["session"] = px.launch_app(use_temp_dir=False, run_in_thread=True)
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.semconv.resource import ResourceAttributes
    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # equivalent of phoenix.otel.register(batch=False), built directly on the OTel SDK
    provider = TracerProvider(resource=Resource({ResourceAttributes.PROJECT_NAME: project or s.phoenix_project}))
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{base_url}/v1/traces")))
    otel_trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _STATE.update(provider=provider, instrumented=True, base_url=base_url)
    return base_url


@contextlib.contextmanager
def trace_attributes(*, session_id: str, user_id: str, run_id: str, tags: list[str]) -> Iterator[None]:
    """Propagate run correlation ids onto every span created inside the block."""
    try:
        from openinference.instrumentation import using_attributes
    except ImportError:  # pragma: no cover
        yield
        return
    with using_attributes(session_id=session_id, user_id=user_id, metadata={"run_id": run_id}, tags=tags):
        yield


def flush() -> None:
    p = _STATE.get("provider")
    if p is not None:
        with contextlib.suppress(Exception):
            p.force_flush()


def get_spans_dataframe(project: Optional[str] = None):
    """Pull every span of the project from the running Phoenix server."""
    from phoenix.client import Client

    flush()
    s = get_settings()
    client = Client(base_url=_STATE["base_url"] or f"http://127.0.0.1:{s.phoenix_port}")
    return client.spans.get_spans_dataframe(project_identifier=project or s.phoenix_project, limit=1_000_000, timeout=120)


def _run_id_of(meta) -> Optional[str]:
    if isinstance(meta, str):
        with contextlib.suppress(Exception):
            meta = json.loads(meta)
    return meta.get("run_id") if isinstance(meta, dict) else None


def export_spans(out_dir: Path = TRACES_DIR, project: Optional[str] = None) -> dict:
    """Export spans to traces/phoenix_spans.parquet (+ a JSONL mirror that is easy to grep/cite)."""
    df = get_spans_dataframe(project)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=df.index.name in df.columns or df.index.name is None)
    if "attributes.metadata" in df.columns:
        df["run_id"] = df["attributes.metadata"].map(_run_id_of)
    # propagate run_id to every span of the same trace (root carries the metadata)
    if "context.trace_id" in df.columns and "run_id" in df.columns:
        by_trace = df.dropna(subset=["run_id"]).groupby("context.trace_id")["run_id"].first()
        df["run_id"] = df["run_id"].fillna(df["context.trace_id"].map(by_trace))
    pq = out_dir / "phoenix_spans.parquet"
    safe = df.copy()
    for c in safe.columns:
        if safe[c].dtype == object:
            safe[c] = safe[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else (None if v is None else str(v)))
    safe.to_parquet(pq, index=False)
    jl = out_dir / "phoenix_spans.jsonl"
    keep = [c for c in ("run_id", "context.trace_id", "context.span_id", "parent_id", "name", "span_kind", "start_time",
                        "end_time", "status_code", "status_message", "attributes.llm.token_count.prompt",
                        "attributes.llm.token_count.completion", "attributes.llm.token_count.total",
                        "attributes.llm.model_name", "attributes.session.id", "attributes.tool.name") if c in df.columns]
    with jl.open("w", encoding="utf-8") as f:
        for rec in df[keep].to_dict(orient="records"):
            if rec.get("start_time") is not None and rec.get("end_time") is not None:
                rec["latency_ms"] = round((rec["end_time"] - rec["start_time"]).total_seconds() * 1000, 1)
            f.write(json.dumps(rec, default=str) + "\n")
    return {"spans": len(df), "parquet": str(pq), "jsonl": str(jl),
            "runs": int(df["run_id"].nunique()) if "run_id" in df.columns else 0}
