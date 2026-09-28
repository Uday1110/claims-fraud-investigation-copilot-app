"""Tool-invocation middleware (AC-07, NFR-04).

Every tool the agents use -- the three MCP tools and the local RAG tool -- is executed through
``ToolBox.call``. The middleware:

* enforces a per-call timeout and bounded retries with exponential back-off for transient failures,
* never raises into the graph: failures come back as ``ToolResult(status="error"|"timeout")`` so the calling agent
  can degrade gracefully,
* appends one machine-generated record per call to ``logs/tool_calls.jsonl``:
  ``{timestamp, run_id, thread_id, agent, tool_name, tool_source, args, result, latency_ms, status, error, attempt}``
* mirrors MCP calls into ``logs/mcp_transcript.jsonl`` (request + response frames).

Tool names in the log are the registered tool names, which reconcile 1:1 with ``mcp_server/server.py`` and
``src/tools/rag_tool.py`` (checked by ``tests/test_tool_contracts.py::test_tool_log_names_reconcile_with_code``).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.tools import BaseTool

from src.audit.audit import audit
from src.config import LOGS_DIR, get_settings
from src.observability.run_context import append_jsonl, current_ids, now_iso

TOOL_LOG = LOGS_DIR / "tool_calls.jsonl"
MCP_TRANSCRIPT = LOGS_DIR / "mcp_transcript.jsonl"

_TRANSIENT = ("timeout", "temporarily", "connection", "unavailable", "broken pipe", "reset")


@dataclass
class ToolResult:
    tool_name: str
    status: str                      # ok | error | timeout
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def parse_tool_output(raw: Any) -> Any:
    """Normalise LangChain / MCP-adapter tool output (str | list[str] | content blocks | tuple) to Python."""
    if isinstance(raw, tuple) and len(raw) == 2:
        raw = raw[0]
    if isinstance(raw, list):
        texts = []
        for item in raw:
            if isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            elif hasattr(item, "text"):
                texts.append(item.text)
            elif isinstance(item, str):
                texts.append(item)
            else:
                return raw
        raw = texts[0] if len(texts) == 1 else "\n".join(texts)
    if hasattr(raw, "content") and not isinstance(raw, (str, bytes)):
        return parse_tool_output(raw.content)
    if isinstance(raw, str):
        s = raw.strip()
        if s[:1] in "{[":
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return raw
    return raw


def _summarise(data: Any, limit: int = 1500) -> Any:
    s = json.dumps(data, default=str)
    return data if len(s) <= limit else {"truncated": True, "preview": s[:limit], "chars": len(s)}


@dataclass
class ToolBox:
    tools: dict[str, BaseTool]
    mcp_tool_names: set[str] = field(default_factory=set)
    timeout_s: float = field(default_factory=lambda: get_settings().tool_timeout_s)
    max_attempts: int = 2
    log_path: Any = TOOL_LOG
    mcp_session: Any = None          # live MCP ClientSession (for resource reads), may be None in tests
    resources: dict[str, str] = field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.tools)

    def get(self, name: str) -> BaseTool:
        return self.tools[name]

    async def call(self, name: str, args: dict[str, Any], *, agent: str) -> ToolResult:
        source = "mcp" if name in self.mcp_tool_names else "local"
        if name not in self.tools:
            res = ToolResult(name, "error", error=f"unknown tool {name!r}; available: {self.names()}")
            self._log(res, args, agent, source)
            return res
        tool = self.tools[name]
        attempt, t0 = 0, time.perf_counter()
        last_err: Optional[str] = None
        status = "error"
        while attempt < self.max_attempts:
            attempt += 1
            if source == "mcp":
                append_jsonl(MCP_TRANSCRIPT, {"timestamp": now_iso(), **current_ids(), "direction": "client->server",
                                              "jsonrpc_method": "tools/call", "params": {"name": name, "arguments": args},
                                              "attempt": attempt})
            try:
                raw = await asyncio.wait_for(tool.ainvoke(args), timeout=self.timeout_s)
                data = parse_tool_output(raw)
                if isinstance(data, str) and data.lower().startswith(("error", "tool error")):
                    raise RuntimeError(data)
                res = ToolResult(name, "ok", data=data, latency_ms=round((time.perf_counter() - t0) * 1000, 1), attempts=attempt)
                if source == "mcp":
                    append_jsonl(MCP_TRANSCRIPT, {"timestamp": now_iso(), **current_ids(), "direction": "server->client",
                                                  "jsonrpc_method": "tools/call", "tool": name, "is_error": False,
                                                  "result": _summarise(data, 800), "latency_ms": res.latency_ms})
                self._log(res, args, agent, source)
                return res
            except asyncio.TimeoutError:
                last_err, status = f"timed out after {self.timeout_s}s", "timeout"
            except Exception as exc:  # MCP isError -> ToolException, validation errors, transport errors
                last_err, status = f"{type(exc).__name__}: {exc}", "error"
            if source == "mcp":
                append_jsonl(MCP_TRANSCRIPT, {"timestamp": now_iso(), **current_ids(), "direction": "server->client",
                                              "jsonrpc_method": "tools/call", "tool": name, "is_error": True,
                                              "error": last_err})
            transient = status == "timeout" or any(k in (last_err or "").lower() for k in _TRANSIENT)
            if not transient:
                break
            await asyncio.sleep(0.5 * 2 ** (attempt - 1))
        res = ToolResult(name, status, error=last_err, latency_ms=round((time.perf_counter() - t0) * 1000, 1), attempts=attempt)
        self._log(res, args, agent, source)
        return res

    def _log(self, res: ToolResult, args: dict, agent: str, source: str) -> None:
        append_jsonl(self.log_path, {
            "timestamp": now_iso(),
            **current_ids(),
            "agent": agent,
            "node": agent,
            "tool_name": res.tool_name,
            "tool_source": source,
            "args": args,
            "result": _summarise(res.data) if res.ok else None,
            "latency_ms": res.latency_ms,
            "status": res.status,
            "error": res.error,
            "attempts": res.attempts,
        })
        audit("tool_invoked", decision="executed" if res.ok else f"failed:{res.status}", tool=res.tool_name,
              agent=agent, details={"latency_ms": res.latency_ms, "source": source})
