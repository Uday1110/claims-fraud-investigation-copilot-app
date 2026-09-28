"""MCP client: consumes the custom ``mcp_server`` over stdio via ``langchain-mcp-adapters``.

``open_toolbox()`` starts ONE persistent stdio session for the lifetime of a CLI/API run, converts the server's
tools into LangChain tools (``load_mcp_tools``), loads its resources (``load_mcp_resources``), adds the local RAG
tool, and returns a ``ToolBox`` whose middleware logs every call. Session-level JSON-RPC exchanges
(initialize / tools/list / resources/list / resources/read) are recorded in ``logs/mcp_transcript.jsonl``.
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.tools import load_mcp_tools

from src.config import ROOT
from src.observability.run_context import append_jsonl, current_ids, now_iso
from src.tools.rag_tool import search_fraud_playbook
from src.tools.tool_logging import MCP_TRANSCRIPT, ToolBox

SERVER_NAME = "claims_fraud"
RESOURCE_URIS = ["fraud://indicators/catalog", "fraud://schemes/index"]


def server_config() -> dict:
    return {SERVER_NAME: {"command": sys.executable, "args": ["-m", "mcp_server.server"], "transport": "stdio",
                          "cwd": str(ROOT)}}


def _frame(direction: str, method: str, **payload) -> None:
    append_jsonl(MCP_TRANSCRIPT, {"timestamp": now_iso(), **current_ids(), "server": SERVER_NAME,
                                  "direction": direction, "jsonrpc_method": method, **payload})


@asynccontextmanager
async def open_toolbox() -> AsyncIterator[ToolBox]:
    client = MultiServerMCPClient(server_config())
    t0 = time.perf_counter()
    _frame("client->server", "initialize", params={"transport": "stdio", "command": "python -m mcp_server.server"})
    async with client.session(SERVER_NAME) as session:
        _frame("server->client", "initialize", result={"status": "ok"},
               latency_ms=round((time.perf_counter() - t0) * 1000, 1))

        _frame("client->server", "tools/list", params={})
        listed = await session.list_tools()
        _frame("server->client", "tools/list", result={"tools": [
            {"name": t.name, "inputSchema": t.inputSchema} for t in listed.tools]})

        _frame("client->server", "resources/list", params={})
        res_list = await session.list_resources()
        _frame("server->client", "resources/list", result={"resources": [
            {"uri": str(r.uri), "name": r.name, "mimeType": r.mimeType} for r in res_list.resources]})

        mcp_tools = await load_mcp_tools(session)
        resources: dict[str, str] = {}
        for uri in RESOURCE_URIS:
            _frame("client->server", "resources/read", params={"uri": uri})
            blobs = await load_mcp_resources(session, uris=[uri])
            text = blobs[0].as_string() if blobs else ""
            resources[uri] = text
            _frame("server->client", "resources/read", result={"uri": uri, "bytes": len(text), "preview": text[:160]})

        tools = {t.name: t for t in mcp_tools}
        tools[search_fraud_playbook.name] = search_fraud_playbook
        yield ToolBox(tools=tools, mcp_tool_names={t.name for t in mcp_tools}, mcp_session=session, resources=resources)
    _frame("client->server", "shutdown", params={})
