"""Tool-contract tests: every tool's input/output schema + error paths, over the REAL MCP stdio server."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from mcp_server import server as mcp_srv
from src.tools.mcp_client import RESOURCE_URIS, open_toolbox
from src.tools.rag_tool import TOOL_NAME as RAG_TOOL, PlaybookSearchOutput
from src.tools.tool_logging import ToolBox

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MCP_TOOLS = {"get_claim_history", "score_red_flags", "find_linked_claims"}
REQUIRED_LOG_FIELDS = {"timestamp", "agent", "tool_name", "args", "result", "latency_ms", "status"}


# ---- output contracts ------------------------------------------------------------------------------------
class ClaimHistoryOut(BaseModel):
    claim_id: str
    policy_ref_masked: str
    claimant_ref_masked: str
    line_of_business: str
    loss_type: str
    policy_tenure_days: int
    report_lag_days: int
    claimed_amount: float
    prior_claims: list[dict]
    claimant_statement_untrusted: str


class Indicator(BaseModel):
    code: str
    name: str
    weight: int
    evidence: str


class RedFlagsOut(BaseModel):
    claim_id: str
    score: int
    tier: str
    indicators: list[Indicator]


class LinkedOut(BaseModel):
    claim_id: str
    links: list[dict]
    ring_suspected: bool


CONTRACTS = {"get_claim_history": ClaimHistoryOut, "score_red_flags": RedFlagsOut, "find_linked_claims": LinkedOut}


async def test_mcp_server_exposes_expected_tools_and_input_schemas():
    async with open_toolbox() as tb:
        assert EXPECTED_MCP_TOOLS == tb.mcp_tool_names
        for name in EXPECTED_MCP_TOOLS:
            schema = tb.get(name).args_schema
            schema = schema if isinstance(schema, dict) else schema.model_json_schema()
            assert schema["required"] == ["claim_id"]
            assert schema["properties"]["claim_id"]["type"] == "string"
        assert RAG_TOOL in tb.names()


@pytest.mark.parametrize("tool", sorted(EXPECTED_MCP_TOOLS))
async def test_mcp_tool_output_schema(tool):
    async with open_toolbox() as tb:
        res = await tb.call(tool, {"claim_id": "CLM-1001"}, agent="contract_test")
    assert res.ok, res.error
    CONTRACTS[tool].model_validate(res.data)


async def test_mcp_outputs_are_masked_at_source():
    async with open_toolbox() as tb:
        res = await tb.call("get_claim_history", {"claim_id": "CLM-1001"}, agent="contract_test")
    assert res.data["policy_ref_masked"].startswith("POL-****")
    assert res.data["claimant_ref_masked"].startswith("CLT-****")
    raw = json.dumps(res.data)
    assert "claimant_phone" not in raw and "claimant_name" not in raw


@pytest.mark.parametrize("bad_id,msg", [("CLM-99999", "invalid claim_id"), ("1001", "invalid claim_id"),
                                        ("CLM-9999", "not found")])
async def test_mcp_tool_error_path(bad_id, msg):
    async with open_toolbox() as tb:
        res = await tb.call("score_red_flags", {"claim_id": bad_id}, agent="contract_test")
    assert res.status == "error" and msg in res.error
    assert res.attempts == 1                      # validation errors are not retried


async def test_missing_argument_is_rejected():
    async with open_toolbox() as tb:
        res = await tb.call("get_claim_history", {}, agent="contract_test")
    assert res.status == "error"


async def test_mcp_resources_are_readable_json():
    async with open_toolbox() as tb:
        assert set(RESOURCE_URIS) <= set(tb.resources)
        catalog = json.loads(tb.resources["fraud://indicators/catalog"])
        assert {"RF-01", "RF-16"} <= set(catalog["indicators"]) and "tiers" in catalog
        index = json.loads(tb.resources["fraud://schemes/index"])
        assert {d["doc_id"] for d in index} >= {"FS-01", "GL-01"}


async def test_rag_tool_contract_and_error_path():
    async with open_toolbox() as tb:
        ok = await tb.call(RAG_TOOL, {"query": "staged rear-end collision with soft tissue injuries", "k": 3}, agent="t")
        assert ok.ok
        out = PlaybookSearchOutput.model_validate(ok.data)
        assert len(out.results) == 3 and out.results[0].doc_id == "FS-01"
        bad_k = await tb.call(RAG_TOOL, {"query": "arson", "k": 0}, agent="t")
        bad_q = await tb.call(RAG_TOOL, {"query": "", "k": 2}, agent="t")
        bad_f = await tb.call(RAG_TOOL, {"query": "arson", "doc_filter": "XX"}, agent="t")
    assert bad_k.status == bad_q.status == bad_f.status == "error"


async def test_unknown_tool_and_timeout_paths(tmp_path):
    async def slow(x: str) -> str:
        await asyncio.sleep(5)
        return x

    tb = ToolBox(tools={"slow": StructuredTool.from_function(coroutine=slow, name="slow", description="slow")},
                 timeout_s=0.2, max_attempts=2, log_path=tmp_path / "calls.jsonl")
    unknown = await tb.call("nope", {}, agent="t")
    timed = await tb.call("slow", {"x": "a"}, agent="t")
    assert unknown.status == "error" and "unknown tool" in unknown.error
    assert timed.status == "timeout" and timed.attempts == 2          # transient -> retried once
    recs = [json.loads(l) for l in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert all(REQUIRED_LOG_FIELDS <= set(r) for r in recs)


def _registered_tool_names() -> set[str]:
    names = set()
    for name in dir(mcp_srv):
        obj = getattr(mcp_srv, name)
        if callable(obj) and name in EXPECTED_MCP_TOOLS:
            names.add(name)
    return names | {RAG_TOOL}


def test_tool_log_names_reconcile_with_code():
    """AC-07: every tool_name in the committed logs/tool_calls.jsonl exists in the MCP server or RAG tool code."""
    log = ROOT / "logs" / "tool_calls.jsonl"
    if not log.exists():
        pytest.skip("no committed tool log yet")
    records = [json.loads(l) for l in log.read_text("utf-8").splitlines() if l.strip()]
    assert records, "tool log is empty"
    code_names = _registered_tool_names()
    logged = {r["tool_name"] for r in records}
    assert logged <= code_names, f"unreconciled tool names: {logged - code_names}"
    for r in records:
        assert REQUIRED_LOG_FIELDS <= set(r), r
        assert r["status"] in ("ok", "error", "timeout")
