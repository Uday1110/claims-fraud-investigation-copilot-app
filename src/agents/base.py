"""Shared agent plumbing: dependency container + the node-boundary contract.

``node_boundary`` wraps every graph node so that:
* structured outputs are validated against their Pydantic model before they reach state
  (``history`` -> ClaimHistory, ``assessment`` -> FraudAssessment, ``case_file`` -> CaseFile, ...);
* an unexpected exception inside a node degrades gracefully (error recorded, worker marked complete, audit
  ``degraded_mode``) instead of crashing the run -- except LangGraph control-flow signals (interrupts), which
  are re-raised.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langgraph.errors import GraphBubbleUp

from src.audit.audit import audit
from src.config import Settings
from src.llm import LLMClient
from src.memory.long_term import LongTermMemory
from src.schemas import (CaseFile, ClaimHistory, FraudAssessment, GuardResult, HumanReview, IntentDecision,
                         OutputCheck, SchemeMatch)
from src.state import WORKERS
from src.tools.tool_logging import ToolBox

log = logging.getLogger(__name__)

OUTPUT_MODELS = {
    "intent": IntentDecision,
    "history": ClaimHistory,
    "assessment": FraudAssessment,
    "scheme_match": SchemeMatch,
    "case_file": CaseFile,
    "guard": GuardResult,
    "output_check": OutputCheck,
    "human_review": HumanReview,
}


@dataclass
class Deps:
    llm: LLMClient
    toolbox: ToolBox
    memory: LongTermMemory
    settings: Settings


class NodeOutputError(ValueError):
    pass


def validate_boundary(node: str, out: dict[str, Any]) -> dict[str, Any]:
    for key, model in OUTPUT_MODELS.items():
        if key in out and out[key] is not None:
            try:
                out[key] = model.model_validate(out[key]).model_dump()
            except Exception as exc:
                raise NodeOutputError(f"{node} produced invalid {key}: {exc}") from exc
    return out


def node_boundary(name: str) -> Callable[[Callable[..., Awaitable[dict]]], Callable[..., Awaitable[dict]]]:
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(state: dict, *args, **kwargs) -> dict:
            try:
                out = await fn(state, *args, **kwargs) or {}
                return validate_boundary(name, out)
            except GraphBubbleUp:
                raise
            except Exception as exc:  # noqa: BLE001 -- graceful degradation (NFR-04)
                log.exception("node %s failed", name)
                audit("degraded_mode", decision="node_error_contained", agent=name,
                      details={"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
                out = {"errors": list(state.get("errors") or []) + [{"node": name, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}],
                       "degraded": True}
                if name in WORKERS:
                    out["completed"] = list(state.get("completed") or []) + [name]
                return out
        wrapper.node_name = name
        return wrapper
    return deco
