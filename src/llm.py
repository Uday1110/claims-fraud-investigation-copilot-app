"""Gemini access layer (Google Gemini is the only model provider).

``LLMClient`` wraps ``langchain_google_genai.ChatGoogleGenerativeAI`` and adds the resilience required by
NFR-04:
* async calls with a hard timeout (``asyncio.wait_for``),
* bounded retries with exponential back-off (longer for 429 / quota errors),
* structured output (Pydantic) with parse-error detection (``include_raw=True``) and retry,
* graceful degradation: every call site supplies a deterministic ``fallback``; if Gemini is unavailable after
  retries -- or no API key is configured (offline mode) -- the fallback result is used and the degradation is
  audited, so the graph still produces a (flagged) answer instead of crashing,
* cost governance: token usage is metered per run and *optional* calls are skipped once over budget.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import random
import re
from typing import Any, Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from src.audit.audit import audit
from src.config import Settings, get_settings
from src.observability.cost import current_meter

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    pass


def build_chat_model(settings: Settings, model: Optional[str] = None):
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {
        "model": model or settings.gemini_model,
        "google_api_key": settings.google_api_key,
        "max_retries": 0,               # retries are handled here, visibly, per attempt
        "timeout": settings.llm_timeout_s,
    }
    if settings.thinking_level:
        # Gemini 3+ native thinking control (alias of reasoning_effort in langchain-google-genai >= 4)
        kwargs["thinking_level"] = settings.thinking_level
    return ChatGoogleGenerativeAI(**kwargs)


class RateLimiter:
    """Process-wide client-side limiter: max concurrent Gemini calls + minimum spacing (requests/minute)."""

    def __init__(self, max_concurrency: int, rpm: int):
        self.sem = asyncio.Semaphore(max(1, max_concurrency))
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self.sem.acquire()
        if self.interval:
            async with self._lock:
                loop = asyncio.get_running_loop()
                wait = self._next - loop.time()
                self._next = max(loop.time(), self._next) + self.interval
            if wait > 0:
                await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *exc):
        self.sem.release()


_LIMITERS: dict[int, RateLimiter] = {}


def limiter(settings: Settings) -> RateLimiter:
    """One limiter per event loop (asyncio primitives are loop-bound)."""
    key = id(asyncio.get_running_loop())
    if key not in _LIMITERS:
        _LIMITERS[key] = RateLimiter(settings.llm_max_concurrency, settings.llm_rpm)
    return _LIMITERS[key]


_RETRY_DELAY_RE = re.compile(r"retry in ([0-9.]+)s|'retryDelay': '([0-9.]+)s'", re.I)


def _retryable(exc: Exception) -> tuple[bool, float]:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "perday" in msg.replace(" ", "") or "per day" in msg:
        return False, 0.0                      # daily quota exhausted: retrying cannot help -> degrade now
    if "402" in msg or "credits are depleted" in msg or "billing" in msg:
        return False, 0.0                      # prepaid credit exhausted (failure F-7): fail over immediately
    if any(k in msg for k in ("429", "resource_exhausted", "quota", "rate")):
        m = _RETRY_DELAY_RE.search(str(exc))
        hinted = float(next(g for g in m.groups() if g)) if m else 8.0
        return True, min(hinted, 60.0)
    if any(k in msg for k in ("timeout", "deadline", "503", "500", "unavailable", "internal", "overloaded",
                              "connection", "parse", "validation", "structured output", "none")):
        return True, 1.5
    return False, 0.0


class LLMClient:
    def __init__(self, settings: Optional[Settings] = None, *, offline: Optional[bool] = None, model: Optional[str] = None):
        self.settings = settings or get_settings()
        self.offline = (not self.settings.google_api_key) if offline is None else offline
        self.model_name = model or self.settings.gemini_model
        self._chat = None if self.offline else build_chat_model(self.settings, self.model_name)

    @property
    def chat(self):
        if self._chat is None:
            raise LLMUnavailable("offline mode: no Gemini client")
        return self._chat

    async def _fallback(self, fallback: Optional[Callable[[], Any]], agent: str, reason: str) -> Any:
        if fallback is None:
            raise LLMUnavailable(reason)
        audit("degraded_mode", decision="deterministic_fallback", agent=agent, details={"reason": reason[:300]})
        out = fallback()
        return await out if inspect.isawaitable(out) else out

    async def structured(self, schema: Type[T], messages: list, *, agent: str,
                         fallback: Optional[Callable[[], T]] = None, optional: bool = False) -> tuple[T, dict]:
        """Return (parsed_model, meta). meta = {degraded, attempts, error, usage}."""
        meter = current_meter()
        if self.offline:
            return await self._fallback(fallback, agent, "offline"), {"degraded": True, "attempts": 0, "error": "offline"}
        if optional and meter.over_budget():
            meter.skipped_optional_calls += 1
            return await self._fallback(fallback, agent, "cost budget exceeded"), {"degraded": True, "attempts": 0,
                                                                                  "error": "budget"}
        runnable = self.chat.with_structured_output(schema, include_raw=True)
        last: Optional[Exception] = None
        attempts = 0
        for attempt in range(1, self.settings.llm_max_retries + 1):
            attempts = attempt
            try:
                async with limiter(self.settings):
                    out = await asyncio.wait_for(
                        runnable.ainvoke(messages, config={"run_name": f"{agent}:{schema.__name__}",
                                                           "tags": [agent], "metadata": {"agent": agent, "attempt": attempt}}),
                        timeout=self.settings.llm_timeout_s)
                raw = out.get("raw") if isinstance(out, dict) else None
                meter.add(getattr(raw, "usage_metadata", None), agent)
                parsed = out.get("parsed") if isinstance(out, dict) else out
                if parsed is None:
                    raise ValueError(f"structured output parse failure: {out.get('parsing_error') if isinstance(out, dict) else 'none'}")
                if isinstance(parsed, dict):
                    parsed = schema.model_validate(parsed)
                return parsed, {"degraded": False, "attempts": attempt, "error": None,
                                "usage": getattr(raw, "usage_metadata", None)}
            except Exception as exc:  # noqa: BLE001 - classified below
                last = exc
                retry, base = _retryable(exc)
                log.warning("LLM %s attempt %d failed: %s", agent, attempt, str(exc)[:200])
                if not retry or attempt == self.settings.llm_max_retries:
                    break
                await asyncio.sleep(base * 2 ** (attempt - 1) + random.random())
        return await self._fallback(fallback, agent, f"{type(last).__name__}: {last}"), {
            "degraded": True, "attempts": attempts, "error": f"{type(last).__name__}: {str(last)[:300]}"}

    async def text(self, messages: list, *, agent: str, fallback: Optional[Callable[[], str]] = None,
                   optional: bool = False) -> tuple[str, dict]:
        meter = current_meter()
        if self.offline:
            return await self._fallback(fallback, agent, "offline"), {"degraded": True}
        if optional and meter.over_budget():
            meter.skipped_optional_calls += 1
            return await self._fallback(fallback, agent, "cost budget exceeded"), {"degraded": True}
        last: Optional[Exception] = None
        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                async with limiter(self.settings):
                    msg = await asyncio.wait_for(
                        self.chat.ainvoke(messages, config={"run_name": f"{agent}:text", "tags": [agent],
                                                            "metadata": {"agent": agent, "attempt": attempt}}),
                        timeout=self.settings.llm_timeout_s)
                meter.add(getattr(msg, "usage_metadata", None), agent)
                content = msg.content
                if isinstance(content, list):
                    content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
                if not str(content).strip():
                    raise ValueError("empty completion")
                return str(content), {"degraded": False, "attempts": attempt}
            except Exception as exc:  # noqa: BLE001
                last = exc
                retry, base = _retryable(exc)
                if not retry or attempt == self.settings.llm_max_retries:
                    break
                await asyncio.sleep(base * 2 ** (attempt - 1) + random.random())
        return await self._fallback(fallback, agent, f"{type(last).__name__}: {last}"), {"degraded": True,
                                                                                          "error": str(last)[:300]}
