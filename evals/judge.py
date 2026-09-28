"""LLM-as-judge model for DeepEval: Google Gemini (the only approved provider), with resilient retries.

DeepEval's ``GeminiModel`` calls the google-genai SDK directly; under load Gemini returns 503 UNAVAILABLE /
429 RESOURCE_EXHAUSTED, so every judge call is wrapped in bounded exponential back-off. Judge calls do NOT go
through LangChain, so they are not traced into the copilot's Phoenix project (they would distort its golden signals).
"""
from __future__ import annotations

import asyncio
import random
import time

from deepeval.models import GeminiModel

from src.config import get_settings

RETRYABLE = ("503", "429", "unavailable", "resource_exhausted", "deadline", "timeout", "internal", "overloaded")


class RetryingGeminiJudge(GeminiModel):
    max_attempts = 6

    def __init__(self, model: str | None = None):
        s = get_settings()
        super().__init__(model=model or s.judge_model, api_key=s.google_api_key)

    def _retryable(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        if "402" in msg or "credits are depleted" in msg:
            return False                       # prepaid credit exhausted: never retry (failure F-7)
        return any(k in msg for k in RETRYABLE)

    def generate(self, *args, **kwargs):
        for attempt in range(1, self.max_attempts + 1):
            try:
                return super().generate(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_attempts or not self._retryable(exc):
                    raise
                time.sleep(min(60, 3 * 2 ** (attempt - 1)) + random.random())

    async def a_generate(self, *args, **kwargs):
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await super().a_generate(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_attempts or not self._retryable(exc):
                    raise
                await asyncio.sleep(min(60, 3 * 2 ** (attempt - 1)) + random.random())

    def get_model_name(self) -> str:
        return f"{super().get_model_name()} (judge, retrying)"
