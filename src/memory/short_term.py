"""Tier 1 -- short-term / working memory.

The LangGraph state of each conversation thread (messages, running summary, case file, pending human review)
is checkpointed after every node by ``langgraph-checkpoint-sqlite`` into a local SQLite file. Re-opening the
same ``thread_id`` -- even from a new process -- restores the conversation, which is how the copilot "uses facts
stated earlier in the interaction" (AC-05) and how a paused human-sign-off can be resumed later.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def open_checkpointer(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver
