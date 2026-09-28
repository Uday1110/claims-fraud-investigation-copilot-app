"""Tier 2 (episodic) and Tier 3 (semantic) long-term memory, persisted across sessions in SQLite.

* Episodic memory -- one record per investigated claim per investigator: recommendation, likelihood, scheme,
  human-review status, run/thread ids. Upserted by the ``memory_writer`` and ``human_gate`` nodes.
* Semantic memory -- durable facts extracted from investigator turns by a **LangMem** memory manager
  (``langmem.create_memory_manager`` running on Gemini), embedded with Sentence-Transformers and recalled by
  cosine similarity. A deterministic extractor is used when Gemini is unavailable.

Recall happens in the ``load_context`` node at the start of every turn, so a *new* thread (a return visit in a
later session) still sees what happened before (AC-05). Everything stored is PII-masked first.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from src.guardrails.pii import mask_obj, mask_text
from src.observability.run_context import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    actor_id TEXT NOT NULL, claim_id TEXT NOT NULL, thread_id TEXT, run_id TEXT, updated_at TEXT,
    status TEXT, payload TEXT, PRIMARY KEY (actor_id, claim_id));
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, claim_id TEXT, content TEXT NOT NULL, embedding BLOB,
    source TEXT, created_at TEXT);
"""


class LongTermMemory:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:          # commit / rollback
                yield conn
        finally:
            conn.close()        # never hold the file open between calls (cross-process safety)

    # ------------------------------------------------------------------ episodic
    def remember_episode(self, actor_id: str, claim_id: str, payload: dict, *, status: str,
                         thread_id: str = "", run_id: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT INTO episodes VALUES (?,?,?,?,?,?,?) ON CONFLICT(actor_id, claim_id) DO UPDATE SET "
                      "thread_id=excluded.thread_id, run_id=excluded.run_id, updated_at=excluded.updated_at, "
                      "status=excluded.status, payload=excluded.payload",
                      (actor_id, claim_id, thread_id, run_id, now_iso(), status, json.dumps(mask_obj(payload))))

    def update_episode_status(self, actor_id: str, claim_id: str, status: str, extra: Optional[dict] = None) -> None:
        with self._conn() as c:
            row = c.execute("SELECT payload FROM episodes WHERE actor_id=? AND claim_id=?", (actor_id, claim_id)).fetchone()
            payload = json.loads(row[0]) if row else {}
            payload.update(mask_obj(extra or {}))
            c.execute("UPDATE episodes SET status=?, payload=?, updated_at=? WHERE actor_id=? AND claim_id=?",
                      (status, json.dumps(payload), now_iso(), actor_id, claim_id))

    def episodes(self, actor_id: str, claim_id: Optional[str] = None, limit: int = 5) -> list[dict]:
        q = "SELECT claim_id, status, payload, updated_at, thread_id, run_id FROM episodes WHERE actor_id=?"
        args: list[Any] = [actor_id]
        if claim_id:
            q += " AND claim_id=?"
            args.append(claim_id)
        q += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [{"claim_id": r[0], "status": r[1], **json.loads(r[2]), "updated_at": r[3], "thread_id": r[4],
                     "run_id": r[5]} for r in c.execute(q, args).fetchall()]

    # ------------------------------------------------------------------ semantic
    def remember_facts(self, actor_id: str, facts: list[str], *, claim_id: Optional[str], source: str) -> int:
        facts = [mask_text(f.strip()) for f in facts if f and f.strip()]
        if not facts:
            return 0
        from src.tools.rag_tool import embed

        vecs = embed(facts)
        with self._conn() as c:
            existing = {r[0] for r in c.execute("SELECT content FROM facts WHERE actor_id=?", (actor_id,))}
            n = 0
            for f, v in zip(facts, vecs):
                if f in existing:
                    continue
                c.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,?)",
                          (uuid.uuid4().hex, actor_id, claim_id, f, v.astype("float32").tobytes(), source, now_iso()))
                n += 1
        return n

    def search_facts(self, actor_id: str, query: str, k: int = 4, claim_id: Optional[str] = None) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT content, embedding, claim_id, created_at FROM facts WHERE actor_id=?", (actor_id,)).fetchall()
        if not rows:
            return []
        from src.tools.rag_tool import embed

        q = embed([query])[0]
        mat = np.stack([np.frombuffer(r[1], dtype="float32") for r in rows])
        sims = mat @ q
        boost = np.array([0.15 if claim_id and r[2] == claim_id else 0.0 for r in rows])
        order = np.argsort(-(sims + boost))[:k]
        return [{"fact": rows[i][0], "claim_id": rows[i][2], "score": round(float(sims[i]), 3), "created_at": rows[i][3]}
                for i in order]

    def recall(self, actor_id: str, query: str, claim_id: Optional[str] = None) -> dict:
        eps = self.episodes(actor_id, claim_id) or self.episodes(actor_id)
        return {"episodes": eps, "facts": self.search_facts(actor_id, query or claim_id or "case", claim_id=claim_id)}


# ---------------------------------------------------------------------- LangMem extraction
MEMORY_INSTRUCTIONS = (
    "Extract durable facts an SIU investigator stated that will matter in later sessions (e.g. new evidence they "
    "learned, decisions, preferences, deadlines). One short sentence per fact, include the claim id. Do NOT store "
    "claimant free text, names or unmasked identifiers. Ignore greetings and questions."
)
_FACTY = re.compile(r"\b(told me|confirmed|learned|found|note|fyi|heads up|informed|resigned|sold|reported|remember)\b", re.I)


def fallback_extract(turns: list[str], claim_id: Optional[str]) -> list[str]:
    out = []
    for t in turns:
        for s in re.split(r"(?<=[.!?])\s+", t):
            if _FACTY.search(s) and len(s) > 20:
                out.append(f"[{claim_id}] {s.strip()}" if claim_id and claim_id not in s else s.strip())
    return out[:5]


async def extract_facts(llm, investigator_turns: list[str], claim_id: Optional[str]) -> tuple[list[str], str]:
    """Run a LangMem memory manager over the investigator's turns. Returns (facts, method)."""
    if not investigator_turns:
        return [], "none"
    if getattr(llm, "offline", True):
        return fallback_extract(investigator_turns, claim_id), "deterministic"
    try:
        import asyncio

        from langmem import create_memory_manager

        manager = create_memory_manager(llm.chat, instructions=MEMORY_INSTRUCTIONS, enable_inserts=True,
                                        enable_updates=False, enable_deletes=False)
        msgs = [{"role": "user", "content": f"(claim {claim_id}) {t}"} for t in investigator_turns]
        res = await asyncio.wait_for(manager.ainvoke({"messages": msgs}), timeout=llm.settings.llm_timeout_s)
        facts = []
        for m in res or []:
            content = getattr(m, "content", m)
            content = getattr(content, "content", content)
            if isinstance(content, dict):
                content = content.get("content", json.dumps(content))
            facts.append(str(content))
        return facts[:6], "langmem"
    except Exception:  # noqa: BLE001 -- degrade to deterministic extraction
        return fallback_extract(investigator_turns, claim_id), "deterministic_fallback"
