"""Agentic-RAG tool over the synthetic fraud-scheme playbook corpus (``data/policy_corpus/``).

Retrieval stack: Sentence-Transformers ``all-MiniLM-L6-v2`` embeddings + FAISS ``IndexFlatIP`` (cosine on
normalised vectors), persisted under ``data/index/`` and rebuilt automatically when the corpus changes.

Two layers:
* ``search_fraud_playbook`` -- a LangChain tool (typed input/output) executed through the ToolBox middleware.
* ``agentic_playbook_retrieval`` -- retrieval-in-the-loop: the agent (Gemini) PLANS a query from the correlated
  indicators, RETRIEVES, GRADES the evidence, and REWRITES the query and retrieves again when the evidence is
  insufficient (bounded by ``MAX_RETRIEVAL_ROUNDS``). Citations are validated against the retrieved chunks so the
  case file can only cite text that was actually retrieved.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from typing import Any, Optional

import numpy as np
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.config import CORPUS_DIR, INDEX_DIR

log = logging.getLogger(__name__)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOOL_NAME = "search_fraud_playbook"


# ----------------------------------------------------------------------------------------------- corpus + index
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def load_chunks() -> list[dict]:
    """Split every playbook markdown file into one chunk per ``## Section``."""
    chunks = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        raw = path.read_text("utf-8")
        _, front, body = raw.split("---", 2)
        meta = dict((k.strip(), v.strip()) for k, v in (l.split(":", 1) for l in front.strip().splitlines() if ":" in l))
        for sec in re.split(r"\n## ", body)[1:]:
            heading, _, text = sec.partition("\n")
            chunks.append({
                "chunk_id": f"{meta['doc_id']}#{_slug(heading)}",
                "doc_id": meta["doc_id"],
                "title": meta["title"],
                "section": heading.strip(),
                "line_of_business": meta.get("line_of_business", ""),
                "text": text.strip(),
            })
    return chunks


def _corpus_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(CORPUS_DIR.glob("*.md")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    return np.asarray(_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")


class PlaybookIndex:
    def __init__(self) -> None:
        import faiss

        self.chunks = load_chunks()
        digest = _corpus_hash()
        idx_path, meta_path = INDEX_DIR / "playbook.faiss", INDEX_DIR / "playbook_meta.json"
        if idx_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get("hash") == digest:
            self.index = faiss.read_index(str(idx_path))
        else:
            vecs = embed([f"{c['title']} - {c['section']}\n{c['text']}" for c in self.chunks])
            self.index = faiss.IndexFlatIP(vecs.shape[1])
            self.index.add(vecs)
            faiss.write_index(self.index, str(idx_path))
            meta_path.write_text(json.dumps({"hash": digest, "chunks": len(self.chunks), "model": EMBED_MODEL}))

    def search(self, query: str, k: int = 4, doc_filter: Optional[str] = None) -> list[dict]:
        q = embed([query])
        scores, ids = self.index.search(q, min(len(self.chunks), k * 4 if doc_filter else k))
        out = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            c = self.chunks[int(i)]
            if doc_filter and c["doc_id"] != doc_filter:
                continue
            out.append({**c, "score": round(float(s), 4)})
            if len(out) >= k:
                break
        return out


@lru_cache(maxsize=1)
def get_index() -> PlaybookIndex:
    return PlaybookIndex()


# ----------------------------------------------------------------------------------------------- the tool
class PlaybookSearchInput(BaseModel):
    query: str = Field(min_length=3, max_length=500, description="Natural-language search over fraud-scheme playbooks")
    k: int = Field(4, ge=1, le=8, description="Number of chunks to return")
    doc_filter: Optional[str] = Field(None, pattern=r"^(FS|GL)-\d{2}$", description="Restrict to one playbook doc id")


class PlaybookHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    score: float
    text: str


class PlaybookSearchOutput(BaseModel):
    query: str
    results: list[PlaybookHit]


async def _search(query: str, k: int = 4, doc_filter: Optional[str] = None) -> str:
    PlaybookSearchInput(query=query, k=k, doc_filter=doc_filter)  # explicit validation -> error path
    hits = get_index().search(query, k=k, doc_filter=doc_filter)
    return PlaybookSearchOutput(query=query, results=[PlaybookHit(**h) for h in hits]).model_dump_json()


search_fraud_playbook = StructuredTool.from_function(
    coroutine=_search,
    name=TOOL_NAME,
    description="Semantic search over the synthetic fraud-scheme playbook corpus (FAISS). Returns cited chunks.",
    args_schema=PlaybookSearchInput,
)


# ----------------------------------------------------------------------------------------------- agentic loop
class RetrievalPlan(BaseModel):
    query: str = Field(description="Search query for the playbook index, <= 30 words")
    rationale: str


class RetrievalJudgement(BaseModel):
    sufficient: bool = Field(description="True if the retrieved chunks are enough to identify (or rule out) a scheme")
    rewritten_query: Optional[str] = Field(None, description="Better query if not sufficient")
    matched: bool = Field(description="True only if the claim's indicators satisfy a scheme's correlation pattern")
    scheme_id: Optional[str] = None
    scheme_name: Optional[str] = None
    match_confidence: float = Field(ge=0, le=1)
    matched_indicators: list[str] = Field(default_factory=list, description="RF codes present in BOTH claim and scheme")
    cited_chunk_ids: list[str] = Field(default_factory=list, description="chunk_id values from the retrieved list only")
    recommended_steps: list[str] = Field(default_factory=list)


def _fallback_query(context: dict) -> str:
    names = " ".join(i.get("name", "").lower().replace("_", " ") for i in context.get("indicators", [])[:5])
    return f"{context.get('loss_type', '').replace('_', ' ')} {context.get('line_of_business', '')} fraud scheme {names}".strip()


def fallback_judgement(context: dict, hits: list[dict]) -> RetrievalJudgement:
    """Deterministic scheme selection when the LLM is unavailable: indicator-code overlap x retrieval score."""
    codes = {i["code"] for i in context.get("indicators", []) if i.get("code") not in ("RF-15", "RF-16")}
    tier = context.get("tier", "low")
    best, best_score, best_overlap = None, 0.0, []
    for h in hits:
        if not h["doc_id"].startswith("FS-"):
            continue
        doc_codes = set(re.findall(r"RF-\d{2}", " ".join(c["text"] for c in get_index().chunks if c["doc_id"] == h["doc_id"])))
        overlap = sorted(codes & doc_codes)
        s = len(overlap) + h["score"]
        if s > best_score:
            best, best_score, best_overlap = h, s, overlap
    matched = bool(best) and tier != "low" and len(best_overlap) >= 2 or (bool(best) and "RF-13" in best_overlap)
    return RetrievalJudgement(
        sufficient=True, matched=matched, scheme_id=best["doc_id"] if matched else None,
        scheme_name=best["title"] if matched else None, match_confidence=min(0.9, 0.3 + 0.15 * len(best_overlap)) if matched else 0.2,
        matched_indicators=best_overlap if matched else [], cited_chunk_ids=[best["chunk_id"]] if best else [],
        recommended_steps=[])


def quote_of(text: str) -> str:
    """First substantive line of a chunk, verbatim (list markers / numbering stripped), max 220 chars."""
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line).strip()
        if len(line) >= 25:
            return line[:220]
    return text.strip()[:220]


def validate_citations(judgement: RetrievalJudgement, retrieved: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Keep only citations that resolve to retrieved chunks; quote verbatim from the chunk (anti-hallucination)."""
    cits, dropped = [], []
    for cid in judgement.cited_chunk_ids:
        c = retrieved.get(cid)
        if not c:
            dropped.append(cid)
            continue
        first = quote_of(c["text"])
        cits.append({"doc_id": c["doc_id"], "section": c["section"], "chunk_id": cid, "quote": first})
    if judgement.matched and judgement.scheme_id and not any(c["doc_id"] == judgement.scheme_id for c in cits):
        for c in retrieved.values():
            if c["doc_id"] == judgement.scheme_id:
                first = quote_of(c["text"])
                cits.append({"doc_id": c["doc_id"], "section": c["section"], "chunk_id": c["chunk_id"], "quote": first})
                break
    return cits, dropped


async def agentic_playbook_retrieval(llm, toolbox, context: dict, *, agent: str, max_rounds: int,
                                     question: Optional[str] = None) -> dict[str, Any]:
    """Plan -> retrieve -> grade -> (rewrite -> retrieve)* -> validated SchemeMatch dict."""
    from langchain_core.messages import HumanMessage, SystemMessage

    ctx_json = json.dumps(context, default=str)[:3500]
    task = question or "Identify which fraud scheme (if any) this flagged claim matches."
    plan, _ = await llm.structured(RetrievalPlan, [
        SystemMessage("You plan searches over an insurer's fraud-scheme playbook index. Write one focused query."),
        HumanMessage(f"TASK: {task}\nCLAIM CONTEXT (tool-derived, trusted):\n{ctx_json}"),
    ], agent=agent, fallback=lambda: RetrievalPlan(query=question or _fallback_query(context), rationale="deterministic fallback"))

    query, rounds, retrieved, judgement = plan.query, 0, {}, None
    queries = []
    while rounds < max_rounds:
        rounds += 1
        queries.append(query)
        res = await toolbox.call(TOOL_NAME, {"query": query, "k": 4}, agent=agent)
        hits = res.data.get("results", []) if res.ok and isinstance(res.data, dict) else []
        for h in hits:
            retrieved.setdefault(h["chunk_id"], h)
        listing = "\n\n".join(f"[{h['chunk_id']}] ({h['title']} / {h['section']}, score={h['score']})\n{h['text']}"
                              for h in retrieved.values())
        judgement, _ = await llm.structured(RetrievalJudgement, [
            SystemMessage(
                "You are the fraud-playbook retrieval agent. Judge whether the RETRIEVED CHUNKS are sufficient to decide "
                "which scheme the claim matches. A scheme matches only when the claim's fired indicator codes satisfy that "
                "scheme's Correlation Pattern. Claims with low scores / no strong indicators must be matched=false. "
                "Cite ONLY chunk_ids that appear in the list. If insufficient, propose rewritten_query."),
            HumanMessage(f"TASK: {task}\nCLAIM CONTEXT:\n{ctx_json}\n\nRETRIEVED CHUNKS (round {rounds}):\n{listing}"),
        ], agent=agent, fallback=lambda: fallback_judgement(context, list(retrieved.values())))
        if judgement.sufficient or not judgement.rewritten_query or rounds >= max_rounds:
            break
        query = judgement.rewritten_query

    citations, dropped = validate_citations(judgement, retrieved)
    return {
        "scheme_match": {
            "matched": judgement.matched, "scheme_id": judgement.scheme_id if judgement.matched else None,
            "scheme_name": judgement.scheme_name if judgement.matched else None,
            "match_confidence": judgement.match_confidence, "matched_indicators": judgement.matched_indicators,
            "citations": citations, "recommended_steps": judgement.recommended_steps[:6], "retrieval_rounds": rounds,
        },
        "retrieved_chunks": [{k: v for k, v in c.items()} for c in retrieved.values()],
        "queries": queries,
        "dropped_citations": dropped,
    }
