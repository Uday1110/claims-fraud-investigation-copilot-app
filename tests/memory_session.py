"""Helper executed as a SEPARATE PROCESS by tests/test_memory_persistence.py.

    python -m tests.memory_session session1 <state_dir>
    python -m tests.memory_session session2 <state_dir>

Each invocation is an independent OS process (new Python interpreter, new MCP server, new SQLite connections),
so anything session2 knows about session1 must have been persisted to disk.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


async def main(phase: str, state_dir: Path) -> dict:
    os.environ["CHECKPOINT_DB"] = str(state_dir / "checkpoints.sqlite")
    os.environ["MEMORY_DB"] = str(state_dir / "long_term_memory.sqlite")
    os.environ["EVIDENCE_LOGS_DIR"] = str(state_dir / "logs")
    os.environ["TRACING_ENABLED"] = "false"
    from src.runner import Copilot

    out: dict = {"phase": phase, "pid": os.getpid()}
    async with Copilot(offline=True) as cp:
        if phase == "session1":
            r1 = await cp.ask("Investigate CLM-1003. FYI the fire marshal confirmed accelerant traces in the garage.",
                              thread_id="mem-thread-1", actor_id="INV-07")
            out["turn1"] = {"intent": r1["intent"], "action": (r1["case_file"] or {}).get("recommended_action"),
                            "pending_signoff": r1["pending_signoff"]}
            r1b = await cp.review("mem-thread-1", decision="approve", reviewer="SIU-LEAD-01", note="approved")
            out["review"] = (r1b["human_review"] or {}).get("status")
            r2 = await cp.ask("What did I tell you about the fire marshal?", thread_id="mem-thread-1", actor_id="INV-07")
            out["turn2"] = {"intent": r2["intent"], "answer": r2["answer"]}
        else:
            snap = await cp.graph.aget_state({"configurable": {"thread_id": "mem-thread-1"}})
            out["restored_thread_messages"] = len(snap.values.get("messages", []))
            out["restored_case_file_claim"] = (snap.values.get("case_file") or {}).get("claim_id")
            r3 = await cp.ask("I'm back - where did we land on CLM-1003 last time, and what did the fire marshal find?",
                              thread_id="mem-thread-2", actor_id="INV-07")
            out["turn3"] = {"intent": r3["intent"], "answer": r3["answer"],
                            "recalled_episodes": [e["claim_id"] + ":" + e["status"] for e in (r3["memory_context"] or {}).get("episodes", [])],
                            "recalled_facts": [f["fact"] for f in (r3["memory_context"] or {}).get("facts", [])]}
    return out


if __name__ == "__main__":
    print("RESULT " + json.dumps(asyncio.run(main(sys.argv[1], Path(sys.argv[2])))))
