"""Tiered-memory test with VERIFIED cross-session persistence.

Session 1 and session 2 run in two separate OS processes that share only the on-disk SQLite stores:
* Tier 1 (short-term): the LangGraph SQLite checkpointer restores thread ``mem-thread-1`` in the new process, and
  within session 1 a follow-up turn uses a fact stated earlier in the same conversation.
* Tier 2 (episodic) + Tier 3 (semantic): a NEW thread in the new process recalls the prior-session case outcome
  (with its human sign-off status) and the investigator's fact via semantic search.

The full, machine-generated transcript is written to ``logs/memory_test.log`` (committed evidence).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "memory_test.log"


def _run(phase: str, state_dir: Path) -> dict:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "tests.memory_session", phase, str(state_dir)], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=600)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert p.returncode == 0 and line, f"{phase} failed:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}"
    res = json.loads(line[len("RESULT "):])
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)
    return res


def test_cross_session_memory_persistence(tmp_path):
    lines = [f"# memory persistence test - {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"# state dir (fresh): {tmp_path.name}", "# stores: checkpoints.sqlite (tier 1), long_term_memory.sqlite (tiers 2+3)"]
    checks: list[tuple[str, bool]] = []

    s1 = _run("session1", tmp_path)
    lines.append(f"\n[session1] pid={s1['pid']} elapsed={s1['elapsed_s']}s")
    lines.append(f"  turn1 investigate CLM-1003 -> intent={s1['turn1']['intent']} action={s1['turn1']['action']} "
                 f"pending_signoff={s1['turn1']['pending_signoff']}")
    lines.append(f"  human review -> {s1['review']}")
    lines.append(f"  turn2 (same thread) 'What did I tell you about the fire marshal?' -> {s1['turn2']['answer']!r}")
    checks.append(("short-term: same-thread follow-up uses earlier stated fact (accelerant)", "accelerant" in s1["turn2"]["answer"].lower()))
    checks.append(("adverse recommendation paused for human sign-off", s1["turn1"]["pending_signoff"] is True))

    s2 = _run("session2", tmp_path)
    assert s2["pid"] != s1["pid"]
    lines.append(f"\n[session2] pid={s2['pid']} (new process) elapsed={s2['elapsed_s']}s")
    lines.append(f"  restored thread mem-thread-1 from SQLite checkpointer: {s2['restored_thread_messages']} messages, "
                 f"case_file={s2['restored_case_file_claim']}")
    lines.append(f"  new thread mem-thread-2 recalled episodes={s2['turn3']['recalled_episodes']}")
    lines.append(f"  recalled semantic facts={s2['turn3']['recalled_facts']}")
    lines.append(f"  answer -> {s2['turn3']['answer']!r}")
    checks.append(("tier1: checkpointed thread restored in a new process", s2["restored_thread_messages"] >= 4
                   and s2["restored_case_file_claim"] == "CLM-1003"))
    checks.append(("tier2: prior-session episode recalled with sign-off status",
                   any(e.startswith("CLM-1003:signoff_approved") for e in s2["turn3"]["recalled_episodes"])))
    checks.append(("tier3: prior-session fact recalled by semantic search",
                   any("accelerant" in f.lower() for f in s2["turn3"]["recalled_facts"])))
    checks.append(("return-visit answer uses prior-session context", "refer_to_siu" in s2["turn3"]["answer"]
                   and "accelerant" in s2["turn3"]["answer"].lower()))

    lines.append("\n# checks")
    lines += [f"  [{'PASS' if ok else 'FAIL'}] {name}" for name, ok in checks]
    lines.append(f"\nRESULT: {'PASS' if all(ok for _, ok in checks) else 'FAIL'} ({sum(ok for _, ok in checks)}/{len(checks)})")
    LOG.parent.mkdir(exist_ok=True)
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert all(ok for _, ok in checks), "\n".join(lines)
