"""Finish an evidence run WITHOUT any Gemini calls (used when API credit ran out mid-run; failure F-7).

The final `regenerate_evidence --benchmark` run completed its demo session and golden-set evaluation live on
gemini-3.5-flash, then the prepaid credit was exhausted during the red-team stage and the process was stopped before
its export step. Its spans are still in the local Phoenix store (data/runtime/phoenix). This script:

  1. forces offline mode (COPILOT_FORCE_OFFLINE=1: the API key is ignored, zero model calls)
  2. re-opens that Phoenix store in-process (no wipe) and re-instruments LangChain
  3. re-runs the red-team set offline (tag `redteam-offline`; guardrail defences are deterministic)
  4. exports ALL spans -> traces/, then rebuilds golden signals, dashboard, screenshot, failure candidates,
     PII sample and the evidence manifest

Usage:  python -m scripts.finalize_offline
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

os.environ["COPILOT_FORCE_OFFLINE"] = "1"

from src.config import REPORTS_DIR, get_settings  # noqa: E402


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    assert get_settings().google_api_key is None, "offline guard failed"
    from src.observability.tracing import export_spans, init_tracing

    base_url = init_tracing(fresh=False)
    print("Phoenix (existing store):", base_url)

    from scripts.run_redteam import run as run_redteam

    rt = asyncio.run(run_redteam(3, tag="redteam-offline"))
    print(f"red team (offline) defended {rt['defended']}/{rt['attacks']}")

    time.sleep(3)
    print(export_spans())

    from scripts.golden_signals import build

    gs = build()
    (REPORTS_DIR / "golden_signals.json").write_text(json.dumps(gs, indent=2), encoding="utf-8")
    print(json.dumps(gs["slo"], indent=2))

    from scripts.build_dashboard import draw, export_csv, phoenix_ui_screenshot

    export_csv()
    draw()
    try:
        print("screenshot:", phoenix_ui_screenshot(base_url, get_settings().phoenix_project))
    except Exception as exc:  # noqa: BLE001
        print("phoenix screenshot skipped:", exc)

    from scripts import find_failures, pii_redaction_sample
    from scripts.regenerate_evidence import manifest

    find_failures.main()
    pii_redaction_sample.main()
    manifest()
    print("finalised offline.")


if __name__ == "__main__":
    main()
