"""Command-line interface for the Claims Fraud-Investigation Copilot.

    python -m src.cli demo                                   # run the committed sample session (single command)
    python -m src.cli investigate CLM-1001 --investigator INV-07
    python -m src.cli ask "What is the playbook for staged collisions?"
    python -m src.cli review --thread <thread_id> --decision approve --reviewer SIU-LEAD-01
    python -m src.cli chat --investigator INV-07             # interactive multi-turn session

Global flags: --offline (deterministic fallbacks, no Gemini calls), --no-trace (skip Phoenix), --export-traces,
--keep-open (keep the Phoenix UI at http://localhost:6006 running after the command, until Enter is pressed).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from src.config import DATA_DIR, REPORTS_DIR, get_settings

SAMPLE_SESSION = DATA_DIR / "sample_inputs" / "demo_session.json"


def _print_result(r: dict, *, verbose: bool = False) -> None:
    print(f"\n[run {r['run_id']} | thread {r['thread_id']} | intent {r['intent']} | "
          f"risk {(r.get('output_check') or {}).get('risk_tier')} | cost ${r['cost']['cost_usd']:.4f}"
          f"{' | DEGRADED' if r['degraded'] else ''}]")
    print(r["answer"] or "(no answer)")
    if r["pending_signoff"]:
        i = r["interrupt"] or {}
        print(f"\n>>> HUMAN SIGN-OFF REQUIRED for {i.get('claim_id')}: requested action '{i.get('requested_action')}'. "
              f"Resume with: python -m src.cli review --thread {r['thread_id']} --decision approve|reject --reviewer SIU-LEAD-01")
    if verbose and r.get("output_check", {}).get("violations"):
        print("guardrail repairs:", r["output_check"]["violations"])


async def _session(args):
    from src.runner import Copilot

    return Copilot(offline=True if args.offline else None)


def _setup(args) -> None:
    if args.no_trace:
        os.environ["TRACING_ENABLED"] = "false"
    else:
        from src.observability.tracing import init_tracing

        url = init_tracing()
        if url:
            print(f"Phoenix tracing on: {url}")


def _maybe_export(args) -> None:
    if getattr(args, "export_traces", False) and not args.no_trace:
        from src.observability.tracing import export_spans

        print("exported:", export_spans())


async def cmd_investigate(args) -> None:
    async with await _session(args) as cp:
        req = f"Investigate flagged claim {args.claim_id} and draft the investigation case file."
        if args.note:
            req += f" Investigator note: {args.note}"
        r = await cp.ask(req, thread_id=args.thread, actor_id=args.investigator,
                         attachments=[args.attach] if args.attach else None)
        _print_result(r, verbose=True)
        if r["pending_signoff"] and (args.decision or sys.stdin.isatty()):
            decision = args.decision or input("Sign-off decision [approve/reject/skip]: ").strip().lower()
            if decision in ("approve", "reject"):
                reviewer = args.reviewer or input("Reviewer id: ").strip()
                _print_result(await cp.review(r["thread_id"], decision=decision, reviewer=reviewer, note=args.review_note or ""))
    _maybe_export(args)


async def cmd_ask(args) -> None:
    async with await _session(args) as cp:
        _print_result(await cp.ask(args.text, thread_id=args.thread, actor_id=args.investigator,
                                   actor_role=args.role), verbose=True)
    _maybe_export(args)


async def cmd_review(args) -> None:
    async with await _session(args) as cp:
        _print_result(await cp.review(args.thread, decision=args.decision, reviewer=args.reviewer, note=args.note or ""))
    _maybe_export(args)


async def cmd_chat(args) -> None:
    import uuid

    thread = args.thread or f"chat-{uuid.uuid4().hex[:8]}"
    async with await _session(args) as cp:
        print(f"SIU copilot chat (thread {thread}). Type 'exit' to quit.")
        while True:
            try:
                text = input("\ninvestigator> ").strip()
            except EOFError:
                break
            if text.lower() in ("exit", "quit"):
                break
            if text:
                r = await cp.ask(text, thread_id=thread, actor_id=args.investigator)
                _print_result(r)
                if r["pending_signoff"]:
                    d = input("sign-off [approve/reject/skip]> ").strip().lower()
                    if d in ("approve", "reject"):
                        _print_result(await cp.review(thread, decision=d, reviewer=input("reviewer> ").strip()))
    _maybe_export(args)


async def run_demo(offline: bool = False, session_file: Path = SAMPLE_SESSION, transcript: Path | None = None) -> list[dict]:
    """Run the committed sample session. Used by `demo` and by scripts/regenerate_evidence.py."""
    from src.runner import Copilot

    spec = json.loads(Path(session_file).read_text("utf-8"))
    results, lines = [], [f"# Demo run transcript\n\nGenerated by `python -m src.cli demo` from `{Path(session_file).as_posix().split('claims-fraud-investigation-copilot-app/')[-1]}`.\n"]
    async with Copilot(offline=True if offline else None) as cp:
        for i, step in enumerate(spec["steps"], 1):
            if "review" in step:
                rv = step["review"]
                r = await cp.review(rv["thread"], decision=rv["decision"], reviewer=rv["reviewer"], note=rv.get("note", ""),
                                    tags=["demo"])
                title = f"Step {i} - human review by {rv['reviewer']} ({rv['decision']})"
            else:
                r = await cp.ask(step["request"], thread_id=step["thread"], actor_id=step.get("actor", "INV-07"),
                                 actor_role=step.get("role", "investigator"), attachments=step.get("attachments"),
                                 tags=["demo"])
                title = f"Step {i} - {step.get('label', '')}"
            _print_result(r)
            results.append({"step": i, **{k: r[k] for k in ("run_id", "thread_id", "intent", "pending_signoff", "degraded")},
                            "risk_tier": (r.get("output_check") or {}).get("risk_tier"), "cost_usd": r["cost"]["cost_usd"]})
            lines += [f"\n## {title}\n", f"- run_id: `{r['run_id']}`  thread: `{r['thread_id']}`  intent: `{r['intent']}`  "
                      f"risk: `{(r.get('output_check') or {}).get('risk_tier')}`  pending sign-off: `{r['pending_signoff']}`\n"]
            if "request" in step:
                lines.append(f"**Request ({step.get('actor', 'INV-07')}):** {step['request']}\n")
            lines.append("**Copilot:**\n\n" + (r["answer"] or "") + "\n")
    if transcript:
        transcript.write_text("\n".join(lines), encoding="utf-8")
    return results


async def cmd_demo(args) -> None:
    res = await run_demo(offline=args.offline, transcript=REPORTS_DIR / "demo_run.md")
    (REPORTS_DIR / "demo_run.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDemo complete: {len(res)} steps. Transcript: reports/demo_run.md")
    _maybe_export(args)


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true", help="no Gemini calls; deterministic fallbacks")
    ap.add_argument("--no-trace", action="store_true", help="disable Phoenix tracing")
    ap.add_argument("--export-traces", action="store_true", help="export Phoenix spans to traces/ at the end")
    ap.add_argument("--keep-open", action="store_true",
                    help="keep the Phoenix UI running after the command finishes (press Enter to stop)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("investigate", help="build a case file for a flagged claim")
    p.add_argument("claim_id")
    p.add_argument("--investigator", default="INV-07")
    p.add_argument("--thread")
    p.add_argument("--note", help="trusted investigator note")
    p.add_argument("--attach", help="untrusted claimant-supplied text (quarantined)")
    p.add_argument("--decision", choices=["approve", "reject"])
    p.add_argument("--reviewer")
    p.add_argument("--review-note")
    p.set_defaults(fn=cmd_investigate)

    p = sub.add_parser("ask", help="free-text request")
    p.add_argument("text")
    p.add_argument("--investigator", default="INV-07")
    p.add_argument("--role", default="investigator", choices=["investigator", "claimant", "reviewer"])
    p.add_argument("--thread")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("review", help="resume a paused run with a human sign-off decision")
    p.add_argument("--thread", required=True)
    p.add_argument("--decision", required=True, choices=["approve", "reject"])
    p.add_argument("--reviewer", required=True)
    p.add_argument("--note")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("chat", help="interactive multi-turn session")
    p.add_argument("--investigator", default="INV-07")
    p.add_argument("--thread")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("demo", help="run the committed sample session end-to-end")
    p.set_defaults(fn=cmd_demo)

    args = ap.parse_args(argv)
    _setup(args)
    get_settings()
    asyncio.run(args.fn(args))
    if args.keep_open and not args.no_trace:
        # Phoenix runs inside this process: keep it alive so the traces can be browsed after the run
        from src.observability.tracing import flush

        flush()
        project = get_settings().phoenix_project
        print(f"\nPhoenix is still running: http://localhost:{get_settings().phoenix_port}  "
              f"(open the '{project}' project). Press Enter to stop.")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
