# Claims Fraud-Investigation Copilot (BC-AAIE-HACK-09)

An agentic copilot for an insurer's Special Investigation Unit (SIU). It takes a flagged claim, assembles the claim
history, correlates red-flag indicators against known fraud-scheme patterns, retrieves and cites the matching
playbook, and drafts an investigation case file (close / monitor / refer to SIU). Adverse recommendations stop at a
**human decision gate**. The system is built on LangGraph with Google Gemini as the only model provider (default model
`gemini-3.5-flash-lite`, set by `GEMINI_MODEL`; the committed evidence was generated with `gemini-3.5-flash`), instrumented with Arize Phoenix, governed for cost/latency, hardened with guardrails and an audit trail, and
evaluated with DeepEval (Gemini as LLM-as-judge) plus agent-level pytest suites.

All data is **synthetic** (`scripts/generate_synthetic_data.py`). Everything installs with pip; no Docker or external
database is needed.

---

## 1. Local-run runbook

### 1.1 Install (Python 3.11+; built and evidenced on 3.12, Windows 11)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm          # Presidio NLP model (PII detection)
python -m playwright install chromium            # optional: Phoenix UI screenshot
cp .env.example .env                             # then put your GOOGLE_API_KEY in .env
```

`.env` is git-ignored and takes precedence over shell variables. Without a key the copilot still runs in a
deterministic **offline mode** (every LLM node has a rule-based fallback), which the tests use.

> Windows note: `PYTHONUTF8=1` is recommended (Phoenix prints emoji); the code also forces UTF-8 streams itself.

### 1.2 Command 1: run the copilot (single command, committed sample inputs)

```bash
python -m src.cli demo
```

This runs the 14-step sample session in `data/sample_inputs/demo_session.json`:
- a full investigation with human sign-off;
- a same-session follow-up and a new-session return visit (memory);
- injection, PII-exfiltration and cross-claimant attacks;
- a deny request (escalated to a human);
- a playbook question, an ambiguous request and an out-of-scope request.

The transcript is written to `reports/demo_run.md`.

Other entry points:

```bash
python -m src.cli investigate CLM-1001 --investigator INV-07          # prompts for sign-off if required
python -m src.cli review --thread <thread_id> --decision approve --reviewer SIU-LEAD-01
python -m src.cli ask "What does the playbook say about arson for profit?"
python -m src.cli chat --investigator INV-07                          # interactive multi-turn session
python -m src.cli --offline demo                                      # no Gemini calls
uvicorn src.api.app:app --port 8000                                   # bonus: async SSE streaming API
```

### 1.3 Command 2: regenerate the Phoenix traces, the evaluation and every evidence artifact

```bash
python -m scripts.regenerate_evidence --benchmark
```

This is one traced process. It starts Phoenix in-process at http://localhost:6006, then runs:
1. the demo session;
2. the 22-case golden-set evaluation with the DeepEval Gemini judge;
3. the 12-attack red-team set;
4. the baseline-vs-optimized benchmark.

It then exports the spans and writes the golden signals, the dashboard (CSV + PNG + Phoenix UI screenshot), the failure
candidates and a manifest with the sha256 and producing script of every artifact (`reports/evidence_manifest.json`).
Use `--offline` for a no-API mechanics check.

If an evidence run stops early (e.g. the API credit runs out), `python -m scripts.finalize_offline` finishes it with
zero model calls: it re-opens the local Phoenix store, re-runs the red team offline, and rebuilds the exports and
reports.

### 1.4 Tests

```bash
python -m pytest            # offline & deterministic; also regenerates logs/memory_test.log
```

| Suite | What it proves |
|---|---|
| `tests/test_routing.py` | supervisor conditional edges route the right worker for each intent/state; real graph node order |
| `tests/test_loops.py` | max-steps loop guard, LangGraph recursion limit, bounded agentic-RAG loop |
| `tests/test_tool_contracts.py` | each MCP/RAG tool's input + output schema over the real stdio server, error paths, timeouts, log-name reconciliation |
| `tests/test_memory_persistence.py` | short + long-term memory across **two separate processes**; writes `logs/memory_test.log` |
| `tests/test_guardrails.py` | input/output guardrails, quarantine, PII masking, leak scan of all committed evidence |
| `tests/test_context.py` | write / select / compress / isolate / summarization middleware |
| `tests/test_citations_resolve.py` | every path / symbol / run_id / span_id / log line cited in the docs exists |
| `tests/test_secrets.py` | no keys committed, `.env` ignored |
| `tests/test_api.py` | bonus FastAPI SSE endpoint + review; writes `reports/api_demo_run.log` |

---

## 2. Architecture

```
START -> input_guard --(blocked)--> refuse ------------------------------------------+
           | allowed                                                                 |
         load_context   (summarisation middleware + long-term memory recall)         |
           |                                                                         v
         intent_router  (Gemini -> IntentDecision)                             output_guard
           |                                                                         |
         supervisor <----------------+                                         memory_writer
           | conditional edge        |                                               |
           +-> history_agent --------+  MCP get_claim_history, find_linked_claims    +--(refer / deny / escalate)--> request_signoff -> human_gate (interrupt) -> END
           +-> indicator_agent ------+  MCP score_red_flags + indicator catalog      +--(otherwise)----------------> END
           +-> playbook_agent -------+  agentic RAG over FAISS playbook corpus
           +-> case_writer | respond | clarify | out_of_scope | escalate_human | loop_guard --> output_guard
```

| Requirement | Implementation |
|---|---|
| LangGraph graph (`src/graph.py`) | typed state `src/state.py`; supervisor `src/agents/supervisor.py` + 3 workers `src/agents/workers.py`; conditional edges; `AsyncSqliteSaver` checkpointer; Pydantic structured output validated at every node boundary (`src/schemas.py`, `src/agents/base.py`) |
| Custom MCP server (`mcp_server/`) | FastMCP stdio server: 3 tools (`get_claim_history`, `score_red_flags`, `find_linked_claims`), 2 resources (`fraud://indicators/catalog`, `fraud://schemes/index`), 1 prompt; consumed via `langchain-mcp-adapters` in `src/tools/mcp_client.py`; transcript `logs/mcp_transcript.jsonl` |
| Context engineering (`src/context/`) | write (`write.py`), select (`select.py`), compress (`compress.py`), isolate (`isolate.py`), summarization middleware (`summarizer.py`), quarantine of untrusted claimant text (`quarantine.py`, dual-LLM pattern) |
| Tiered memory (`src/memory/`) | tier 1: SQLite checkpointer (`short_term.py`); tiers 2+3: episodic + semantic memory with LangMem extraction (`long_term.py`); verified by `tests/test_memory_persistence.py` → `logs/memory_test.log` |
| Agentic-RAG tool (`src/tools/rag_tool.py`) | FAISS + Sentence-Transformers over `data/policy_corpus/`; plan → retrieve → grade → rewrite loop with citation validation |
| Phoenix instrumentation (`src/observability/tracing.py`) | Phoenix launched in-process, OTel tracer provider + openinference LangChain instrumentor, run_id/session metadata on every span, export to `traces/` |
| Tool-invocation log | `ToolBox` middleware `src/tools/tool_logging.py` → `logs/tool_calls.jsonl` |
| Guardrails (`src/guardrails/`) | input validators (`input_guard.py`), output validators (`output_guard.py`), PII (`pii.py`, Presidio), access control (`access.py`), output-risk tiers (`output_risk.py`) |
| Audit trail | `src/audit/audit.py` → `logs/agent_actions.jsonl` |
| Cost & latency governance | `src/observability/cost.py` (per-run budget), `src/llm.py` (rate limiter, retries), `scripts/golden_signals.py`, `scripts/build_dashboard.py` |
| Evaluation | `evals/run_eval.py` (DeepEval: hallucination, faithfulness, answer relevancy, GEval correctness; judge = Gemini `evals/judge.py`) → `reports/eval_report.json` |

---

## 3. Evidence index

| Artifact | Produced by |
|---|---|
| `traces/phoenix_spans.parquet`, `traces/phoenix_spans.jsonl` | `src/observability/tracing.py::export_spans` |
| `logs/tool_calls.jsonl` | `src/tools/tool_logging.py::ToolBox` |
| `logs/agent_actions.jsonl` | `src/audit/audit.py::audit` |
| `logs/mcp_transcript.jsonl` | `src/tools/mcp_client.py::open_toolbox` + ToolBox |
| `logs/run_costs.jsonl` | `src/observability/cost.py::CostMeter` |
| `logs/memory_test.log` | `tests/test_memory_persistence.py` |
| `reports/golden_signals.json` | `scripts/golden_signals.py` |
| `reports/dashboard.png`, `reports/dashboard_data.csv`, `reports/phoenix_ui.png` | `scripts/build_dashboard.py` |
| `reports/eval_report.json` | `evals/run_eval.py` |
| `reports/redteam_results.json` | `scripts/run_redteam.py` |
| `evidence_archive/run-1/reports/optimization/` (baseline vs optimized golden signals) | `scripts/optimization_benchmark.py` |
| `evidence_archive/run-1/` (complete first live run, cited by the failure analysis) | `scripts/regenerate_evidence.py` (run 1) |
| `reports/demo_run.md` | `src/cli.py::run_demo` |
| `reports/pii_redaction_sample.json` | `scripts/pii_redaction_sample.py` |
| `reports/failure_candidates.json` | `scripts/find_failures.py` |
| `reports/api_demo_run.log` | `tests/test_api.py` |
| `reports/evidence_manifest.json` | `scripts/regenerate_evidence.py::manifest` |

Governance pack: `docs/risk-register.md`, `docs/model-card.md`, `docs/compliance.md`, `docs/output-risk.md`.
Failure analysis: `docs/failure-analysis.md`. Optimisation note: `docs/optimization-note.md`.

## 4. Results (committed evidence)

The final evidence run used the code in this repo with `gemini-3.5-flash`:
- The demo session and the 22-case golden-set evaluation ran **live**.
- The API's prepaid credit ran out during the red-team stage (HTTP 402, failure F-7). Those calls degraded gracefully,
  and the run was completed offline with `python -m scripts.finalize_offline`, which re-ran the red team offline
  and exported all spans with **zero** model calls.
- Run 1, the first complete live run including the live red team and the before/after benchmark, is preserved in
  `evidence_archive/run-1/` and cited by the failure analysis.

| Area | Result | Evidence |
|---|---|---|
| Golden-set task accuracy (22 cases) | **100 %** (intent, tier, action, scheme + citation, sign-off gate, injection quarantine, safety) | `reports/eval_report.json` |
| LLM-as-judge (DeepEval, Gemini) | hallucination rate **0 %**, grounding 1.0, faithfulness 0.994, relevancy 1.0, case correctness 1.0 | `reports/eval_report.json` |
| Red team (12 attacks) | **12/12 defended** (final run, offline); 12/12 live in run 1 | `reports/redteam_results.json`, `evidence_archive/run-1/reports/redteam_results.json` |
| Golden signals (Phoenix) | turn latency p50 2.5 s / p95 29.9 s; LLM p50 2.4 s; tool p95 101 ms; error-span rate 2.4 % (all 402s); all SLOs met | `reports/golden_signals.json`, `reports/dashboard.png`, `reports/phoenix_ui.png` |
| Optimisation | thinking_level low + compression: latency −50.5 %, reasoning tokens −89 %, est. cost −64 %, same decisions | `docs/optimization-note.md` |
| Memory | short-term + episodic + semantic recall across two separate processes (6/6) | `logs/memory_test.log` |
| Failure analysis | 7 real failures with Phoenix run_id/span_id or log-line citations, root cause, fix and verification | `docs/failure-analysis.md` |

Cost figures use **placeholder** prices (`PRICE_*_PER_1M` in `.env.example`). Actual Gemini billing was higher, so check
current pricing, and remember that the DeepEval judge makes its own calls, before running
`scripts.regenerate_evidence` on a paid key. Use `COPILOT_FORCE_OFFLINE=1` or `--offline` for zero-cost runs.

