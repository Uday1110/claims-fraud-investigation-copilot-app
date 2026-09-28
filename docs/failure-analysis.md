# Failure-Mode Analysis (AC-08)

Seven **real** failures found in our own runs, each with the Phoenix `run_id` + `span_id` (or log line) that shows it,
the root cause, and the fix. Where they came from:
- **Run 1:** the first full evidence run on `gemini-3.5-flash` (2026-09-26 17:41–17:55 UTC). It is preserved unchanged
  in `evidence_archive/run-1/`: Phoenix export `evidence_archive/run-1/traces/phoenix_spans.jsonl` (+ `.parquet`),
  logs and reports.
- **Final run:** after the fixes, `python -m scripts.regenerate_evidence --benchmark` produced the current top-level
  `traces/`, `logs/` and `reports/`.

The last section shows each fix verified in the final run.

How the failures were found: `python -m scripts.find_failures` (error spans, tool failures, audit anomalies, failed
eval checks → `reports/failure_candidates.json`) plus reading the Phoenix traces and golden signals. Every citation
below is checked by `tests/test_citations_resolve.py`.

| ID | Failure | Layer | Severity | Status |
|---|---|---|---|---|
| F-1 | 45 s cold-start latency on the first request | performance | High | Fixed |
| F-2 | Human-gate routing audited twice on resume | audit / governance | Medium | Fixed |
| F-3 | Hallucination rate reported as 100 % (metric polarity) | evaluation | High | Fixed |
| F-4 | False-negative judge scores on G15 (criteria + truncated context) | evaluation | Medium | Fixed |
| F-5 | Model mislabelled indicator RF-14 in a playbook answer | agent / grounding | Medium | Fixed |
| F-6 | LLM tokens and cost not attributable to agents | observability | Low | Fixed |
| F-7 | Prepaid Gemini credit exhausted mid-run (HTTP 402) | resilience / cost | High | Handled + hardened |

---

## Failure F-1: 45-second cold start on the first request

**Evidence (run 1).** The first turn of the demo session, run_id `4d80710094434829`:
- Its `input_guard` node took **44,994.7 ms**: span_id `5f2d4529f669f2c2` (run_id `4d80710094434829`).
- Its first `search_fraud_playbook` tool call took **9,810.1 ms**: span_id `4d625a5b8582b1d5` (run_id `4d80710094434829`),
  and tool-log record `evidence_archive/run-1/logs/tool_calls.jsonl#L4` (run_id `4d80710094434829`).
- Every later `input_guard` ran in about 75 ms (p50), and the tool p50 was 10.9 ms.

The first investigator of the day would wait almost a minute, and the outlier inflated the run p95 and max
(85.6 s) in `evidence_archive/run-1/reports/golden_signals.json`.

**Root cause.** Heavy local models were loaded lazily *inside the request path*:
- The Presidio `AnalyzerEngine`, with its spaCy pipeline, loads on the first `redact_free_text` call made by the input
  guardrail (`src/guardrails/pii.py::_presidio`).
- The Sentence-Transformers model and FAISS index load on the first playbook search
  (`src/tools/rag_tool.py::get_index`).

**Fix.** `Copilot.__aenter__` now calls `src/runner.py::Copilot.warm_up` in a worker thread before serving any
request. The warm-up loads Presidio, the FAISS index and the embedding model once per process.

---

## Failure F-2: human-gate routing audited twice on resume

**Evidence (run 1).** The audit trail shows `routed_for_human_signoff` for CLM-1001 twice:
- once when the case was drafted: `evidence_archive/run-1/logs/agent_actions.jsonl#L11` (run_id `4d80710094434829`);
- again when SIU-LEAD-01 resumed the thread to approve it: `evidence_archive/run-1/logs/agent_actions.jsonl#L12`
  (run_id `4b8222a3959c4b02`).

The second record comes from the resumed `human_gate` node, span_id `4d54cd519d957516` (run_id `4b8222a3959c4b02`).
An auditor would count two sign-off requests for one decision.

**Root cause.** LangGraph resumes an interrupted node by **re-executing it from the start**. The value passed to
`Command(resume=...)` is returned by `interrupt()`, but code that ran *before* `interrupt()` runs again. The audit
call sat before the interrupt in the same node.

**Fix.** The gate is split into two nodes:
- `request_signoff` (`src/agents/finalize.py::make_request_signoff`) runs once. It writes the audit record and sets
  `human_review.status = pending` in checkpointed state.
- `human_gate` (`src/agents/finalize.py::make_human_gate`) now only calls `interrupt()` and records the decision.

The regression test
`tests/test_routing.py::test_human_gate_audits_routing_exactly_once_across_interrupt_and_resume` asserts exactly one
routing record and one decision record per sign-off.

---

## Failure F-3: hallucination rate reported as 100 %

**Evidence (run 1).** `evidence_archive/run-1/reports/eval_report.json` reported `hallucination_rate: 1.0` and
`hallucination_score_mean: 1.0`, so the SLO in `evidence_archive/run-1/reports/golden_signals.json` showed
`hallucination_rate … met: false`.

The per-case judge output contradicts this. For G01 (run_id `57873a86d5fa4b5e`, case-writer span_id
`20c3a89513b9bba6`), the judge's reason reads "*the actual output is fully aligned with the context … without any
contradictions*", and DeepEval itself marked the metric **passed**. The same holds for all 16 judged cases, alongside
faithfulness 0.98 and relevancy 1.0.

**Root cause.** DeepEval 4.x reversed the polarity of `HallucinationMetric`. The score is now the *share of context
items the answer agrees with* (1.0 = fully grounded), and `threshold` is a *minimum*. DeepEval emits a
`DeprecationWarning` saying so. Our harness still used the 3.x meaning ("proportion of contradicted contexts;
> 0.5 = hallucinated"), so perfect grounding was reported as total hallucination.

**Fix.** `evals/run_eval.py` now:
- sets `GROUNDING_THRESHOLD = 0.8` (at most 20 % of evidence items may be contradicted);
- records `grounding_score` and `hallucination_score = 1 - grounding` per case;
- counts a case as hallucinated only when DeepEval's own `is_successful()` is false.

`hallucination_rate` is therefore the share of judged cases that failed grounding. The report states the definition
in `hallucination_definition`.

---

## Failure F-4: false-negative judge scores on G15 (history answer)

**Evidence (run 1).** Golden case G15 ("Show me the claim history and timeline for CLM-1005") has run_id
`9126b9dee4dc41e6`, and its answer comes from LLM span_id `d1512c903ec01d28` (run_id `9126b9dee4dc41e6`). The
deterministic checks all passed, but `evidence_archive/run-1/reports/eval_report.json` shows:
- **CaseCorrectness 0.2 (failed).** The judge explained that "*fraud likelihood tier, recommended action, matched
  fraud scheme ID … are almost entirely missing from both the expected and actual outputs*". Those facts are
  irrelevant to a history question.
- **Faithfulness 0.833.** The judge claimed the years 2023/2025 and the shared provider were "*not mentioned in the
  retrieval context*". They are in the claim record, both prior claims were at Synth Wellness Clinic 4, and the
  answer was correct.

**Root cause.** Two harness defects:
1. The GEval criteria hard-coded case-file facts (tier, action, scheme, sign-off), so any non-case-file answer was
   judged against facts it should not contain.
2. `evidence_context` truncated each tool record to 3,000 characters. For CLM-1005 that cut off `prior_claims`, so
   the judge never saw the dates and providers it later called unsupported.

**Fix.** In `evals/run_eval.py`:
- the GEval criteria now judge only the facts the *expected* output states;
- `evals/run_eval.py::evidence_context` gives the judge one evidence item per record facet (scalar fields, each prior
  claim, each indicator, each link), never cutting a record mid-way.

---

## Failure F-5: model mislabelled indicator RF-14 in a playbook answer

**Evidence (run 1).** Golden case G13 asked for arson-for-profit red flags (run_id `d05093663ffb441a`; answer from
LLM span_id `9faa1c2c8ca6a60f`, run_id `d05093663ffb441a`). The answer described **RF-14 as "physical
indicators"**. The judge flagged it (faithfulness 0.857 in `evidence_archive/run-1/reports/eval_report.json`):
"*the retrieval context specifies that it represents 'evidence inconsistency'*". This is a genuine, if small,
hallucination in a user-facing answer.

**Root cause.** For `playbook_lookup` the `respond` agent received the retrieved playbook chunks. Those chunks
mention codes such as "RF-14 Evidence inconsistency: accelerant patterns, contents moved out", but the agent had no
authoritative code→meaning table. The model paraphrased the code from the arson examples ("physical" evidence)
instead of its catalog name.

**Fix.** `src/agents/writers.py::make_respond` now adds the MCP resource `fraud://indicators/catalog` (every RF code
with its canonical name and rule) to the context for playbook, indicator and follow-up answers. The prompt already
restricts the model to provided context, so codes are now named from the catalog.

---

## Failure F-6: LLM tokens and cost could not be attributed to agents

**Evidence (run 1).** `evidence_archive/run-1/reports/golden_signals.json` reported
`tokens.by_agent = {"ChatGoogleGenerativeAI": 211719}`: all 162 LLM spans were lumped together. Every openinference
LLM span is named after the model class, e.g. span_id `9faa1c2c8ca6a60f` (run_id `d05093663ffb441a`) is
`ChatGoogleGenerativeAI`. Cost governance could not tell which agent to optimise.

**Root cause.** The golden-signals script grouped LLM spans by span name. The calling agent is only visible on an
ancestor span: our `LLMClient` names each call `<agent>:<Schema>` (e.g. `respond:Answer`, span_id `a9038ec904d53de8`,
run_id `d05093663ffb441a`).

**Fix.** `scripts/span_analytics.py::load_spans` walks each LLM span's parent chain to the nearest `<agent>:…` or
agent-node span. `scripts/golden_signals.py` then reports calls, prompt / completion / reasoning tokens, cost and
p50/p95 latency **per agent**. The case writer turned out to be the most expensive agent, which guided the thinking-
level optimisation in `docs/optimization-note.md`.

---

## Failure F-7: prepaid Gemini credit exhausted mid-run (HTTP 402)

**Evidence (final run).** During the red-team stage of the final evidence run the Gemini API began answering
`402 RESOURCE_EXHAUSTED — "Your prepayment credits are depleted"`:
- 19 LLM spans ended with that error, e.g. span_id `45aa70153dbdb380` (run_id `8d93832eba564f80`), recorded with
  status ERROR in `traces/phoenix_spans.jsonl`.
- The audit trail shows the graph switching to its deterministic fallback, e.g. `logs/agent_actions.jsonl#L305`
  (run_id `5b9126d11f5d45e6`, `degraded_mode / deterministic_fallback`).
- Two DeepEval judge calls failed the same way. `reports/eval_report.json` records them as `judge_errors: 2`
  (G14 and G16 case-correctness) rather than as scores.

No run crashed, and no guardrail was bypassed. The process was then stopped before its export step.

**Root cause.**
1. The account ran out of prepaid credit. Our cost model priced tokens with assumed rates (`PRICE_INPUT_PER_1M` /
   `PRICE_OUTPUT_PER_1M`) and did not meter the DeepEval judge's own Gemini calls, so real spend was well above the
   in-run estimate.
2. `src/llm.py::_retryable` classified the 402 as a retryable `RESOURCE_EXHAUSTED`, so each call was retried with
   back-off before falling back. That wasted time, and would have wasted money had credit remained.

**Fix.**
- `src/llm.py::_retryable` and `evals/judge.py::RetryingGeminiJudge` treat 402 / "credits are depleted" as
  **non-retryable**, so the node fails over to its deterministic fallback at once.
- `COPILOT_FORCE_OFFLINE=1` (`src/config.py`) guarantees zero model calls.
- `scripts/finalize_offline.py` completed the run with no API use. It re-opened the local Phoenix store, re-ran the
  red-team set offline (tag `redteam-offline`; the defences are deterministic guardrails), exported all spans and
  rebuilt every report.
- Remaining action: treat the configured prices as placeholders to be checked against current Gemini pricing, and
  count judge calls in cost estimates before any paid run.

---

## Verification of the fixes in the final run

The final run used the current code: demo and 22-case evaluation live on `gemini-3.5-flash`, and the red team
offline after F-7.

| Failure | Run 1 (before) | Final run (after) |
|---|---|---|
| F-1 cold start | first `input_guard` 44,994.7 ms; first playbook search 9,810.1 ms | first `input_guard` span_id `4000670f3159a1f5` (run_id `3c02b2e9d0c241d1`) took 61.2 ms; max `input_guard` 595.9 ms and max playbook search 177.9 ms (`reports/golden_signals.json`) |
| F-2 duplicate audit | two `routed_for_human_signoff` records for one CLM-1001 sign-off | one routing record `logs/agent_actions.jsonl#L11` (run_id `3c02b2e9d0c241d1`, agent `request_signoff`), then the decision `logs/agent_actions.jsonl#L12` (run_id `0f055c38911045b1`) from resumed span_id `96d8734df659c08f` (run_id `0f055c38911045b1`) |
| F-3 metric polarity | hallucination_rate 1.0 while every judge reason said "no contradictions" | `reports/eval_report.json`: hallucination_rate 0.0, grounding_score_mean 1.0, faithfulness 0.994; hallucination SLO met in `reports/golden_signals.json` |
| F-4 G15 false negatives | CaseCorrectness 0.2, faithfulness 0.833 | G15 (run_id `d642f174dfa347d7`, respond span_id `62fc028fdfb750c8`): CaseCorrectness 1.0, faithfulness 1.0 |
| F-5 RF-14 mislabel | G13 faithfulness 0.857 | G13 (run_id `19029b309256409f`, respond span_id `130550a32bee6792`): faithfulness 1.0 |
| F-6 token attribution | `{"ChatGoogleGenerativeAI": 211719}` | per-agent calls / tokens / cost / latency in `reports/golden_signals.json` (`tokens.by_agent`) |
| F-7 credit exhaustion | n/a | runs degraded instead of failing; offline completion via `scripts/finalize_offline.py`; red team 12/12 defended (`reports/redteam_results.json`) |

---

## Other failures observed during development (not in the committed traces)

These happened before tracing was enabled for evidence, so they are listed for completeness only:
- **Gemini free-tier quota exhaustion.** `429 RESOURCE_EXHAUSTED`, quotaId `GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
  limit 20. Fix: `src/llm.py::_retryable` treats per-day quota errors as non-retryable, so the run degrades to
  deterministic fallbacks immediately. Other 429s honour the server's `retryDelay`. A client-side limiter was added
  (`src/llm.py::RateLimiter`, `LLM_RPM`).
- **Gemini `503 UNAVAILABLE` ("high demand").** Handled by bounded exponential back-off in `src/llm.py::LLMClient`
  and by `evals/judge.py::RetryingGeminiJudge` for the DeepEval judge.
- **Phoenix in-process start-up timeout on Windows.** Phoenix's thread server hard-codes a 5 s limit, and a cold start
  that creates a fresh SQLite DB takes longer. Fix in `src/observability/tracing.py::init_tracing`.
- **Case-file citation quoted `"1."`.** Sentence splitting on a numbered list. Fix:
  `src/tools/rag_tool.py::quote_of`.


