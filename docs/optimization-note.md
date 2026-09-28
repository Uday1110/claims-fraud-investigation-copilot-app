# Optimisation Note: measured before/after (Phoenix-derived)

**Change:** run the agents at Gemini `thinking_level=low` with context compression on and at most 2 agentic-RAG
rounds, instead of `thinking_level=high`, compression off and 3 rounds.

**Result:** end-to-end latency −50.5 % (p50), cost −64 %, the same recommendations, and the same number of LLM calls.

## Method
`scripts/optimization_benchmark.py` ran the same three investigations (CLM-1001 high / CLM-1005 medium / CLM-1008
low) twice inside the traced process of evidence **run 1**. Each phase's runs were tagged, then the Phoenix spans
were split by tag and a golden-signals report was computed for each phase with the same code as the headline report
(`scripts/golden_signals.py`).

| Phase | Config |
|---|---|
| `bench-baseline` | `GEMINI_THINKING_LEVEL=high`, `CONTEXT_COMPRESSION=false`, `MAX_RETRIEVAL_ROUNDS=3` |
| `bench-optimized` | `GEMINI_THINKING_LEVEL=low`, `CONTEXT_COMPRESSION=true`, `MAX_RETRIEVAL_ROUNDS=2` (the shipped defaults) |

The two Phoenix-derived reports:
- `evidence_archive/run-1/reports/optimization/golden_signals_baseline.json`
- `evidence_archive/run-1/reports/optimization/golden_signals_optimized.json`
- The comparison is in `evidence_archive/run-1/reports/optimization/comparison.json`.
- The underlying spans are in `evidence_archive/run-1/traces/phoenix_spans.jsonl`. For example, the CLM-1001 baseline
  turn is root span_id `72301279a7d3c597` (run_id `63c5b8fef565408f`, 50.7 s), and the optimised turn is root span_id
  `40e13910d99fba74` (run_id `70f651337b03499d`, 24.9 s).

## Results (3 investigations per phase)

| Metric (Phoenix-derived) | Baseline | Optimized | Change |
|---|---|---|---|
| End-to-end run latency p50 | 47.8 s | 23.6 s | **−50.5 %** |
| End-to-end run latency p95 | 50.4 s | 24.7 s | **−50.9 %** |
| LLM ("thinking") span latency p50 | 5.07 s | 2.72 s | −46.4 % |
| Prompt tokens | 20,252 | 19,672 | −2.9 % |
| Completion tokens (incl. reasoning) | 27,834 | 8,525 | −69.4 % |
| Reasoning tokens | 22,225 | 2,448 | **−89.0 %** |
| Estimated cost (configured prices) | $0.0757 | $0.0272 | **−64.0 %** |
| LLM calls | 21 | 21 | 0 % |
| Recommended actions | refer / monitor / close | refer / monitor / close | unchanged |

## Interpretation
- **Where the saving comes from.** Almost entirely from reasoning tokens: at `thinking_level=high`, Gemini spent
  about 22k hidden reasoning tokens on tasks that are mostly structured extraction and drafting. Output tokens
  dominate both cost and latency, so cutting them halved latency and cut cost by about two-thirds.
- **Why quality held.** The indicator score and tier come from the deterministic MCP tool, and the output guard
  enforces the GL-01 action policy (`src/guardrails/output_guard.py::check_case_file`). The model's reasoning depth
  therefore doesn't change the decision. It only changes the narrative, which the eval still rates grounded
  (`reports/eval_report.json`: hallucination rate 0 %, faithfulness 0.994 at the optimised settings).
- **Compression.** Prompt tokens fell only 3 %, because the tool outputs for these claims are small. Compression
  matters more for long claim histories and multi-turn threads, where `src/context/compress.py` and the
  summarisation middleware bound the prompt.
- **Where to optimise next.** The per-agent breakdown in `reports/golden_signals.json` (`tokens.by_agent`, enabled by
  fix F-6) shows `case_writer` as the most expensive agent. It's the next target, e.g. a shorter case-file template.

## Caveats
- **Sample size.** Three runs per phase, so treat the percentages as indicative, not a benchmark.
- **Prices are assumptions.** Cost figures use the configured placeholder prices (`PRICE_INPUT_PER_1M=0.30`,
  `PRICE_OUTPUT_PER_1M=2.50` in `.env.example`), not verified `gemini-3.5-flash` rates. Actual billing for this project
  was higher (see F-7 in `docs/failure-analysis.md`). The *relative* saving holds because both phases use the same
  prices and the saving is almost all output tokens.
