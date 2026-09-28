# Compliance Mapping — EU AI Act · NIST AI RMF 1.0 · India DPDP Act 2023

System: Claims Fraud-Investigation Copilot (decision-support for an insurer's Special Investigation Unit).
Data: 100 % synthetic (`scripts/generate_synthetic_data.py`). Model provider: Google Gemini only.
Every row maps an obligation → how the system addresses it → the committed evidence artifact.
This is an engineering mapping for the hackathon, not legal advice.

## 1. Classification rationale

* **EU AI Act.** Insurance fraud *detection* is not itself listed in Annex III (Annex III 5(c) covers risk
  assessment and pricing for life/health insurance; 5(b) explicitly carves out fraud detection from the
  credit-scoring category). Outputs can nonetheless affect individuals (an SIU referral), so the system is operated
  **as if high-risk**: we voluntarily apply the Articles 9–15 controls below, plus the Article 50 transparency duty.
* **NIST AI RMF.** Applied across GOVERN / MAP / MEASURE / MANAGE; the risk register (`docs/risk-register.md`) is
  the MAP/MANAGE artifact.
* **DPDP Act 2023 (India).** Claim data is personal data of claimants (data principals); the insurer is the data
  fiduciary. The prototype processes only synthetic data, but the controls are built as they would be for real data.

## 2. EU AI Act

| Obligation | How addressed | Evidence |
|---|---|---|
| Art. 9 — risk-management system | Risk register with OWASP/NIST categories, owners and residual ratings; red-team suite re-run on every evidence regeneration | `docs/risk-register.md`, `reports/redteam_results.json`, `scripts/run_redteam.py` |
| Art. 10 — data & data governance | Only synthetic, seeded, reproducible data; data model excludes protected attributes; identifiers masked at the MCP source | `scripts/generate_synthetic_data.py`, `data/claims/`, `mcp_server/claims_data.py` |
| Art. 11 + Annex IV — technical documentation | Model/system card, architecture in module docstrings, runbook, governance pack | `docs/model-card.md`, `src/graph.py`, `README.md` |
| Art. 12 — record-keeping (automatic logs) | Machine-generated audit trail of consequential actions; per-call tool log; MCP transcript; OpenTelemetry traces exported from Phoenix, all correlated by `run_id` | `logs/agent_actions.jsonl` (`src/audit/audit.py`), `logs/tool_calls.jsonl` (`src/tools/tool_logging.py`), `logs/mcp_transcript.jsonl`, `traces/phoenix_spans.parquet` |
| Art. 13 — transparency to deployers | Case files list every supporting indicator with evidence, the matched playbook section with verbatim quotes, limitations and a degraded-mode flag; indicator catalog published as an MCP resource | `reports/demo_run.md`, `mcp_server/server.py` (`fraud://indicators/catalog`) |
| Art. 14 — human oversight | Adverse / escalation outputs pause at a LangGraph `interrupt()` until an authorised SIU reviewer approves or rejects; copilot never auto-actions; reviewer authority checked | `src/agents/finalize.py::make_human_gate`, `src/guardrails/access.py::can_sign_off`, `human_signoff_recorded` events in `logs/agent_actions.jsonl` |
| Art. 15 — accuracy, robustness, cybersecurity | Golden-set evaluation (accuracy, hallucination, faithfulness); retries/timeouts/fallbacks; loop guards; input/output guardrails; quarantine of untrusted text | `reports/eval_report.json`, `reports/golden_signals.json`, `tests/test_loops.py`, `src/guardrails/`, `src/context/quarantine.py` |
| Art. 26 — deployer duties (monitoring, logs retention) | Golden-signals SLOs and cost/latency dashboard for operational monitoring | `reports/golden_signals.json`, `reports/dashboard.png`, `reports/dashboard_data.csv` |
| Art. 50 — transparency of AI interaction | CLI/API outputs are labelled as drafts from the copilot; case files are marked DRAFT and state when human sign-off is required | `src/agents/writers.py::render_case_file` |

## 3. NIST AI RMF 1.0

| Function / subcategory | How addressed | Evidence |
|---|---|---|
| GOVERN 1.1–1.4 (policies, accountability) | Named risk owners; SIU action policy GL-01 encoded as a playbook and enforced in code | `docs/risk-register.md`, `data/policy_corpus/GL-01_siu_action_guidelines.md`, `src/guardrails/output_guard.py` (`TIER_ACTION`) |
| GOVERN 1.6 (inventory) / 6.1 (third-party) | Pinned dependency inventory; single model provider | `requirements.txt`, `src/llm.py` |
| MAP 1.1, 3.4 (context, intended use, limits) | Intended use, users, out-of-scope uses and limitations documented | `docs/model-card.md` |
| MAP 5.1 (likelihood & impact of harms) | Risk register likelihood × impact | `docs/risk-register.md` |
| MEASURE 2.3, 2.5 (performance, validity) | Deterministic task accuracy + DeepEval LLM-as-judge on a golden set | `evals/run_eval.py`, `reports/eval_report.json` |
| MEASURE 2.7 (security & resilience) | Red-team suite; guardrail unit tests | `reports/redteam_results.json`, `tests/test_guardrails.py` |
| MEASURE 2.10 (privacy) | Leak scan over all committed logs/traces/reports; Presidio redaction sample | `tests/test_guardrails.py`, `reports/pii_redaction_sample.json` |
| MEASURE 2.11 (fairness) | Transparent weighted indicators with no protected attributes; human review of adverse outcomes | `mcp_server/claims_data.py` (`INDICATORS`), `src/agents/finalize.py` |
| MEASURE 3.1 (ongoing monitoring) | Phoenix tracing on every run; golden signals with SLOs | `src/observability/tracing.py`, `scripts/golden_signals.py` |
| MANAGE 2.2–2.4 (risk treatment, fallback, disengage) | Guardrail blocks, loop guard, deterministic fallbacks, human gate, per-run cost budget | `src/agents/supervisor.py`, `src/llm.py`, `src/observability/cost.py` |
| MANAGE 4.1 (post-deployment monitoring, incidents) | Failure-mode analysis from real runs with root cause and fix | `docs/failure-analysis.md`, `reports/failure_candidates.json` |

## 4. DPDP Act 2023 (India)

| Obligation | How addressed | Evidence |
|---|---|---|
| s.4 / s.7 — lawful purpose (fraud prevention as a legitimate use; purpose limitation) | Copilot scope limited to SIU investigation intents; out-of-scope requests declined | `src/agents/intake.py` (intent router), `src/agents/writers.py::make_out_of_scope` |
| s.5 — notice | Deployment note: claimant-facing notice that claims may be screened by an AI-assisted SIU process (process control, not code) | this document |
| s.8(3) — accuracy/completeness where data is used for a decision affecting the principal | Scores computed from source records by a deterministic tool; hallucinated indicators dropped; human sign-off before any adverse action | `src/agents/workers.py`, `src/guardrails/output_guard.py::check_case_file`, `src/agents/finalize.py::make_human_gate` |
| s.8(4) — reasonable security safeguards (and data minimisation) | Masking at source, ingress and egress; access control per assignment; least-context selection per agent | `mcp_server/server.py`, `src/runner.py`, `src/guardrails/pii.py`, `src/guardrails/access.py`, `src/context/select.py` |
| s.8(5)/(6) — breach prevention and intimation | Audit trail + tool logs support incident reconstruction; access refusals logged | `logs/agent_actions.jsonl` |
| s.8(7) — erasure when purpose is served | Runtime stores (checkpoints, long-term memory) are local, git-ignored and wiped by the regeneration pipeline; memory is keyed per investigator and claim so it can be deleted selectively | `.gitignore`, `scripts/regenerate_evidence.py::reset`, `src/memory/long_term.py` |
| s.11–13 — rights of access, correction, grievance | Case files cite the source indicators and evidence, enabling an explanation / correction workflow by the SIU team | `src/agents/writers.py::render_case_file` |
| s.9 — children's data | Not processed (synthetic adult claimants only) | `scripts/generate_synthetic_data.py` |
