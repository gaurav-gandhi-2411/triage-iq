# ADR-0011: Eval-Regression CI Gate

Status: Accepted
Date: 2026-06-20

## Context

Every quality check — conformal Q-invariant, calibration ECE, LLM-judge score — was run by
hand as a manual escalation step. That doesn't scale and relies on a human remembering to run
it. This ADR documents the decision to automate these checks as a CI gate on every PR/push.

The gate is split into two layers:

1. **Structural invariants** (no LLM, deterministic, fast): verifies mathematical contracts,
   schema shape, and calibration properties that must hold regardless of model quality.
2. **Quality regression** (cassette-replayed LLM-judge): verifies that the end-to-end pipeline
   score on a frozen eval set does not drop below a committed baseline. **This layer is deferred
   to a follow-up PR** — see "Consequences" below.

A key design constraint: the gate must run fully offline in CI (no live LLM, no live judge calls,
no Groq quota consumed). All LLM interactions are pre-recorded into committed cassettes and
replayed deterministically.

## Decision

**Structural invariant suite** (`eval/test_invariants.py`) — shipped in this PR:

Five deterministic checks, zero LLM calls:

| Test | What it guards |
|------|---------------|
| `test_conformal_q_formula` | `conformal.upper == raw_hi + Q` and `conformal.lower == max(0, raw_lo - Q)` (1e-6 tolerance) |
| `test_conformal_layer_active` | Both repos present in JSON; Q > 0; empirical_coverage > 50%; target_coverage ≈ 80% |
| `test_triage_plan_schema_contract` | All 14 required `TriagePlan` fields present with correct types; `ConformalIntervalResult` fields present |
| `test_calibration_ece_in_tolerance` | Calibrated ECE per repo within ±0.15 of recorded values (vscode 0.1381, k8s 0.1558) |
| `test_conformal_coverage_on_eval_set` | Per-repo conformal coverage on 60-issue eval set in [40%, 100%]; no empty intervals |

Eval set (`eval/eval_set.jsonl`): 60 issues (30 vscode + 30 k8s), stratified by resolution
bucket (<7d, 7–30d, >30d), derived from `data/gold_triage_plans.parquet`.

Cassette mechanism (`eval/cassette.py`): JSON-backed replay player keyed on SHA-256 of the
canonical request (same hash as `LLMCache`). `strict=True` raises `CassetteMissError` on miss
(CI replay mode). `strict=False` falls through to the live call (recording mode, local only).

CI wiring (`.github/workflows/eval-gate.yml`): runs on every push/PR, downloads production
models from GCS, runs `pytest eval/test_invariants.py -v --no-cov`. **Non-blocking**
(`continue-on-error: true`) — will be promoted to a required status check after one confirmed
green cycle on the main branch.

**Quality regression suite** — deferred to follow-up PR:

The full pipeline (synthesis + judge) must be re-run in replay-only mode and scored per-repo
against a committed baseline (`reports/eval_baseline.json`). The recording pass to populate
cassettes was interrupted by Groq TPD at issue 44/60 (vscode complete 30/30, k8s partial
13/30). The follow-up PR will:

- Resume recording from checkpoint (16 k8s issues remaining)
- Build `eval/test_quality_regression.py` with per-repo thresholds (0.0 drop + 1e-4 epsilon)
- Commit `reports/eval_baseline.json` (blessed scores + eval-set hash + cassette hash)
- Add quality regression job to `eval-gate.yml`, non-blocking

## Important limitation: code-vs-recorded, not code-vs-live

**The gate verifies code against recorded interactions, not against the live API.** This is a
deliberate design choice (quota-immunity, determinism, CI speed) with a known consequence:
the gate does not detect model drift, provider-side changes, or shifts in live Groq behavior.
It catches unintended changes to *this codebase* — prompt edits, schema changes, retrieval
changes — that alter what requests are sent or how responses are parsed.

An intentional change (prompt update, model swap) requires: re-record cassettes locally
(`python eval/record_cassettes.py`), re-run eval to get new scores (`python eval/run_eval.py`),
update `reports/eval_baseline.json`, and include all three artifacts in the same PR. A cassette
miss in CI is therefore a signal to re-record, not a signal to revert.

## Consequences

- **What changes:** structural invariants now run automatically on every PR; breaks in the
  conformal layer, schema, calibration, or coverage are caught before merge.
- **What becomes harder:** intentional changes to `TriagePlan` schema, calibration results,
  or the conformal JSON require updating the test expectations in the same PR.
- **What becomes easier:** verifying that infrastructure PRs (dependency updates, refactors,
  CI changes) haven't accidentally broken the pipeline's mathematical contracts.
- **Known gap (until quality layer ships):** synthesis quality regressions are not caught
  automatically. Manual eval (`scripts/11_evaluate_triage.py`) is still required for LLM
  prompt changes.

## Alternatives considered

- **Freeze plans, skip re-running synthesis** — rejected because it guards parse/score logic
  only, not the synthesis request itself. An upstream prompt change that alters the synthesis
  call would pass the gate silently.
- **vcrpy / responses library** — rejected; the repo already has `LLMCache` with SHA-256
  canonical request hashing. Adding a second cassette library would duplicate that logic and
  add a dependency. `eval/cassette.py` reuses the same key function.
- **Blocking from day one** — rejected per spec. Ship non-blocking first; promote after one
  confirmed green cycle to avoid breaking the branch on a first-run infra issue.
