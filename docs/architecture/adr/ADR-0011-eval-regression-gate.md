# ADR-0011: Eval-Regression CI Gate

Status: Accepted
Date: 2026-06-20
Last updated: 2026-06-20

## Context

Every quality check — conformal Q-invariant, calibration ECE, LLM-judge score — was run by
hand as a manual escalation step. That doesn't scale and relies on a human remembering to run
it. This ADR documents the decision to automate these checks as a CI gate on every PR/push.

The gate is split into two layers:

1. **Structural invariants** (no LLM, deterministic, fast): verifies mathematical contracts,
   schema shape, and calibration properties that must hold regardless of model quality.
2. **Quality regression** (cassette-replayed LLM-judge): verifies that the end-to-end pipeline
   score on a frozen eval set does not drop below a committed baseline.

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

**Quality regression suite** (`eval/test_quality_regression.py`) — shipped in this PR:

**Baseline:** `reports/eval_baseline.json` (committed) — blessed scores with eval-set hash +
cassette hash. Scores were validated against a full 60-issue recording (30 vscode + 30 k8s).
The partial (n=13) k8s sample from the interrupted first recording run was validated: partial
mean 10.69 vs full-30 mean 10.97 (+1.9% delta), confirming the sample was representative
before the baseline was locked.

**Thresholds (per-repo, independent):**

- microsoft/vscode: baseline 9.8667/15 — gate fails if drop > 0.0001
- kubernetes/kubernetes: baseline 10.9667/15 — gate fails if drop > 0.0001
- Threshold = 0.0 absolute drop + 1e-4 epsilon (float noise tolerance only)
- Both repos gated independently: a vscode-only regression fails even if k8s holds
- On failure: per-criterion sub-scores emitted (6 dimensions shown with baseline/current/delta)

**CI wiring:** `eval-gate.yml` job `quality-regression`, `continue-on-error: true`
(non-blocking until promoted after first confirmed green cycle).

**Gate behavior:**

- Cassette hit + score within threshold → PASS
- Cassette miss (any LLM call not in cassette) → hard fail (`CassetteMissError`) — signals
  that the system under test changed without a corresponding cassette update
- Score drop beyond threshold → fail with per-criterion breakdown

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
- **Quality regressions on the frozen 60-issue eval set are now caught automatically.** Manual
  eval is still needed for testing against live API behavior (the gate verifies code-vs-recorded,
  not code-vs-live).

## Alternatives considered

- **Freeze plans, skip re-running synthesis** — rejected because it guards parse/score logic
  only, not the synthesis request itself. An upstream prompt change that alters the synthesis
  call would pass the gate silently.
- **vcrpy / responses library** — rejected; the repo already has `LLMCache` with SHA-256
  canonical request hashing. Adding a second cassette library would duplicate that logic and
  add a dependency. `eval/cassette.py` reuses the same key function.
- **Blocking from day one** — rejected per spec. Ship non-blocking first; promote after one
  confirmed green cycle to avoid breaking the branch on a first-run infra issue.

## Frozen retrieval: synthesis eval vs retrieval correctness

The synthesis eval (`test_quality_regression.py`) uses **frozen similar_issues** committed in
`eval_set.jsonl` rather than live FAISS retrieval. The frozen top-k was produced on CPU float32
(`CUDA_VISIBLE_DEVICES=""`) and its provenance recorded in `eval/frozen_retrieval_provenance.json`.

**Root cause of the original cassette key coupling:** BAAI/bge-base-en-v1.5 query-time
embeddings differ by ~1e-4 between CUDA and CPU float32 paths. These differences propagate to
cosine similarity scores, which can flip top-k ordering at tight rank boundaries (e.g. vscode
#311565, rank-5/6 gap = 0.0000). The synthesis prompt includes the similar_issues text, so any
ordering change produces a different prompt → different cassette key → CI replay miss.

**Option C fix (permanent):** commit `similar_issues` as a field in `eval_set.jsonl` and route
the eval path through `FrozenRetriever` (duck-typed, lives only in `eval/frozen_retriever.py`).
The synthesis prompt is now a deterministic function of committed inputs on any hardware.

**Retrieval correctness is tested separately:** `test_retrieval_top_k` (invariant #7 in
`eval/test_invariants.py`) asserts that live FAISS on the current index agrees with the frozen
top-k for 2 probe issues per repo (top-1 exact match + ≥4/5 set-membership). Full ordering is
not asserted — float-path variation makes exact ordering unstable at tied rank positions.
Production `/triage` is unchanged and continues to use live `SimilarIssueRetriever`.

## Baseline update procedure

When intentionally changing the system under test (prompt edit, model swap, retrieval change):

1. Make the change on your branch.
2. Re-record cassettes locally (requires GROQ_API_KEY):
   `python eval/record_cassettes.py`
3. Compute new scores and update baseline:
   `python eval/run_eval.py --update-baseline`
4. Review the score delta in the baseline diff. If it represents a regression, justify it in
   the PR description.
5. Commit the updated cassette + checkpoint + baseline in the same PR as the system change.
   CI will use the new cassette for replay — no live calls in the gate.

A cassette miss in CI (CassetteMissError) means someone changed the system without following
this procedure. Fix: re-run steps 2–5 above.
