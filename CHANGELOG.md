# Changelog

All notable changes to TriageIQ will be documented in this file.

Format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

The git log (`git log --oneline`) is the authoritative record of historical changes.
This file documents *what matters and why*, not every commit.

---

## [Unreleased]

### Changed

- Internal naming: "duplicate detection" renamed to "similar issue retrieval" throughout the
  codebase. `DuplicateDetector` class → `SimilarIssueRetriever`, `duplicates.py` → `similar_issues.py`,
  `gold_duplicates.parquet` → `gold_related.parquet`, `duplicate_results.json` →
  `related_issue_results.json`, construction scripts renamed. No behavior change, no API change,
  no metric change. User-facing `SimilarIssue` Pydantic type was already correctly named.
  See ADR-0008.

### Added

- `TriagePlan.resolution_bucket` (supplemental): coarse ordinal bucket from new bucket
  classifier (hours/days/weeks/months/long). Bucket computed independently of the float
  fields; both are returned in API responses. k8s passes the 60% off-by-one accuracy
  threshold (65.9%); vscode uses naive prior (insufficient creation-time signal). See ADR-0009.
- `TriagePlan.resolution_confidence_pct`: bucket classifier confidence, 0–100%. Values below
  40% indicate low signal (vscode will typically be below threshold).

### Fixed

- **Resolution predictor: temporal split methodology** — `closed_at` ordering produced a
  systematic train/test distribution shift (train=fast-resolvers, test=decade-long stragglers)
  that made k8s CI coverage 0% and rendered all prior metrics invalid. Split now uses
  `created_at` ordering. k8s CI coverage: 0% → 77%. See ADR-0009.
- **Resolution predictor: feature leakage** — `has_priority` (top feature by gain, corr 0.595
  with log(resolution_hours)) was assigned post-creation during triage. Dropped from
  `engineer_features()` alongside `has_component`, `has_type`, `num_assignees`, and `comp_*`
  one-hots. Honest de-leaked MAE: k8s +1.4% over naive; vscode 0% (no creation-time signal).

### Documentation

- ADR-0009: resolution predictor Phase 1 diagnosis + Phase 2 ablation table + judge impact.
  Prior resolution metrics (k8s +3.3%, vscode +19.1%) explicitly invalidated.
- `reports/01_data_card.md` §6: split methodology correction noted, prior metrics retracted.

### Evaluated (no change shipped)

- **W4 Phase 2 T2.7 — bucket-only LLM prompting** regressed `resolution_estimate_reasonableness`
  by −0.532 (1.617 → 1.085, Cohere judge). Float signals retained in LLM prompt; bucket exposed
  as additional API field only. See ADR-0009 T2.7.
- **W4 Phase 2 final (Config A — de-leaked float):** `resolution_estimate_reasonableness`
  improved +0.333 (1.617 → 1.950). De-leaked calibrated intervals (CI 77%) produce better LLM
  narrative than W1.2's broken intervals (CI 0%). W1.2's 1.617 baseline itself used leaky
  float signals; the honest production baseline is 1.950. Overall total: −0.067 (flat). W4
  ships. See ADR-0009 T1–T5.

### Evaluated (no change shipped — prior sessions)

- **W3 Phase 2 — bge-v2-m3 repo-gated reranker:** Robustness check at n=300 (k8s, seed=42,
  1000-resample bootstrap) showed the W1.3 screening k8s +6pp was a small-sample false positive.
  True delta at n=300: +0.67pp, 95% CI [−3.7pp, +5.3pp]. CI crosses zero → stop condition
  triggered at T2. T3/T4/T5 not run. ADR-0006 rejection stands under reframing. Next step:
  W3 fine-tuning on `gold_related.parquet` pairs. See ADR-0006 Phase 2 verdict section.

---

## W2.A — LLM Response Cache (2026-05-19)

### Added

- LLM response cache (`src/triage_iq/cache/`): opt-in SQLite-backed cache keyed on
  SHA-256 of canonical request JSON (`schema_version + provider + model + messages +
  temperature + max_tokens`, plus `response_format` for Cohere structured output).
  Enabled via `LLM_CACHE_ENABLED=true`. Default: disabled. Thread-safe
  (`threading.Lock` + WAL mode). Cache hits return in <5 ms; misses fall through to live API.
  Integrated in `TriageAssistant`, `TriageJudge` (all three providers), `/triage` endpoint,
  and `scripts/11_evaluate_triage.py`. Four Prometheus metrics: `triage_llm_cache_hits_total`,
  `triage_llm_cache_misses_total`, `triage_llm_cache_size_bytes`, `triage_llm_cache_entries`.
  Admin CLI: `scripts/13_cache_admin.py` (stats / clear / clear-provider / clear-model).
  Validated: cold→warm on 180 Cohere judge calls, 1974.5s→9.2s (214×), 100% hit rate,
  0.000 score drift. ADR-0005.

---

## W1.2 — Classifier Confidence Calibration + Cross-Family Judge Verdict (2026-05-19)

### Added

- Calibration: temperature scaling for component classifier probability estimates.
  `TemperatureScaler` class in `component_classifier.py` scales LR logits by `1/T` before
  softmax (T=0.2981 vscode, T=0.3234 kubernetes), fitted by minimising NLL on val split.
  ECE: vscode 0.50→0.15 (val), kubernetes 0.34→0.13 (val). argmax preserved; test accuracy
  unchanged (+0.0pp on both repos). Calibrator saved inside classifier pkl; loader has
  graceful fallback (`calibrator=None` → raw proba). `triage.py` uses calibrated
  probabilities. ADR-0004. Diagnostic script `scripts/12b_calibration_diagnostic.py`
  documents T1 leakage audit, T2 temperature scaling vs T3 isotonic robustness
  (isotonic accuracy gain was noise: bootstrap CI crosses zero on both repos).

- Cross-family judge support: `TriageJudge` accepts `provider` param; `provider="gemini"`
  routes to `google-genai`; `provider="groq"` (default) is unchanged. ADR-0002 documents
  three failed candidates (gemma2-9b-it decommissioned; gemini-2.5-flash 20 RPD free tier;
  gemini-1.5-flash deprecated) before settling on `qwen/qwen3-32b` via Groq, then pivoting
  to Cohere Command A as the working free-tier cross-family judge path.
- `scripts/11_evaluate_triage.py` now accepts `--judge-model`, `--judge-provider`,
  and `--output-file` CLI args (also via `TRIAGE_JUDGE_MODEL` / `TRIAGE_JUDGE_PROVIDER`
  env vars). Judge checkpoint is scoped per model (`judge_scores_checkpoint_{slug}.jsonl`).
  `_is_tpd_error` narrowed to Groq-specific keywords so Gemini 429s don't trigger TPD exit.
- Baseline llama-70b judge results archived at `reports/triage_results_judge_llama70b.json`
  for before/after comparison.

### Changed

- Eval: cross-family judge validation completed on Cohere Command A (Trial key, `command-a-03-2025`).
  Full 180/180 scores (n=60 issues × 3 systems). Cross-family verdict: W1.2 scores 10.83/15
  (72.2%) vs W1.1 10.40/15 (69.3%), +0.43 (+2.89pp). Calibration improvement concentrated in
  `resolution_estimate_reasonableness` (+0.17) and `component_match`/`overall_quality` (+0.10
  each). Default Llama judge retained (ADR-0003). Groq TPD non-viable at n=60 and Gemini 2.5
  Flash free tier confirmed at 20 RPD (not 1,500) — documented in ADR-0002 amendments.
  Cohere Trial is the established cross-family sanity-check path going forward.

- API: `/triage` endpoint declares `response_model=TriagePlan`; OpenAPI spec now includes
  `TriagePlan` and `SimilarIssue` in `components/schemas` and a typed 200 response ref.
  No runtime change — endpoint still returns `JSONResponse` directly. Unblocks UI OpenAPI
  codegen (ADR-0001). Note: `_request_id` and `_llm_status` are injected post-serialization
  and are not part of the schema; UI types them as an intersection.

### Documentation

- Baseline audit (`docs/audit/2026-05-18-baseline.md`): immutable discovery snapshot
  covering repo topology, data, modeling, eval, serving, CI/CD, and completion status
  as of commit `dbde81679207fb521fe5c65eb91bdf30261f2246`.
- ADR infrastructure (`docs/architecture/adr/`): MADR-lite template, README, and first
  real ADR (ADR-0001: keep repos split, resolve type drift via openapi-typescript).
- ADR-0002: cross-family judge selection (three iterations + Cohere pivot documented).
- ADR-0004: temperature scaling decision, diagnostics, and verdict (W1.1→W1.2 delta).
- Contributing discipline (`docs/CONTRIBUTING.md`): change-documentation requirements
  for every substantive PR.

---

<!-- Links populated when releases are cut -->
[Unreleased]: https://github.com/gaurav-gandhi-2411/triage-iq/compare/HEAD...HEAD
