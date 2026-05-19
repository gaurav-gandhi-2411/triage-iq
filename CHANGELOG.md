# Changelog

All notable changes to TriageIQ will be documented in this file.

Format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

The git log (`git log --oneline`) is the authoritative record of historical changes.
This file documents *what matters and why*, not every commit.

---

## [Unreleased]

### Changed

- Eval: cross-family judge validation completed on Cohere Command A (Trial key, `command-a-03-2025`).
  Full 180/180 scores (n=60 issues × 3 systems). Verdict: r=0.73 vs llama-3.3-70b-versatile,
  gap −3.6pp on Full System — default judge retained (ADR-0003). Groq TPD non-viable at n=60
  and Gemini 2.5 Flash free tier confirmed at 20 RPD (not 1,500) — both documented in ADR-0002
  amendments. Cohere Trial is the established cross-family sanity-check path going forward.

- API: `/triage` endpoint declares `response_model=TriagePlan`; OpenAPI spec now includes
  `TriagePlan` and `SimilarIssue` in `components/schemas` and a typed 200 response ref.
  No runtime change — endpoint still returns `JSONResponse` directly. Unblocks UI OpenAPI
  codegen (ADR-0001). Note: `_request_id` and `_llm_status` are injected post-serialization
  and are not part of the schema; UI types them as an intersection.

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
  gemini-1.5-flash deprecated) before settling on `qwen/qwen3-32b` via Groq.
- `scripts/11_evaluate_triage.py` now accepts `--judge-model`, `--judge-provider`,
  and `--output-file` CLI args (also via `TRIAGE_JUDGE_MODEL` / `TRIAGE_JUDGE_PROVIDER`
  env vars). Judge checkpoint is scoped per model (`judge_scores_checkpoint_{slug}.jsonl`).
  `_is_tpd_error` narrowed to Groq-specific keywords so Gemini 429s don't trigger TPD exit.
- Baseline llama-70b judge results archived at `reports/triage_results_judge_llama70b.json`
  for before/after comparison.

### Documentation

- Baseline audit (`docs/audit/2026-05-18-baseline.md`): immutable discovery snapshot
  covering repo topology, data, modeling, eval, serving, CI/CD, and completion status
  as of commit `dbde81679207fb521fe5c65eb91bdf30261f2246`.
- ADR infrastructure (`docs/architecture/adr/`): MADR-lite template, README, and first
  real ADR (ADR-0001: keep repos split, resolve type drift via openapi-typescript).
- ADR-0002: cross-family judge selection — Gemini 2.5 Flash via Google AI Studio.
- Contributing discipline (`docs/CONTRIBUTING.md`): change-documentation requirements
  for every substantive PR.

---

<!-- Links populated when releases are cut -->
[Unreleased]: https://github.com/gaurav-gandhi-2411/triage-iq/compare/HEAD...HEAD
