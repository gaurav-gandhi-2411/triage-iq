# ADR-0028 — Per-Model Evaluation Audit: Metric Suitability Across All 4 Models

**Status:** Accepted (audit only — no fixes applied, no model or config changes, no cutover)
**Date:** 2026-07-13
**Decider:** Escalated to Gaurav Gandhi for fix-order sign-off; audit executed autonomously by CC per spec.md.

---

## Context

Across this project, metric-suitability bugs have surfaced ACCIDENTALLY, one model at a time,
never as a deliberate sweep:

- Retrieval's k8s gold set was 92% PR→issue pairs — it measured "given a PR, find the issue it
  fixes," not the product task "given a new issue, find related issues" (ADR-0026/0027).
- Resolution's conformal intervals were shown non-diagnostic (ADR-0021/0023), and vscode's bucket
  classifier was found to underperform a naive baseline (ADR-0025).
- The judge-eval gold set had train/test contamination (ADR-0018).

Every one of these was found while working on something else. This ADR is the deliberate,
systematic version: every model checked against the same 4 failure modes — proxy-vs-product,
contamination, output-type appropriateness, powered-per-repo — whether or not a problem was
already suspected there.

**Method:** 4 parallel research passes, one per model (classifier, retriever, resolution,
synthesis), each reading the model code, eval code, ADR history, and reports, then re-computing
corrected metrics directly against the existing trained artifacts and existing eval/gold data —
no retraining, no live LLM API calls (judge-dimension recomputation replayed from committed
cassettes/checkpoint files only). Full detail: `reports/model_eval_audit.json`.

## Decision — the audit table

| Model | Current headline metric | Suitability finding | Corrected metric (honest, per repo) | Triage verdict |
|---|---|---|---|---|
| **Component classifier** | top-1 exact-match accuracy (k8s 51.4%, vscode 69.0%) | Product already treats top-3 containment as correct (`grounding.py::verify_plan_grounding`); classifier's own eval reports only top-1. ~30%/8% of test-set ground truth is a collapsed multi-label artifact. | top-3 accuracy: **k8s 82.5% CI[77.7,86.5]**, **vscode 90.4% CI[85.3,93.8]** — CIs don't overlap top-1 | **METRIC-WRONG** |
| **Similar-issue retriever** | MRR + R@5/R@10 on `gold_related.parquet` v1 (vscode only; k8s absent from README) | Advertised vscode R@5=36.7% is proxy-contaminated (v1 gold is 74% product-task for vscode, 7.6% for k8s); the already-diagnosed-internally correction (ADR-0027) never reached README. | Live-index, product-task-only R@5: **vscode 22.4% CI[17.7,28.0]** (advertised CI doesn't overlap — ~14pp inflation); **k8s UNMEASURABLE** (0 test pairs against the live corpus) | **METRIC-WRONG** (vscode fixable now; k8s blocked on new data) |
| **Resolution estimator** | point MAE + bucket accuracy vs naive + CQR coverage | k8s: metric and model both fine. vscode: model correctly self-protects (naive fallback, `BUCKET_CLASSIFIER_TRUSTED=False`) but the *documentation* is stale/conflated, and the diagnostic script that justifies the vscode decision can no longer re-verify it (tautological self-reference bug introduced by its own fix). | k8s bucket **+3.27pp CI[1.80,4.74]**, MAE **+2.1%** — both real, both honestly reported. vscode raw bucket **−22.08pp CI[−25.81,−18.02]**, MAE **−70.5%** — correctly not served. | k8s: **GENUINELY-FINE**. vscode: **METRIC-RIGHT-AS-SERVED, docs METRIC-WRONG** |
| **LLM synthesis** | 6-dimension local-judge mean, CI-regression gate (not an absolute bar) | Judge never sees provenance — structurally can't detect fabrication. A known hallucination case (vscode #311836) scores *above* its repo's mean. Floor-fail rate (worst-band on a correctness-critical dimension) is hidden by averaging. 2 new contamination leaks found (near-duplicate, CQR-calibration overlap) that ADR-0018 doesn't cover. | Floor-fail rate: **k8s 9.3% CI[4.0,19.9]**, **vscode 45.5% CI[21.3,72.0]** — nearly half of vscode's plans hit a correctness-critical worst-band behind a passable 55.8% mean | **METRIC-WRONG** |

Full per-model detail (suitability check against each of the 4 failure modes, exact source files,
CIs, sample sizes) is in `reports/model_eval_audit.json`.

## Cross-model leverage ranking

Ranked by "how much does fixing this improve the actual product," accounting for both impact and
whether the fix is available now (metric/aggregation change over existing data) or blocked on new
data collection:

1. **LLM synthesis** — highest leverage. It's the capstone, user-facing output; the current gate
   is structurally blind to the failure mode (fabrication) that would most damage trust in the
   tool, demonstrated concretely, not hypothetically. Fix is a metric/aggregation change (floor
   gate + weight grounding into pass/fail) plus a contamination purge — both over existing data,
   no retraining.
2. **Similar-issue retriever** — second-highest. Live, user-facing (5 similar issues shown per
   plan); the advertised number is inflated ~14pp on vscode, and k8s's real-world product-task
   performance is completely unmeasured, a visibility gap on a core feature of one of the
   product's two repos. vscode half fixable immediately (surface the number that already exists);
   k8s half is blocked on new data.
3. **Component classifier** — metric-wrong but lower urgency: the product's actual behavior is
   unaffected today because the downstream grounding check already correctly uses top-3
   internally. Pure reporting correction, already computed.
4. **Resolution estimator** — lowest leverage for a dedicated phase. Model and gating logic are
   already sound and self-protecting in production; the gaps are documentation lag and tooling
   hygiene, not product risk.

## Recommended fix order (Phase B+, gated on this ADR)

1. **Synthesis (B1)** — floor/weighted gate incorporating grounding; purge 2 new contamination
   leaks. No new data.
2. **Retriever, vscode half (B2)** — port the already-computed honest product-task number into
   README, replacing the proxy-contaminated MRR-led table. No new data.
3. **Classifier (B3)** — re-baseline headline to top-3 accuracy (already computed); flag the
   preprocessing label-collapse as a separate future data-quality fix.
4. **Resolution (B4)** — hygiene: refresh README, fix the diagnostic script's self-reference bug,
   correct ADR-0025's transcription error.
5. **Retriever, k8s half (Phase C, data-gated)** — cannot proceed without new data. See below.

## Escalations — corrected metrics that need new data (not actioned this phase)

- **Retriever, k8s**: zero product-task (issue→issue) test pairs exist against the *live* deployed
  index (#1–15,000). The 57 available product-stratum test pairs all fall in the unreleased
  forward-scrape range; the 72 legacy pairs that do fall in the live corpus were all assigned to
  train by the w3-retry split. The live k8s retriever's real-world product-task performance is
  currently unknown and unmeasurable without mining a new held-out, train-disjoint sample from the
  live corpus. Flagged per the vscode-underpowering-trap rule — not proceeding on zero data.
- **Synthesis, vscode**: n=11 gold triage plans give ±25–50pp CIs on any proportion (the 45.5%
  floor-fail estimate spans [21.3%, 72.0%]) — too wide for a confident absolute rate, consistent
  with the project's existing vscode data-ceiling finding (ADR-0017). This does **not** block
  shipping the floor-gate mechanism itself (usable today at current n); it's a note for future
  gold-set growth prioritization.

## Consequences

- **What changes:** nothing yet — this is audit-only. `reports/model_eval_audit.json` is the
  reference artifact for Phase B+ specs.
- **What becomes easier:** each Phase B+ fix now has a pre-diagnosed root cause, exact file
  locations, and a corrected-metric target computed against existing data — no rediscovery needed.
- **What becomes harder:** nothing structurally; the resolution model's diagnostic script
  (`scripts/w6_diagnose_resolution.py`) is currently unable to re-verify its own headline finding
  post-ADR-0025 — flagged as a B4 fix, not yet broken in a way that affects what's served.
- Per spec.md's hard rule: none of these findings were hidden or downplayed — where a metric looks
  fine only because it's measured on a proxy or an easy aggregation, this ADR says so explicitly,
  including for the two models (resolution, and the classifier's underlying model quality) that
  turned out to be substantively fine once measured correctly.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Fix each metric as its audit finding lands (no separate ADR) | Loses the cross-model leverage comparison, which is the actual strategic deliverable — knowing retrieval's k8s gap exists is less useful than knowing it ranks below synthesis for fix priority. |
| Skip re-computing corrected metrics, just flag suitability qualitatively | Spec's hard rule requires honest quantified numbers, not qualitative flags — a flattering number on the wrong metric is exactly what this audit exists to catch, and catching it requires the actual corrected number. |
| Proceed to mine new k8s retriever eval data automatically | Out of scope this phase (audit only) and a real scoping decision (scrape volume, disjointness guards) that needs the same GG sign-off prior data-growth phases got (ADR-0026/0027) — escalated, not assumed. |
