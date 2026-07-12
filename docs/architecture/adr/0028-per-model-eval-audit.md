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
| **Similar-issue retriever** | MRR + R@5/R@10 on `gold_related.parquet` v1 (vscode only; k8s absent from README) | Advertised vscode R@5=36.7% is proxy-contaminated (v1 gold is 74% product-task for vscode, 7.6% for k8s); the already-diagnosed-internally correction (ADR-0027) never reached README. | Live-index, product-task-only R@5: **vscode 22.4% CI[17.7,28.0]** (n=254; advertised CI doesn't overlap — ~14pp inflation); **k8s 23.5% CI[18.4,28.5]** (n=277; corrected 2026-07-12, see below — originally called UNMEASURABLE, that framing was wrong) | **METRIC-WRONG** (both repos fixed now, no new data needed) |
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
2. **Similar-issue retriever** — second-highest at the time this ranking was written. Live,
   user-facing (5 similar issues shown per plan); the advertised number was inflated ~14pp on
   vscode, and k8s's real-world product-task performance was completely unmeasured, a visibility
   gap on a core feature of one of the product's two repos. Both halves have since been fixed
   with no new data needed (2026-07-12 correction below): vscode's honest number was surfaced,
   and k8s's turned out to be measurable now, not blocked. With both numbers in, this model's
   *leverage ranking* (as scored here — reporting-fix cost) turns out to understate its actual
   priority: the corrected numbers show it is the weakest-performing model in the pipeline in
   absolute terms (~23% R@5, both repos) — a product-quality finding this ranking axis wasn't
   designed to surface.
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

## Decision record (2026-07-11, GG) — Phase B1-B4 executed, Phase C banked

B1 (`fix/b1-contamination-leaks`, PR #29), B2 (`fix/b2-honest-reporting`, PR #30), B3
(`fix/b3-synthesis-quality-metrics`, PR #31), and B4 (this ADR's own resolution hygiene items)
have been executed, in that order, per this ADR's recommended fix order. Details in each PR and
in `docs/investigations/gold-set-leakage.md` Sec 5 (B1) / `reports/eval_baseline.json`'s
`synthesis_quality_floor` (B3).

**Phase C (retriever, k8s half) is explicitly BANKED, not actioned:** k8s's live retriever
product-task performance is unmeasurable without new held-out product-task (issue→issue) gold —
the same related-pair-mining gap that held the Phase 2 W3-retry fine-tune (ADR-0027). This is a
data-collection **decision**, not a task with a fix order: there is nothing to execute against
zero data. Revisit alongside the Phase 2 fine-tune data question (ADR-0027's "concrete unblock" —
~700 more product-task pairs, needing related-pair mining at scale beyond the dup-comment
channel) rather than as a standalone eval-audit follow-up.

**Correction (2026-07-12, ADR-0030) — the "unmeasurable" framing above was wrong, not the
underlying gap.** ADR-0030's feasibility analysis found that the "zero product-task test pairs"
claim conflated two different things: the w3-retry train/val/test split (which exists to prevent
leakage for the *separate, unshipped* fine-tuned embedder) was being applied to gate what's
measurable against the *live* index — but the live-serving retriever
(`dup_index_kubernetes_kubernetes_bge`, off-the-shelf `BAAI/bge-base-en-v1.5`, never trained on
any gold pair) carries zero leakage risk from that split. Every product-stratum pair whose query
and target both fall in the live index's number range (#1–15,002) is usable for measurement
regardless of its split label — 277 of 776, not the 0 the split-label framing implied. Re-run
(`scripts/phaseC_k8s_live_product_eval.py`, same method and bootstrap as the vscode number
above): **k8s live product-task R@5 = 23.5% CI[18.4,28.5]**, n=277 — statistically indistinguishable
from vscode's 22.4% CI[17.7,28.0] (n=254; CIs almost entirely overlap). This is now a real,
measured performance number, not a data gap: the live retriever finds the genuinely-related issue
in the top-5 roughly a quarter of the time on **both** repos — **the single lowest absolute score
of any model in this audit, and product-task retrieval is now confirmed the weakest model in this
pipeline**, invisible until this correction because neither repo had ever been measured on the
actual product task against the actual live index before. Read as a performance signal rather
than a reporting gap: ADR-0030 settles the Phase 2 fine-tune (ADR-0027, product-task deltas
+3.5pp k8s / +3.2pp vscode, both CIs crossing zero) as **NO-GO on both repos, decided on value —
not data-availability.** k8s's mining ask was disproportionate anyway, but that's not the
operative reason; a successful gate on either repo would still ship a retriever missing the
related issue roughly 3 times out of 4 — neither fine-tune is a near-miss awaiting data, both are
marginal against a weak baseline. The open question is not "close the data gap" (there wasn't
one, or wasn't one worth closing) but whether retrieval quality itself needs work (hybrid
BM25+dense, reranking, a stronger base embedder) rather than a few points of fine-tune gain — not
actioned this ADR or ADR-0030, deliberately flagged for a future, separate retrieval-quality
phase. Full detail: `docs/architecture/adr/0030-phaseC-product-task-feasibility.md`,
`reports/phaseC_k8s_live_product_eval.json`.

**Tracked for later, not acted on now:** B3's `fabrication_rate` is informational-only by
deliberate GG decision (2026-07-11) — the right conservative start. This ADR's own core finding
is that fabrication is the failure mode that most misleads a human triage engineer, and it
currently scores *above* the mean (vscode #311836). The eventual direction is promoting
`fabrication_rate` from informational to a hard, blocking gate once real-world rate has been
observed for a while — the same path the grounding check itself took (informational in ADR-0015,
still informational here, promotion criteria not yet defined). Not a decision to make yet;
revisit once there's an observation window.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Fix each metric as its audit finding lands (no separate ADR) | Loses the cross-model leverage comparison, which is the actual strategic deliverable — knowing retrieval's k8s gap exists is less useful than knowing it ranks below synthesis for fix priority. |
| Skip re-computing corrected metrics, just flag suitability qualitatively | Spec's hard rule requires honest quantified numbers, not qualitative flags — a flattering number on the wrong metric is exactly what this audit exists to catch, and catching it requires the actual corrected number. |
| Proceed to mine new k8s retriever eval data automatically | Out of scope this phase (audit only) and a real scoping decision (scrape volume, disjointness guards) that needs the same GG sign-off prior data-growth phases got (ADR-0026/0027) — escalated, not assumed. |
