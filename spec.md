# Project Spec: TriageIQ — Per-Model Evaluation Audit + Performance Diagnosis

## Goal

Across this project, metric-appropriateness problems have surfaced ACCIDENTALLY, one model at a
time: retrieval's k8s metric measured a PROXY task (PR→issue, not the product issue→issue);
resolution's intervals were non-diagnostic and vscode served a worse-than-naive classifier;
component_match had gold contamination. Never a DELIBERATE, systematic audit of "is each model
evaluated on a metric suited to what it actually does, and is it actually good on the right metric."

This iteration does that audit — for ALL FOUR models (component classifier, similar-issue retriever,
resolution estimator, LLM synthesis) — then, where the corrected metric reveals a real performance
gap, diagnoses and (in gated follow-on phases) fixes it. Audit FIRST: you cannot fix performance
until you're measuring the right thing, and wrong-metric problems hide under apparently-fine numbers.

The deliverable of Phase A (this spec) is the AUDIT: per model, what task it does, what metric it's
currently evaluated on, whether that metric is SUITABLE, what the performance is on the CORRECT
metric, and a triage — {metric-wrong → fix eval}, {metric-right-but-weak → diagnose+fix model},
{genuinely fine}. Fix phases (B+) are separate, gated on the audit findings.

## Current state (the 4 models + their current evals)

- **Component classifier** (TF-IDF + LR, temp-calibrated): multi-label-ish component prediction.
  Current eval: accuracy, calibration (ECE), component_match judge dimension. Known: gold
  contamination was fixed (load_eval_splits cross-check); calibration was corrected (ADR-0004).
- **Similar-issue retriever** (BGE + FAISS): find related issues. Current eval: recall@k on
  gold_related. Known: k8s gold is 92% PR-query (measures PR→issue, not product issue→issue);
  vscode corpus grown (v2, Phase 2). Fine-tune held pending product-task power.
- **Resolution estimator** (LightGBM quantile + bucket + CQR): predict resolution time/bucket.
  Current eval: MAE, coverage, bucket accuracy vs naive. Known: intervals non-diagnostic
  (ADR-0021/0023); k8s bucket beats naive (CI), vscode on naive fallback (Phase 1).
- **LLM synthesis** (Groq llama-3.1-8b): generate the triage plan. Current eval: local qwen3:8b
  judge, 6 dimensions, mean-band gate. Known: attribution 100% grounded; grounding verifier live.

## Scope

### Phase A — the audit (this iteration; the rest gated on it)

**For EACH of the 4 models, produce:**
1. **Task statement**: what does this model actually do, and what's the PRODUCT use case (what does
   a user actually need from it)?
2. **Current metric**: what metric(s) is it evaluated on today, on what data.
3. **Metric-suitability verdict**: is the current metric SUITABLE for the task + product use case?
   Specifically check the failure modes this project has already hit:
   - Does the metric measure the PRODUCT task or a PROXY? (retrieval's PR→issue lesson — check ALL
     models for this, not just retrieval.)
   - Is the eval data disjoint / uncontaminated? (the gold-contamination lesson.)
   - Is the metric appropriate to the model's OUTPUT type? (e.g. exact-accuracy for a classifier
     where adjacent labels are near-correct may understate it — is top-k or hierarchical accuracy
     more suitable? A point-MAE for resolution when buckets are what's used?)
   - Is the eval powered (n) on the task that matters, per repo? (the vscode underpowering lesson.)
4. **Performance on the CORRECT metric**: re-evaluate on the suitable metric (if different from
   current). Report the honest number, per repo, with CI where it's a comparison.
5. **Triage verdict** per model, one of:
   - METRIC-WRONG → the eval needs fixing (measure the right thing); performance may look different.
   - METRIC-RIGHT-BUT-WEAK → real performance gap; needs diagnose+fix (a gated follow-on phase).
   - GENUINELY-FINE → measured right, performs adequately, no action.

**Cross-model synthesis**: rank the 4 by "how much does fixing this improve the actual product,"
and recommend which model(s) get a deep diagnose+fix phase next, with the expected leverage.

### Phases B+ (gated on the audit — NOT this iteration)

Per the audit's triage, follow-on phases (each its own spec, escalated): fix a wrong metric,
or diagnose+fix a weak model. Sequenced by the audit's leverage ranking. Do NOT start these until
the audit is reviewed and the human picks the order.

### Out of scope (this iteration)

- No model retraining or fixing yet (audit FIRST — Phase A is diagnosis, not treatment).
- No shipping the held Phase 2 fine-tune (separate decision, its own data gate).
- No new features / no live cutover.
- No reopening the closed eval-integrity mechanics (the gate infra is sound; this audits the
  MODELS' metrics, not the gate).

## Tech stack

- Existing Python + the existing eval harness + committed eval data. Re-evaluation on corrected
  metrics uses existing artifacts (no retraining). scipy for CIs. Local judge if synthesis re-eval
  needs it (zero-cost). No new deps without escalation.

## Autonomy & escalation

CC runs the full audit autonomously. Escalate ONLY:
1. **The completed audit** — the per-model table (task / current metric / suitability / corrected
   performance / triage verdict) + the cross-model leverage ranking + which model(s) to fix first.
   This is the strategic output; the human picks the fix order from it.
2. Any point where re-evaluating on a "corrected" metric requires new data/labeling (flag it, don't
   silently proceed on insufficient data — that's the vscode-underpowering trap).

## Hard rules

- Audit is HONEST: if a model looks fine only because it's measured on a proxy/easy metric, SAY SO
  (the retrieval PR→issue lesson — apply it to every model). A flattering number on the wrong
  metric is the failure mode this audit exists to catch.
- Per-repo, powered, disjoint — the same discipline: don't claim on underpowered/contaminated evals;
  vscode indicative where n is small.
- Corrected metrics stand on their own; if a corrected metric changes the story vs the current one,
  that's a finding to disclose (like the k8s-retrieval relabel), not to hide.
- No fixing in this phase — audit and triage only. Branch only (`analysis/model-eval-audit`);
  I merge. Zero-cost, local judge if needed. Claude Max — never ANTHROPIC_API_KEY. Don't touch
  aetherart-497918.

## Success criteria

- All 4 models audited: task, current metric, suitability verdict, corrected-metric performance,
  triage verdict — in one comparable table.
- Every model checked against the 4 known failure modes (proxy-vs-product, contamination, output-type
  appropriateness, powered-per-repo).
- Cross-model leverage ranking + recommended fix order.
- reports/model_eval_audit.json + ADR-0028 documenting the audit + recommendations.
- Escalated for the human to pick the fix sequence (Phases B+).

## Build order (CC autonomous)

1. Component classifier: task, current metric (accuracy/ECE/component_match), suitability (is exact
   accuracy right, or does hierarchical/top-k fit better? is the eval disjoint post-contamination-fix?),
   corrected performance, verdict.
2. Retriever: task, current metric (recall@k), suitability (PR→issue proxy already known — quantify
   product-task performance honestly per repo), corrected performance, verdict.
3. Resolution: task, current metric (MAE/coverage/bucket), suitability (is point-MAE right when
   buckets are used? is bucket-accuracy-vs-naive the product metric?), corrected performance, verdict.
4. Synthesis: task, current metric (judge dimensions), suitability (do the 6 dimensions measure what
   synthesis is for? is the judge a suitable evaluator?), corrected performance, verdict.
5. Cross-model synthesis: leverage ranking, recommended fix order.
6. ESCALATE the audit. ADR-0028.
```

