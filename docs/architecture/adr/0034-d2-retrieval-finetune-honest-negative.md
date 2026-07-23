# ADR-0034 — D2 Retrieval Fine-Tune: Documented Honest Negative (vscode-duplicate)

**Status:** Accepted (documented negative; no cutover, no HF release)
**Date:** 2026-07-23
**Decider:** Gaurav Gandhi (measure-first run + diagnostic leg executed autonomously by CC per spec.md, both escalated)

---

## Context

D1 (ADR-0033) produced clean, hand-verified, issue-level-disjoint training pools and held-out eval
sets, with an honest baseline: vscode_duplicate R@5 43.5% [37.0, 50.5] (n=200, 1,734 clean training
pairs), k8s_related R@5 9.3% [5.3, 14.0] (n=150, 264 clean training pairs — the thinnest pool in the
project). D2's mandate (spec.md) was to fine-tune BGE-base locally on this clean data, measure-first
(one default run before any sweep), and ship only on a MEANINGFUL lift — the W3 fine-tune
(ADR-0027) was explicitly held at a marginal +3.5pp (k8s) / +3.2pp (vscode) product-task lift, both
CIs crossing zero, specifically to avoid shipping a marginal-polish result. This ADR holds D2 to the
same bar.

Training ran locally on GG's RTX 3070 (8GB VRAM) as a detached process — GCP GPU provisioning was
attempted first (billing relink, quota increases, a borrowed-project Spot GPU) and abandoned after
exhausting on-demand and Spot capacity across three regions; local execution was both cheaper and,
once the GPU was free, faster to get a trustworthy answer from. Total cost: **$0**.

## Method

**1. Leakage guard — reasserted before every run (3×), PASSED every time**: vscode_duplicate
train=1,734 pairs / 3,008 issues vs. eval=200 pairs / 391 issues, **overlap=0**. This is D1's
payoff directly: the negative result below is believable *because* contamination is asserted
programmatically before each run, not assumed.

**2. Measure-first run** (`d2_train.py --task vscode_duplicate`, defaults): BGE-base MNRL fine-tune,
8,660 anchor/positive/negative triplets (in-batch + training-pool-only mined hard negatives), 5
epochs, lr=2e-5, batch=16, seed=42, temperature=0.05. Loss: 0.678 → 0.117 → 0.062 → 0.048 →
**0.045** — a near-zero final loss, the textbook overfitting signature.

| Metric | Baseline | Fine-tuned | Δ | 95% CI (paired) | Excludes zero? |
|---|---|---|---|---|---|
| R@1 | 22.5% | 22.0% | −0.5pp | [−6.0, +4.5] | No |
| **R@5** | **43.5%** | **38.5%** | **−5.0pp** | **[−11.0, +0.5]** | No |
| R@10 | 47.5% | 46.5% | −1.0pp | [−6.0, +4.0] | No |
| MRR | 0.313 | 0.289 | — | — | — |

Every point estimate moved backward. Not the W3 marginal-*lift* trap — a marginal-*loss* result
instead, numerically negative everywhere but not (yet) statistically confirmed.

**3. Diagnostic leg — a hypothesis test, not a sweep leg**: tests directly whether overfitting
explains the regression. 2 epochs, lr=1e-5 (the textbook anti-overfit correction: fewer epochs,
lower learning rate). Loss: 0.742 → **0.255** — training ends 5.7× higher / far less memorized than
the first run.

| Metric | Baseline | Fine-tuned | Δ | 95% CI (paired) | Excludes zero? |
|---|---|---|---|---|---|
| R@1 | 22.5% | 21.0% | −1.5pp | [−6.0, +3.5] | No |
| **R@5** | **43.5%** | **33.5%** | **−10.0pp** | **[−16.0, −4.5]** | **Yes** |
| R@10 | 47.5% | 43.5% | −4.0pp | [−9.5, +1.0] | No |
| MRR | 0.313 | 0.266 | — | — | — |

If overfitting were the mechanism, correcting for it should have narrowed or reversed the
regression. It widened it — R@5 dropped a further 5pp — and the paired CI now **excludes zero**: a
statistically confirmed regression, worse than the first run, despite meaningfully less-memorized
training.

**4. k8s_related (264 training pairs)**: NOT attempted. See Decision §4.

## Decision

1. **D2's vscode_duplicate fine-tune is REJECTED — no cutover, no HF release.** Off-the-shelf
   `BAAI/bge-base-en-v1.5` (the currently-served model, unchanged) beats both fine-tune
   configurations tested on the held-out eval.

2. **Overfitting is REJECTED as the mechanism.** The anti-overfit diagnostic (fewer epochs, lower
   LR, 5.7× higher final loss) made the regression *larger and statistically significant*, not
   smaller. This is what elevates the finding from "inconclusive, might still just need better
   hyperparameters" to a confirmed negative: the leading alternative explanation was directly
   tested and ruled out, not left open for someone to re-litigate later.

3. **Leading remaining hypothesis — UNTESTED, not pursued further**: train/eval
   candidate-distribution mismatch. Training exposes the model to ~5 mined hard negatives per
   anchor-positive pair (8,660 triplets total); the held-out eval scores against the full corpus of
   13,315 vscode issues. Fine-tuning may sharpen the embedding space specifically against the
   narrow negative distribution it was shown, at the cost of the general semantic quality
   `bge-base` ships with out of the box — training-time discrimination and eval/deployment-time
   retrieval may simply be different distributions. What would test this, without committing to
   running it: (a) in-batch-only or full-corpus negative sampling during training instead of the
   mined-hard-negative set, or (b) evaluating the fine-tuned model against the same restricted
   candidate pool used in training, to check whether the regression is specific to the distribution
   shift or holds even on the training-like distribution.

4. **k8s_related (264 pairs) is explicitly NO-GO, not merely unattempted.** If fine-tuning
   regresses full-corpus retrieval on 1,734 clean, well-mined vscode pairs, a training pool 6.6×
   smaller is a strictly worse bet by the same mechanism (even less data to characterize the true
   negative distribution, if that is the cause). Spending GPU time to re-confirm the same failure
   mode on thinner data is not a defensible use of the measure-first budget.

5. **HF release (D3): declined on evidence, not deferred.** A fine-tune that loses to the base
   model it started from has no USP to publish under any framing.

6. **Retrieval quality is now a characterized limit across FIVE independently-tried levers, not a
   gap awaiting the next idea:** hybrid BM25+RRF fusion (ADR-0031, CI crosses zero), a pretrained
   cross-encoder reranker (ADR-0031, quality regresses + 190–330× latency), a stronger pretrained
   embedder (ADR-0031, CI crosses zero), the W3 in-domain fine-tune on noisy pairs (ADR-0027, held
   at a marginal +3.5pp/+3.2pp, both CIs cross zero), and this D2 in-domain fine-tune on clean,
   leakage-asserted pairs (confirmed regression, CI excludes zero on the diagnostic leg). BGE-base
   off-the-shelf remains the shipped retriever, unbeaten across all five.

## Consequences

- **What changes**: nothing in production. Confirmed by inspection: `src/triage_iq/api/loader.py`
  loads only `dup_index_{slug}_bge` (the pre-existing served index); `SUPPORTED_MODELS` in
  `src/triage_iq/models/similar_issues.py` maps `"bge"` to the HF hub id
  `BAAI/bge-base-en-v1.5`, never a local fine-tuned path. Neither `d2_finetuned_vscode_duplicate/`
  nor its `_lowepoch` variant is referenced anywhere in serving code.
- **What becomes easier**: the retrieval-fine-tuning question is now closed with a clean,
  causally-supported answer instead of an open "maybe it would help" thread — a future session
  doesn't need to re-try fine-tuning from scratch without first addressing the distribution-mismatch
  hypothesis above.
- **What becomes harder**: nothing new shipped. The README's retrieval section is now a five-lever
  rejection list — a more honest but less flattering headline than a shipped fine-tune would have
  been. The project's value proposition here is the rigor of the measurement, not the retrieval
  number itself.
- **Cost**: $0. Local RTX 3070, shared with GG's AetherArt workload (never touched or interrupted).
  Multiple GCP GPU-provisioning paths were evaluated and abandoned before falling back to local — no
  GPU-hour was ever billed on GCP.
- **Open thread, explicitly not pursued**: the train/eval distribution-mismatch hypothesis
  (Decision §3) is untested. If a future session revisits in-domain fine-tuning, that hypothesis —
  and a fix for it — is the prerequisite, not another hyperparameter sweep on the current setup.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Run a full hyperparameter sweep after the first regression | Measure-first discipline (spec.md): a sweep is justified by a signal worth chasing; the first run showed none. Sweeping around a regression risks landing on a config that clears significance by chance, not by a real fix. |
| Treat the first (5ep/lr=2e-5) result as inconclusive and stop, without testing overfitting | Would leave the leading alternative explanation unaddressed, making the negative weaker and reversible by "just needs fewer epochs" speculation. The diagnostic leg closed that gap for ~1 GPU-hour and converted a soft negative into a confirmed one. |
| Attempt k8s_related anyway, since it was pre-registered as a valid experiment regardless of outcome | The vscode_duplicate result and its confirmed mechanism generalize as a reason not to spend GPU time re-running the same likely failure mode on a strictly thinner, higher-overfit-risk pool. Pre-registration commits to reporting an outcome honestly, not to running an experiment whose expected value has since dropped to near zero. |
| Chase the distribution-mismatch hypothesis immediately (in-batch/full-corpus negatives) | Out of scope for this measure-first-plus-one-diagnostic budget; a new experiment requiring its own escalation, not a natural extension of "confirm or reject overfitting." |
