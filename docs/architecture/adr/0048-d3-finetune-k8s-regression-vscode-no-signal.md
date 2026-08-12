# ADR-0048 — D3 fine-tune results: k8s significant regression, vscode no signal (do not ship)

**Status:** Rejected (documented negative; no cutover, no HF release)
**Date:** 2026-08-12
**Decider:** Gaurav Gandhi

## Context

ADR-0047 expanded the training pools (k8s_related 264→448, vscode_duplicate 1,734→1,958) after
correcting two D1 channel-drop decisions. This is the D2 retry D2 never actually got: D2's first
run (ADR-0034) was confounded by a mean/CLS pooling mismatch and 128-token truncation, withdrawn
by the harness-correction ADR; D2's second, corrected run (ADR-0035) was underpowered (+2.0pp,
CI[-4.5,+8.5], n=200). This run uses the same corrected config (CLS pooling, MAX_LEN=256,
gradient accumulation) — verified explicitly BEFORE training, not assumed
(`reports/d3_config_verification.json`, all three checks PASS: pooling matches BGE's own
`1_Pooling/config.json` field-for-field; tokenizer identity confirmed; MAX_LEN=256 covers the
freshly-measured p95=228 on this session's actual training texts, negatives included) — on the
larger, precision-corrected pools.

## Method

`scripts/d3_train.py` (BGE-base MNRL fine-tune, same architecture/loss as D2), one measure-first
default run per task, no sweep:

| Task | Pairs | Epochs | Loss (first→last) | Train time |
|---|---|---|---|---|
| k8s_related | 448 | 4 | 1.3439 → 0.0903 | 547.6s |
| vscode_duplicate | 1,958 | 5 | 0.6170 → 0.0237 | 3,019.4s |

Both runs pace-checked at 20 steps before trusting the full run (~0.48-0.50s/step, ~5.6GB/8GB
VRAM, stable — no contention, no OOM risk) and ran detached with heartbeat monitoring
(`reports/logs/d3_train_*.log`).

`scripts/d3_eval_finetuned.py` — a corrected eval harness, NOT `scripts/d2_eval_finetuned.py`
(deliberately frozen to D1's original title-only-query, pre-ADR-0040-corpus measurement for its
own internal apples-to-apples comparison, and therefore stale relative to what this project
actually ships). Two corrections applied: (1) query text is the real, full, untruncated
title+body looked up from the processed corpus — matching production
(`triage.py::_collect_signals`) and the 2026-08-11 clean-eval methodology, not D1's title-only
measurement; (2) baseline/corpus index is `dup_index_{repo}_bge` — confirmed byte-identical
(SHA-256) to the live-serving index by the 2026-08-10 investigation — not D1's stale,
pre-ADR-0040 char-truncated `d1_full_corpus_index_{repo}_bge`. Both baseline numbers reproduce
their known-current values exactly (k8s 24.67% full-150 / 39.39% clean-66, vscode 53.50%),
confirming the harness is correctly calibrated before trusting the delta.

## Results

**k8s_related — significant, decisive regression.**

| Eval population | n | Baseline R@5 | Fine-tuned R@5 | Δ | 95% CI (paired) | Excludes zero? |
|---|---|---|---|---|---|---|
| Full 150-pair eval set | 150 | 24.67% | 15.33% | **−9.33pp** | [−14.0, −4.7] | **Yes** |
| Clean 66-pair VALID subset | 66 | 39.39% | 24.24% | **−15.15pp** | [−24.2, −7.6] | **Yes** |

R@1 and R@10 move the same direction on both populations (R@1 clean-subset: 21.2%→12.1%,
−9.1pp, CI excludes zero; R@10 clean-subset: 47.0%→31.8%, −15.15pp, CI excludes zero). MRR drops
on both (0.161→0.113 full-set, 0.282→0.173 clean-subset). Every metric, every population, moved
backward, most CIs clearing significance on the harmful side.

**vscode_duplicate — no signal, replicates D2's null result at larger scale.**

| Metric | Baseline | Fine-tuned | Δ | 95% CI (paired) | Excludes zero? |
|---|---|---|---|---|---|
| R@1 | 32.0% | 30.0% | −2.0pp | [−8.5, +4.5] | No |
| R@5 | 53.5% | 54.5% | +1.0pp | [−5.5, +7.5] | No |
| R@10 | 61.0% | 62.0% | +1.0pp | [−5.5, +7.5] | No |

MRR essentially flat (0.403→0.398). +224 more training pairs and a newly-included
higher-precision channel (`vscode_body_refs`/`body_related_ext`, 74.0% strict precision) did not
move this task past D2's original NO SIGNAL finding at 1,734 pairs — more data from a similar
precision distribution did not unlock a different result.

## Decision: REJECTED — no cutover, no HF release, either task

Neither task clears the ship bar (meaningful lift, CI clearly excluding zero). k8s is not merely
"doesn't clear the bar" — it's a statistically confirmed, actively harmful regression. This is a
**valid** negative, not another broken-measurement finding: config was verified before training
(pooling/seq-length/tokenizer all PASS), the eval harness reproduces known-current baseline
numbers exactly on both populations, and the leakage guard confirms zero issue-level overlap
between the expanded training pool and either held-out eval set.

## Mechanism — why k8s regressed while vscode held flat

Two candidate mechanisms, not mutually exclusive, both consistent with the evidence:

**1. Training-pair precision dilution.** k8s's expanded pool measures ~42-54% precision under
the strict rubric (ADR-0047) — meaning roughly half of the "positive" pairs the MNRL loss trains
on are not actually topically related (the dominant failure mode being `EXCLUDE_OTHER`: two
issues that share a citation but are substantively about different things, per this session's
blind-labeling reasons). Training a contrastive objective to pull unrelated embeddings together
on ~half the signal is a plausible direct cause of embedding-space corruption, not just a failure
to improve. vscode's pool (74-77% precision on the newly-included/dup_comment channels) has
roughly half the noise rate — consistent with the two tasks' very different outcomes if this is
the operative mechanism.

**2. Candidate-distribution mismatch (ADR-0034's untested hypothesis, now more credible).** ADR-
0034 flagged, but never tested, that training exposes the model to hard negatives mined from a
CANDIDATE POOL RESTRICTED to training-pool issues only (k8s: 792 issues) while eval scores
against the full live corpus (~30,000 issues) — fine-tuning may sharpen discrimination against
the narrow training-time negative distribution at the cost of general full-corpus retrieval
quality. k8s's restricted pool (792 issues) is proportionally much narrower relative to its full
corpus (30,000, ~2.6%) than vscode's (3,338 of ~13,000+, ~25%+) — if this mismatch is the
mechanism, k8s's much narrower ratio predicts a much larger distortion, matching the much larger
regression observed. This ADR does not distinguish between the two mechanisms (both are
consistent with the data; disentangling them would need a dedicated experiment — e.g. training
with full-corpus negative sampling instead of restricted-pool mining) — reported as two live,
unresolved candidates, not resolved to one.

**Loss curves are informative but not dispositive on their own.** k8s's final loss (0.090) is
higher than the near-zero (0.045) that flagged D2's original run as a memorization artifact — this
run does not show the same textbook overfitting signature. The regression is real gradient signal
pulling the model in a genuinely wrong direction (candidate mechanisms above), not simple
memorization of a tiny dataset.

## Consequences

- **What changes:** nothing in production. `src/triage_iq/api/loader.py` continues to load only
  `dup_index_{slug}_bge`; neither `d3_finetuned_k8s_related` nor `d3_finetuned_vscode_duplicate`
  is referenced anywhere in serving code. Confirmed by inspection, same check ADR-0034/0035
  performed.
- **What becomes easier:** the "more/cleaner training data alone fixes this" hypothesis is now
  closed with an honest, correctly-measured answer for k8s (actively harmful) and vscode (no
  effect at this scale increase) — a future session doesn't need to re-try a bigger-pool fine-tune
  without first addressing one of the two mechanisms above.
- **What becomes harder:** nothing new shipped; retrieval's README section gains another
  rejected-lever entry. The value here, consistent with this project's pattern, is the rigor of
  a correctly-measured negative over an unmeasured "maybe."
- **Open, not pursued here:** distinguishing the two candidate mechanisms (precision dilution vs.
  candidate-distribution mismatch) would need a dedicated experiment — e.g. re-training k8s on
  only the highest-confidence sub-channel (`k8s_forward_scrape`/`body_related`, the strongest-
  pattern, most declarative citation type) to isolate precision, or re-mining hard negatives from
  the full corpus instead of the restricted training-pool candidate set to isolate distribution
  mismatch. Neither attempted here — this ADR reports the result and the mechanism candidates, it
  does not chase a fix without a fresh escalation.
- **Cost:** $0. Local RTX 3070, ~9min (k8s) + ~50min (vscode) GPU time, uncontended.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Ship vscode_duplicate anyway (positive point estimate, +1.0pp) | Directly against the project's ship bar and ADR-0035's own explicit lesson: a small, non-significant positive point estimate is the same "don't ship" call as a negative one, just with more optimistic-looking noise. |
| Sweep hyperparameters on k8s before concluding regression | Measure-first discipline (this project's standing pattern, ADR-0034's own precedent): the first default run shows a clear, CI-confirmed signal (harmful), not an ambiguous one that would justify a sweep. Sweeping around a confirmed regression risks landing on a config that clears significance by chance. |
| Immediately chase the distribution-mismatch fix (full-corpus negative sampling) | Out of scope for this run's escalation budget — a new experiment needing its own scoping, not a natural extension of "measure the retry." Flagged as the concrete next step if this thread is revisited. |
