# ADR-0049 — k8s D3 regression: mechanism disentanglement (neither candidate confirmed; a third surfaces)

**Status:** Accepted (negative result; no cutover, no HF release; supersedes neither ADR-0048 decision)
**Date:** 2026-08-12
**Decider:** Gaurav Gandhi

## Context

ADR-0048 found the D3 k8s_related fine-tune to be a confirmed, CI-excludes-zero regression
(clean 66-pair subset R@5: 39.39% -> 24.24%, -15.15pp, CI[-24.2,-7.6]) and named two candidate
mechanisms without disentangling them: (A) training-pair precision dilution (the 448-pair pool is
~42-56% precision under the strict rubric; training a contrastive objective on ~half-invalid
pairs could corrupt the embedding space) and (B) hard-negative candidate-pool vs. full-corpus-eval
distribution mismatch (negatives are mined from a 792-issue training-pool-restricted candidate
space; eval ranks against the full ~30,000-issue corpus). Per GG's instruction and rule 101c (a
strong, correctly-measured negative is diagnosable, not a ceiling to accept without exhausting the
candidate mechanisms), this ADR runs the specified tests in order: a zero-cost diagnostic on the
already-trained model, then Candidate A (retrain on a blind-labeled high-precision subset), then
Candidate B (retrain with full-corpus-mined negatives instead of restricted-pool mining).

## Method and results

### Zero-cost diagnostic: does the D3 model prefer the pairs the strict rubric excluded?

If precision dilution is the operative mechanism, the model should have improved (or held flat)
on the pairs the strict rubric excludes (structurally invalid, but trained on as if valid
elsewhere in the pool) while degrading on VALID pairs. Re-scored the already-trained D3 k8s model
(no retraining) on the 84 EXCLUDE-labeled pairs from `reports/track2_k8s_clean_eval.json`
(`scripts/d3_diagnose_precision_dilution.py`, reusing `d3_eval_finetuned.py`'s hit-vector logic):

| Population | n | Baseline R@5 | Trained R@5 | Delta | CI95 (paired) | Excludes zero? |
|---|---|---|---|---|---|---|
| VALID | 66 | 39.39% | 24.24% | -15.15pp | [-24.2, -7.6] | Yes |
| EXCLUDE (all 3 reasons) | 84 | 13.10% | 8.33% | -4.76pp | [-9.5, -1.2] | Yes |

**Result: no directional support for pure dilution.** The model degraded on EXCLUDE too, not the
"improved on invalid, degraded on valid" signature dilution predicts. Read in relative terms
(VALID: -15.15pp off a 39.4% base = -38% relative; EXCLUDE: -4.76pp off a 13.1% base = -36%
relative), the two populations lost almost identical *proportions* of their baseline recall —
consistent with a general embedding-space effect hitting both populations alike, not a selective
pull toward exactly the pairs the training data mistakenly treated as positive.

### Candidate A: retrain on a blind-labeled high-precision-only subset

Extracted all 448 k8s_related training pairs into blind form (`scripts/d3a_extract_pool_blind.py`
— query+target title/body only, no channel/source/precision field) and dispatched 15 independent
labeling batches of ~30 pairs each (matching this project's own precedent — ADR-0033, the 2026-
08-11 clean-eval build) against the same pre-registered rubric used to build the clean 66-pair
eval subset (VALID / EXCLUDE_UMBRELLA / EXCLUDE_CAUSAL_ONLY / EXCLUDE_OTHER). All 448 pair_ids
labeled exactly once, zero missing, zero malformed labels
(`reports/d3a_pool_labeled_batch_1..15.json`).

**Full-census pool precision: 253/448 = 56.5% VALID** — notably higher than ADR-0047's n=50
per-channel sample estimates (42-54%), a real full-census-vs-sample discrepancy worth noting but
not chased further here (both are legitimate measurements at different sample sizes; the full
census is the more precise number and is what actually built the retrain pool). One incidental
finding from the batch labelers: a real, previously-undocumented failure mode —
**cross-repo issue-number collisions**, where a mining regex matched a citation to a *different*
GitHub repo (docker/docker, kubernetes/heapster, golang/go, openshift/origin, coreos/rkt) whose
issue number coincidentally also exists, unrelated, in kubernetes/kubernetes — labeled
`EXCLUDE_OTHER`, several dozen instances across batches.

Filtered to the 253 VALID pairs (`reports/mining_precision_train_pool_k8s_related_valid_subset.json`),
re-asserted disjointness (trivially still disjoint — a subset of an already-disjoint pool), mined
hard negatives from the same restricted-pool candidate space as D3 (470 issues, unchanged method),
verified seq-length (p95=192, well under MAX_LEN=256, PASS) and trained with D3's identical
hyperparameters (CLS pooling, MAX_LEN=256, lr=2e-5, epochs=4, batch=16, wd=0.05) — the only
variable changed is pool precision (56.5% VALID-only vs. the original pool's 56.5%-background
average, i.e. this pool IS purely the VALID pairs, 100% by construction). Pace-checked at 20 steps
(0.503s/step, ~5.6GB/8GB VRAM, no contention) before trusting the full run.

| Task | Pairs | Epochs | Loss (first->last) | Train time |
|---|---|---|---|---|
| k8s_related_valid_subset | 253 | 4 | 0.8557 -> 0.1161 | 310.3s |

| Eval population | Baseline R@5 | Trained R@5 | Delta | CI95 (paired) | Excludes zero? |
|---|---|---|---|---|---|
| Clean 66-pair VALID subset | 39.39% | 27.27% | **-12.12pp** | [-22.7, -1.5] | **Yes** |
| Full 150-pair set | 24.67% | 17.33% | -7.33pp | [-13.3, -2.0] | Yes |

**Result: smaller, but the CIs overlap heavily with D3's original -15.15pp** ([-24.2,-7.6] vs.
[-22.7,-1.5]) — the ~3pp point-estimate improvement is not itself statistically distinguishable
from noise. The regression survives essentially intact even on a purely, blind-labeled
100%-VALID training pool. **Precision dilution is not the dominant mechanism.**

### Candidate B: retrain with full-corpus-mined negatives instead of restricted-pool mining

Same 448-pair pool as D3 (isolating this one variable), but hard negatives mined from the full
live corpus (29,717 candidate issues = 29,994 corpus issues minus the 277 eval-set issues,
explicitly excluded to avoid a new leakage path — the original restricted-pool mining was
disjoint from eval by construction since its candidate space *was* the training pool;
full-corpus mining removes that implicit protection, so eval-set issues are excluded explicitly
instead, `scripts/d3_mine_train_negatives.py --full-corpus-negatives`). Encoding ~30k issues used
GPU (the restricted-pool default stays on CPU as before; full-corpus embedding on CPU would take
close to two hours, an unjustified wait for a deterministic forward pass). Same hyperparameters
as D3 otherwise.

| Task | Pairs | Epochs | Loss (first->last) | Train time |
|---|---|---|---|---|
| k8s_related_fullcorpus_negs | 448 | 4 | 1.6897 -> 0.0591 | 549.4s |

| Eval population | Baseline R@5 | Trained R@5 | Delta | CI95 (paired) | Excludes zero? |
|---|---|---|---|---|---|
| Clean 66-pair VALID subset | 39.39% | 19.70% | **-19.70pp** | [-31.8, -7.6] | **Yes** |
| Full 150-pair set | 24.67% | 15.33% | -9.33pp | [-15.3, -3.3] | Yes |

R@1 (valid-66): 21.2%->10.6%, -10.6pp, CI excludes zero. R@10 (valid-66): 47.0%->25.8%, -21.2pp,
CI excludes zero. MRR: 0.282->0.150 (the worst of all three runs).

**Result: full-corpus negative mining made the regression WORSE, not better — the opposite of
what the candidate-pool-mismatch hypothesis predicts.** If restricted-pool mining were teaching
the model a negative distribution unlike eval's full-corpus candidate space, matching that
distribution should have reduced or eliminated the regression. It didn't; it deepened it by
~4.5pp beyond D3's original result. **This is evidence against Candidate B as framed, not for
it** — full-corpus mining does not fix the problem, and on this run's evidence, actively
compounds it.

## Cross-run comparison and a candidate unifying observation (not confirmed, flagged honestly)

| Run | Pool | Negatives | Final loss | Clean-66 R@5 delta | CI95 |
|---|---|---|---|---|---|
| D3 original (ADR-0048) | 448 pairs, 56.5% VALID | restricted (792-issue) | 0.0903 | -15.15pp | [-24.2,-7.6] |
| D3a Candidate A | 253 pairs, 100% VALID | restricted (470-issue) | 0.1161 | -12.12pp | [-22.7,-1.5] |
| D3a Candidate B | 448 pairs, 56.5% VALID | full-corpus (29.7k-issue) | **0.0591** | **-19.70pp** | [-31.8,-7.6] |

Across these three runs, final training loss and eval-harm move together: the lowest-loss run
(Candidate B, 0.0591 — the closest of the three to D2's original 0.045 memorization-artifact
threshold, ADR-0048) produced the worst regression; the highest-loss run (Candidate A, 0.1161)
produced the least-bad regression. This is **n=3, directionally suggestive, not statistically
established** — reported as an observation to weigh, not a confirmed third mechanism. It's
consistent with a reading where k8s's fundamental constraint is dataset scale, not pair quality
or negative-source distribution specifically: at 253-448 pairs, a contrastive fine-tune can
converge to near-zero training loss on this tiny, idiosyncratic pool while systematically losing
the base BGE checkpoint's general-purpose k8s-domain retrieval competence (a small-data
overfitting/catastrophic-forgetting shape) — and Candidate B's harder, more diverse negatives
plausibly accelerated exactly this by giving the contrastive objective a sharper, easier-to-fit
gradient signal, not a more distribution-matched one. A specific, testable sub-hypothesis this
also surfaces: full-corpus hard-negative mining is more likely to surface **false negatives**
(candidates that are actually topically related to the anchor but not labeled positive by the
regex-mining heuristic) than the narrow 792-issue restricted pool is — training a contrastive loss
to push away from a true near-duplicate would directly corrupt useful structure. Not measured
here; flagged as the mechanistically cleanest next test if this thread is revisited.

## Decision: REJECTED, all three configurations — no cutover, no HF release

None of the three k8s_related fine-tunes (D3 original, D3a Candidate A, D3a Candidate B) clears
the ship bar. Candidate A is the least-bad but still a confirmed, CI-excludes-zero harmful
regression. **Neither of ADR-0048's two named candidate mechanisms is confirmed as the primary
driver**: Candidate A (precision dilution) explains at most a small, statistically indistinguishable
share of the effect; Candidate B (candidate-pool mismatch) is actively contradicted by its own
test — matching the eval-time candidate distribution made results worse, not better. The
data now more plausibly points to a general small-dataset fine-tuning instability (with a
specific, testable false-negative-contamination sub-hypothesis for why full-corpus mining
was the worst run), not either originally-hypothesized mechanism specifically.

## Consequences

- **What changes:** nothing in production. `src/triage_iq/api/loader.py` continues to load only
  the unmodified `dup_index_kubernetes_kubernetes_bge` baseline. Neither
  `d3_finetuned_k8s_related_valid_subset` nor `d3_finetuned_k8s_related_fullcorpus_negs` is
  referenced anywhere in serving code (confirmed by inspection, same check ADR-0048 performed).
- **What becomes easier:** the "which of the two named mechanisms explains it" question is closed
  with correctly-measured, disentangling evidence — a future session revisiting k8s fine-tuning
  doesn't need to re-run either test blind. The false-negative-contamination sub-hypothesis is a
  concrete, cheap-to-test next step if this thread is picked up again (compare mined-negative
  precision: sample and blind-label whether full-corpus-mined negatives are truer negatives than
  restricted-pool-mined ones).
- **What becomes harder:** nothing new shipped; retrieval's README section gains a second
  rejected-lever entry for the same underlying investigation.
- **Open, not pursued here:** (1) the false-negative-contamination test above; (2) whether
  reducing epochs (early stopping before the loss/regression correlation observed above
  fully develops) changes the picture — not attempted, since GG's instruction scoped this session
  to the two named candidates and this ADR reports exactly that, not an open-ended hyperparameter
  chase; (3) Candidate A and B were never combined (VALID-only pool + full-corpus negatives) —
  flagged, not run, to avoid compounding an already-large session's GPU/labeling spend without a
  fresh escalation.
- **Cost:** $0. Local RTX 3070. Candidate A: ~5.2min train + eval. Candidate B: ~2min GPU
  full-corpus encoding + ~9.2min train + eval. 15 parallel blind-labeling agent batches for the
  448-pair pool census.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Stop at the zero-cost diagnostic since it didn't confirm dilution outright | GG's instruction was explicit: zero-cost first, then A, then B if needed — the zero-cost result was inconclusive-but-suggestive (not a clean refutation), so A was still the correct next step per the pre-agreed escalation order, not a discretionary skip. |
| Combine Candidate A and B (VALID-only pool + full-corpus negatives) in this pass | Out of the two-candidate scope GG specified; B's own result (full-corpus negatives made things worse in isolation) makes the combined test lower-priority than the false-negative-contamination test, which more directly explains B's own surprising direction. Flagged as future work, not silently dropped. |
| Chase the loss/regression correlation with an early-stopping sweep now | n=3 is suggestive, not established; a sweep is new scope beyond the two named candidates this ADR was scoped to test. Flagged as an open question, not run without a fresh escalation, consistent with this project's measure-first discipline (ADR-0034/0048 precedent: don't sweep around a result that isn't ambiguous yet). |
