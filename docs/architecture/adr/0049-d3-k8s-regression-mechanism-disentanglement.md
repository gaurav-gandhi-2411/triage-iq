# ADR-0049 — k8s D3 regression: mechanism disentanglement — real data-scale ceiling confirmed, exit condition recorded

**Status:** Accepted (negative result; no cutover, no HF release; extends ADR-0048's rejection with a stated mechanism and retry threshold)
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

### Follow-up: false-negative contamination check (measured, not just flagged)

GG's escalation: before chasing the small-dataset-instability reading below, check the cheaper,
directly-measurable explanation for *why* Candidate B specifically backfired — full-corpus mining
draws "hard negatives" from a far larger, more diverse candidate space (29,717 issues vs. 792),
so its highest-similarity matches are more likely to actually be true near-duplicates the
regex-based gold-mining heuristic simply never captured, i.e. **false negatives** the contrastive
loss then explicitly trains the model to push apart.

Sampled 40 mined (query, negative) pairs from `data/d3_hard_negatives_k8s_related_fullcorpus_negs.parquet`
(seed=42, `scripts/d3b_false_negative_audit.py --sample`), blind-labeled in 2 independent batches
of 20 against the same VALID/EXCLUDE rubric used throughout this investigation (here, VALID means
"this negative is actually a false negative" — genuinely related, should not have been trained as
a negative).

**Result: 11/40 = 27.5% false-negative rate, Wilson95 CI [16.1%, 42.8%].** Materially above zero
— over a quarter of the sampled mined negatives are, on inspection, genuinely related to their
query. Representative examples (all VALID/false-negative, all high cosine rank):

- q#6077 vs. neg#6195 (rank 1, score 0.838): both describe the identical bug — marking a node
  unschedulable breaks when kubelet (not the node controller) pushes status updates.
- q#14520 vs. neg#14568 (rank 2, score 0.821): the negative is literally the fix PR, its own text
  reads "Fixes kubernetes/kubernetes#14520."
- q#19095 vs. neg#16668 (rank 1, score 0.800): both concern the same specific HPA
  cross-namespace privilege-escalation issue.

**This is a confirmed, measured, material mechanism — not a speculative one.** It directly
explains why Candidate B regressed harder than D3's original restricted-pool run: a materially
higher share of its "negatives" were actively mistrained.

### Follow-up 2: does restricted-pool mining (D3 original, Candidate A) share this contamination? Measured — no, materially lower.

Same open question this ADR flagged: D3's original and Candidate A both mine negatives from the
792-issue training-pool-restricted candidate space, not the full corpus. If that space is
comparably contaminated, false-negative contamination would explain the *base* regression too,
not just Candidate B's excess over it — a solvable data-hygiene problem, not a ceiling. If low,
the base regression stays unexplained by this mechanism.

Sampled 40 mined (query, negative) pairs from `data/d3_hard_negatives_k8s_related.parquet` (D3's
original restricted-pool run; seed=42, `scripts/d3b_false_negative_audit.py --run restricted
--sample`), blind-labeled in 2 independent batches of 20 against the identical rubric.

**Result: 4/40 = 10.0% false-negative rate, Wilson95 CI [4.0%, 23.1%]** — materially lower than
the full-corpus run's 27.5% [16.1%, 42.8%]. A two-proportion z-test on the two samples: z=2.005,
two-sided p=0.045 — right at the conventional significance threshold, consistent with (not
overwhelming proof of, at n=40 each) a real difference between the two mining strategies' true
false-negative rates. The 4 restricted-pool false negatives found are real (e.g. q#19095 vs.
neg#21578, rank 1/score 0.775: both concern the same concrete HPA GA-readiness effort) but at
roughly a third the rate of the full-corpus run.

**Verdict, per GG's own pre-registered decision rule:** 10.0% (CI upper bound 23.1%) sits below
the ">15% material" bar GG specified for "contamination explains both regressions." **False-
negative contamination explains Candidate B's excess regression over the baseline, but does not
explain the base D3/-15.15pp regression itself.** That regression's mechanism remains
unaccounted-for by either of ADR-0048's two originally-named candidates (disentangled above) or
by this contamination check — which means the small-dataset-instability reading (below) regains
standing as the leading explanation for the base case, on the strength of the three converging,
monotonic-in-size lines GG named: k8s (~250-450 pairs) regresses hard; vscode (~1,958 pairs)
nulls; DeBERTa-v3-base (184M params) lost by a wider margin than DistilBERT (66M) on the same
small classifier training data (ADR-0046's per-generation audit, item 7).

## Cross-run comparison and a candidate unifying observation

| Run | Pool | Negatives | Final loss | Clean-66 R@5 delta | CI95 |
|---|---|---|---|---|---|
| D3 original (ADR-0048) | 448 pairs, 56.5% VALID | restricted (792-issue) | 0.0903 | -15.15pp | [-24.2,-7.6] |
| D3a Candidate A | 253 pairs, 100% VALID | restricted (470-issue) | 0.1161 | -12.12pp | [-22.7,-1.5] |
| D3a Candidate B | 448 pairs, 56.5% VALID | full-corpus (29.7k-issue) | **0.0591** | **-19.70pp** | [-31.8,-7.6] |

Across these three runs, final training loss and eval-harm also move together: the lowest-loss
run (Candidate B, 0.0591 — the closest of the three to D2's original 0.045 memorization-artifact
threshold, ADR-0048) produced the worst regression; the highest-loss run (Candidate A, 0.1161)
produced the least-bad regression. The false-negative audits give the Candidate B end of this
pattern a concrete mechanistic reading: its negatives, at 27.5% (Wilson95 [16.1,42.8]) false-
negative contamination — materially higher than the restricted-pool runs' 10.0% [4.0,23.1],
z=2.005/p=0.045 — gave the contrastive objective more actively-wrong gradient signal per step,
consistent with both its lower final loss (fitting corrupted labels faster) and its worse
held-out regression. That explains Candidate B's *excess* over the baseline. It does not explain
why D3 original (448 pairs, 10.0% contamination — measured low, not zero) and Candidate A (253
pairs, presumably similarly low, not separately measured) both still regressed hard on their own.

## Decision: REJECTED, all three configurations — no cutover, no HF release. Base regression: real ceiling, stated mechanism, exit condition recorded below.

None of the three k8s_related fine-tunes (D3 original, D3a Candidate A, D3a Candidate B) clears
the ship bar. **Neither of ADR-0048's two originally-named candidate mechanisms is confirmed as
the primary driver of the base regression**: Candidate A (precision dilution) explains at most a
small, statistically indistinguishable share; Candidate B (candidate-pool mismatch, as originally
framed) is actively contradicted by its own test. False-negative contamination is now measured,
confirmed material for Candidate B (27.5%) specifically, and measured materially lower for the
restricted-pool runs (10.0%, below GG's own >15% "material" bar) — so it explains Candidate B's
*excess* harm over D3 original, not the base -15.15pp regression itself.

**This satisfies rule 101c's bar for accepting a real limit.** GG's escalation conditioned
"call it a ceiling" on the false-negative check coming back low for the restricted-pool runs —
it did (10.0% vs. Candidate B's 27.5%, a measured, not assumed, gap). With both of ADR-0048's
named mechanisms disentangled and refuted as primary drivers, and the cheapest, most direct
alternative explanation (false-negative contamination) also measured and found insufficient for
the base case, the small-dataset fine-tune instability reading is now the best-supported
explanation on three independent, monotonic-in-size lines:

| Model / task | Effective training scale | Result |
|---|---|---|
| k8s_related bi-encoder fine-tune | 253-448 pairs | Confirmed harmful regression (-12 to -20pp R@5) |
| vscode_duplicate bi-encoder fine-tune | 1,734-1,958 pairs | No signal (null, not harmful) |
| DeBERTa-v3-base (184M) component classifier, ARM 1/2 | k8s/vscode multi-label training sets (same order of magnitude as the classifier's pre-existing small-data regime) | Lost to DistilBERT (66M) by a wider margin than DistilBERT itself lost to TF-IDF+LR — bigger model, same small data, worse result (ADR-0046 per-generation audit, item 7) |

**Stated mechanism:** contrastive (or classification) fine-tuning on a pool in the low hundreds of
examples can converge to near-zero training loss on that pool's idiosyncrasies while eroding the
base checkpoint's pretrained general-domain competence — a small-data overfitting/catastrophic-
forgetting shape, not a data-quality defect this session's three targeted interventions (precision
filtering, full-corpus negative re-mining, contamination measurement) could reach.

**Residual confound, disclosed not resolved:** k8s's pool isn't only smaller than vscode's — it's
also lower-precision (56.5% vs. 76.7-85% strict/D1-measured VALID) and the two properties covary
across repos in this dataset by construction (vscode's dominant channel, `dup_comment`, is both
higher-volume and structurally higher-precision). Candidate A showed that within k8s, precision
alone (holding pool size confounded with it) doesn't rescue the result — but the *cross-repo* size
comparison above still can't cleanly separate "more pairs" from "more pairs of a channel that
happens to also be higher-precision." Recorded honestly as an open confound, not chased further.

**Exit condition (per GG's instruction — a size threshold, not a flat "can't be done"):** the
data brackets a k8s retry between "448 pairs, confirmed harmful" and "roughly vscode's scale,
confirmed null (not positive)." A k8s fine-tune becoming *worth retrying* — i.e., plausibly not
harmful — would need a pool on the order of **vscode's ~1,700-2,000-pair scale**, the only point
in this project's own history where this architecture/loss combination stopped actively
regressing at small scale. That's the honest floor, not a guarantee of a win: even at that scale
vscode itself only reached NULL, not a positive result, so clearing the harm floor and clearing
the ship bar (meaningful lift, CI excluding zero) are two different, both-still-open bars. k8s's
own regex-mined channels currently ceiling at 448 pairs (ADR-0047) — reaching ~1,700-2,000 would
need a structurally new data source (k8s has no `dup_comment`-equivalent channel, per ADR-0047's
own census), not incremental mining of what exists today. **Blocked pending a new k8s training-
data channel at roughly 4x current volume, not "can't be done."**

## Consequences

- **What changes:** nothing in production. `src/triage_iq/api/loader.py` continues to load only
  the unmodified `dup_index_kubernetes_kubernetes_bge` baseline. Neither
  `d3_finetuned_k8s_related_valid_subset` nor `d3_finetuned_k8s_related_fullcorpus_negs` is
  referenced anywhere in serving code (confirmed by inspection, same check ADR-0048 performed).
- **What becomes easier:** the k8s fine-tune question is closed with a stated mechanism and a
  concrete, numeric exit condition (below) instead of an open-ended "keep trying levers" loop — a
  future session doesn't need to re-litigate whether this is a ceiling or re-try any of the five
  configurations this investigation already measured (D3 original, Candidate A, Candidate B, and
  both false-negative audits). Negative-mining hygiene (filter high-similarity mined negatives
  through a cheap validity check before training) remains a real, cheap win *if* a future k8s
  retry ever has enough data to be worth attempting — recorded as a concrete implementation note
  for that future attempt, not a reason to retry now.
- **What becomes harder:** nothing new shipped; retrieval's README section gains a second
  rejected-lever entry for the same underlying investigation, this time with a closed mechanism
  and a stated retry threshold rather than an open question.
- **Open, not pursued here:** (1) whether reducing epochs (early stopping before the
  loss/regression correlation observed above fully develops) changes the picture for a future
  attempt at sufficient scale — not attempted, out of this session's scope; (2) Candidate A and B
  were never combined (VALID-only pool + full-corpus negatives) — moot now that full-corpus
  mining's specific defect (false negatives) is measured and the base regression is attributed
  elsewhere; (3) the size/precision confound noted above (vscode's larger pool is also
  higher-precision) — disentangling would need a same-precision, size-varied ablation, out of
  scope without a new data source.
- **Cost:** $0. Local RTX 3070. Candidate A: ~5.2min train + eval. Candidate B: ~2min GPU
  full-corpus encoding + ~9.2min train + eval. 15 parallel blind-labeling agent batches for the
  448-pair pool census. Two false-negative audits: 4 parallel blind-labeling agent batches total
  (80 pairs, `scripts/d3b_false_negative_audit.py --run {fullcorpus,restricted}`), no retraining.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Stop at the zero-cost diagnostic since it didn't confirm dilution outright | GG's instruction was explicit: zero-cost first, then A, then B if needed — the zero-cost result was inconclusive-but-suggestive (not a clean refutation), so A was still the correct next step per the pre-agreed escalation order, not a discretionary skip. |
| Combine Candidate A and B (VALID-only pool + full-corpus negatives) in this pass | Out of the two-candidate scope GG specified; B's own result (full-corpus negatives made things worse in isolation) makes the combined test lower-priority than the false-negative-contamination test, which more directly explains B's own surprising direction. Flagged as future work, not silently dropped. |
| Chase the loss/regression correlation with an early-stopping sweep now | n=3 is suggestive, not established; a sweep is new scope beyond the two named candidates this ADR was scoped to test. Flagged as an open question, not run without a fresh escalation, consistent with this project's measure-first discipline (ADR-0034/0048 precedent: don't sweep around a result that isn't ambiguous yet). |
| Declare a real data-scale ceiling immediately after Candidate B's false-negative result (27.5%), without checking restricted-pool mining | GG's escalation explicitly conditioned "call it a ceiling" on the restricted-pool check, not the full-corpus one — the full-corpus result alone only explains Candidate B's excess, not the base regression. Measuring the restricted-pool rate (10.0%, below the ">15% material" bar) is what actually clears rule 101c's bar for accepting a limit; skipping it would have been declaring a ceiling on an unmet condition. |
| Report the base regression as fully unexplained rather than attributing it to small-dataset instability | Two of ADR-0048's candidates are now refuted/insufficient (precision dilution, candidate-pool mismatch) and the third (false-negative contamination) is measured as insufficient for the base case specifically (10.0%, below threshold) — three independent, mechanism-consistent, monotonic-in-size lines (k8s fine-tune, vscode fine-tune, DeBERTa-vs-DistilBERT classifier) is the strongest evidentiary basis this investigation has reached for any explanation, and rule 101c requires naming the best-supported mechanism once genuinely exhausted, not declining to conclude. |
