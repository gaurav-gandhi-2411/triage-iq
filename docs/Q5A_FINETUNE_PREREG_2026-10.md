# Q5a part 3: fine-tuned dense retriever for vscode duplicates (pre-registration, runbook R8)

Written and committed 2026-10-29 BEFORE any training run and before any fine-tuned number exists.
Inherits unchanged from `docs/Q5A_DENSE_PREREG_2026-10.md` (#344) and
`docs/Q5A_DENSE_RESULTS_2026-10.md` (#345): the task (query = the later issue of a duplicate pair,
target = its earlier canonical), the time-respecting index, the tie rule, the train-only BM25
statistics, Recall@5 over all queries as the primary metric, Wilson / paired-bootstrap CIs (2000
resamples, seed 42) and the ceiling. Code that scores is `scripts/qp2_vscode_dup_dense.py`, reused
as is. Target (ADR-0066): test Recall@5 >= 80% as a point estimate; the frozen #345 winner reaches
33.0% [30.8, 35.3] on test, so the target is open.

## Hypothesis (falsifiable)

H1: contrastive fine-tuning of the production encoder on the train-split duplicate pairs raises
validation Recall@5 of the fused retriever above the frozen #345 winner (RRF k=10 of BM25 title x3 +
error lines and zero-shot BGE; validation R@5 40.3%, 1,156 queries). Mechanism: zero-shot BGE
puts topically similar issues near each other, while duplicates in vscode share a specific symptom
that a generic embedding does not weight; pairs teach it which similarity matters. H0: no gain
(paired CI includes zero or the point gain is below the bar below). BELIEVED, not measured: only
~2.1k training pairs exist, so the gain, if any, is modest and the 80% target stays out of reach;
the ceiling is 92.8% on validation, 95.8% on test.

## Model

`BAAI/bge-base-en-v1.5` at the production revision `a5beb1e3`, CLS pooling + L2 normalisation,
max length 512, fine-tuned end to end with MultipleNegativesRankingLoss: cosine similarity scaled
by 20 (temperature 0.05, the sentence-transformers default), cross-entropy over the batch, anchor =
the later issue (query side), positive = its canonical, both encoded by the same encoder
(document-to-document, no query instruction, as the production vscode setting
`QUERY_INSTRUCTION_REPO_OVERRIDE = False` and as D1 in #345). Chosen over a cross-encoder reranker
because it reuses the production index shape (one vector per issue), costs nothing at serve time
beyond swapping the model, and recall@50 of the current stack (55%) would cap a reranker anyway.
Weights are fp32 (`.float()`), forward in fp16 autocast with a GradScaler; the earlier R11 run
with fp16 master weights was broken (R11). Gradient checkpointing; AdamW, weight decay
0.01, linear warmup 10% then linear decay; batch 32 pairs; seed 42 (python, numpy, torch).

False in-batch negatives are masked: duplicate pairs form connected components (a canonical has
several duplicates; a source can have several canonicals), and any candidate in the same component
as the anchor, other than its own positive, is removed from the softmax. Without this, in-batch
collisions would be common: 2,063 sources share 1,156 canonicals, so
in-batch collisions would be common.

## Data (train only, nothing else)

* Pairs: rows of `derived_v2/microsoft__vscode/duplicate_pairs.parquet` where BOTH ends are
  issues of the train split (created < 2025-07-01, `splits/microsoft__vscode/train.parquet`, the
  exact splits of #345) and the canonical was created strictly before the source. Measured
  2026-10-29: 2,593 train-train rows, 2,069 after the date rule, 2,063 distinct sources, 1,156
  distinct canonicals. Pairs with an end in validation or test, or outside the issue table, are
  excluded. No validation or test pair, query or canonical enters as positive or negative.
* Text: `qk_kaggle.build_texts` production recipe (`title. clean_body`, body cut by tokens to fit 512
  with the model tokenizer, as `similar_issues._build_text` and the #345 vectors); the kernel
  re-implements the cut exactly as `embed_bge.py` does.
* Hard negatives (grid factor): one per pair, drawn at random (seed 42) from BM25 ranks 10-50 (B2,
  statistics from train issues only) among train issues created before the source and outside the
  pair's component; the 10-rank gap lowers the chance of an unlabeled duplicate. Pairs with no
  eligible candidate fall back to a random earlier train issue outside the component.
* Leakage guard (code, asserted in the local build and again in the kernel from a split table
  shipped with the data): every number in anchors, positives and negatives is a train issue and
  none is in the validation or test set; the build fails otherwise. Texts of train issues that
  later gain a near-identical validation or test issue (`leak_texthash`) are train data by time
  order and are kept.

## Grid (4 configs, fixed; no additions after seeing numbers)

| id | lr | hard negative | epochs | batch |
|---|---|---|---|---|
| F1 | 1e-5 | no | 3 | 32 |
| F2 | 3e-5 | no | 3 | 32 |
| F3 | 1e-5 | yes | 3 | 32 |
| F4 | 3e-5 | yes | 3 | 32 |

About 195 optimiser steps per config. The final weights of each config are used (no checkpoint
picking on validation inside a run).

## Candidates and selection (validation only)

For each config, the fine-tuned vectors replace the zero-shot ones for ALL issues (one pass per
config, run on Kaggle, no labels in the kernel). Candidates per config: dense alone (FT dense) and
RRF with the already-frozen BM25 variant (title x3 + error lines, selected in #345) at k in {10,
30, 60}, ranks over the full time-respecting index, as #345. That is 16 candidates. Reference
(re-computed in the same run from the zero-shot vectors, must reproduce 40.31%): the frozen #345
winner H(k=10). Selection: highest validation R@5 among the 16; ties -> higher MRR -> simpler
(dense before RRF, smaller k, no hard negative, smaller lr). Kernel-vs-#345 pipeline parity: 2,000
issues (seed 42 sample) embedded in the kernel with the untouched base model must match the #345
vectors at min cosine >= 0.999; if not, the run is not used.

## Success criterion

Primary: paired validation difference (selected candidate - frozen #345 winner H(k=10)) in R@5,
over the same 1,156 queries, percentile bootstrap 95% CI (2000, seed 42). SUCCESS iff the point
gain is >= +2.0 points AND the CI lower bound is > 0. The selection over 16 candidates makes the
validation gain optimistic, so success only licenses the next step: ONE test evaluation of the
selected candidate against the #345 frozen winner on the same test queries, with its config frozen
and hashed in a committed file first (as #345); the test criterion is the paired R@5 CI excluding
zero, and ADR-0066's 80% is reported as met / not met. Secondary, reported not selecting: R@1/10/50,
MRR, ceiling, FT dense vs zero-shot D1.

NEGATIVE RESULT: any of (a) point gain < +2.0, (b) CI includes zero, (c) pipeline parity fails.
Then the test split is not touched, the result is written up as is (including per-config numbers),
and fine-tuning on ~2.1k pairs is recorded as not paying off under this recipe. Not retried with
other hyperparameters under this registration.

## Compute, reproducibility

Kaggle private kernel, one T4 (or P100 if assigned), no paid resource. Budget
(ESTIMATE, not measured): <= 35 min per config (train <= 10, embed 109,537 issues <= 25), <= 2.5
h total; the kernel stops starting new configs after 4 h elapsed and records which were skipped
(a partial grid is then reported as partial). Account weekly GPU quota is shared (30 h, D85); a
quota error on push is recorded and the run stops. Seed 42 everywhere; model revision pinned;
datasets and kernel private; kernel and driver sha256 recorded with the results. Outputs
(embeddings, weights) go to `D:\triageiq-data\models\microsoft__vscode\dup_finetune\`, not into
git. No serving change, no index build, no cutover (GG's).
