# Q5b: domain cross-encoder reranker for vscode duplicates (pre-registration, runbook R9)

Written and committed 2026-10-30 BEFORE any reranker training and before any reranker number
exists. GG decision D95 item (e); spec-quality-push.md section 7, method 4. Mirrors the structure
and rigour of `docs/Q5A_FINETUNE_PREREG_2026-10.md` (R8).

Inherits unchanged from `docs/Q5A_DENSE_PREREG_2026-10.md` (#344), `docs/Q5A_DENSE_RESULTS_2026-10.md`
(#345) and `docs/Q5A_FINETUNE_RESULTS_2026-10.md` (R8): the task (query = the later issue of a
duplicate pair, target = its earlier canonical), the time-respecting index, the tie rule, the
train-only BM25 statistics, Recall@5 over all queries as the primary metric, Wilson / paired
bootstrap CIs (2000 resamples, seed 42) and the splits. Scoring code is
`scripts/qp2_vscode_dup_dense.py` / `scripts/qp2_vscode_dup_eval.py`, reused as is. Target
(ADR-0066): test Recall@5 >= 80% as a point estimate; the frozen R8 winner reaches 40.4% on test.

## Hypothesis (falsifiable)

H1: a cross-encoder fine-tuned on the train-split duplicate pairs, reranking the top-50 of the
frozen R8 retriever, raises validation Recall@5 above the frozen R8 winner (F4 fine-tuned bge-base
+ RRF k=10 with the #345 BM25 variant; validation R@5 524/1,156 = 45.33%). Mechanism: the
bi-encoder compares two independently encoded vectors; a cross-encoder attends across both texts
and can match the specific symptom (error string, feature, stack frame) that makes two issues
duplicates rather than merely related. H0: no gain (paired CI includes zero or the gain is below
the bar below). BELIEVED, not measured: with ~2.1k training pairs the gain, if any, is modest;
a reranker over the top-50 can never exceed the retriever's recall@50 (validation 750/1,156 =
64.9%, the R8 validation report), so 80% is unreachable by this lever alone and the result is
reported against that ceiling.

## Scope: what is reranked

Candidates come from the frozen R8 winner (F4 dense + the frozen #345 BM25 variant, RRF k=10,
ranks over the full time-respecting index). The reranker sees the top **N = 50** of that list per
query and reorders it; ranks beyond 50 are unchanged (a canonical outside the top 50 keeps its
retriever rank, so the full-list metrics R@1/5/10/50 and MRR are directly comparable with R8; R@50
is identical to R8 by construction). N = 50 is chosen to match the R@50 ceiling reported
everywhere in this project; metrics for the top-30 prefix (the spec's N) are derived from the same
scores and reported as secondary, not selecting. CPU latency is measured at N = 30 and N = 50.

## Models (two classes, pinned revisions, both open-licensed)

| id | model | revision | size | licence | role |
|---|---|---|---|---|---|
| M | `cross-encoder/ms-marco-MiniLM-L12-v2` | 7b0235231ca2674cb8ca8f022859a6eba2b1c968 | 33M, 12 layers, hidden 384 | Apache-2.0 | serving-viable class |
| B | `BAAI/bge-reranker-base` | 2cfc18c9415c912f9d8155881c133215df768a70 | 278M, XLM-R base | MIT | quality upper bound, reported with its CPU cost |

Justification: both are strong public relevance cross-encoders with a 1-logit head, so no new
head is initialised. M is small enough that top-30 on CPU can plausibly meet the spec's 300 ms
budget; B is the stronger-prior class and shows whether capacity matters, at a latency that
likely does not meet it (BELIEVED, measured in the report). A third class (bge-reranker-v2-m3,
568M) is excluded: far over any CPU budget for ~2k training pairs. Weights fp32 (`.float()`),
forward in fp16 autocast with a GradScaler (the R8 recipe; R11's fp16 master weights were
broken), gradient checkpointing, AdamW, weight decay 0.01, linear warmup 10% then linear decay,
grad-norm clip 1.0, seed 42 (python, numpy, torch). Pair input: (anchor text, candidate text),
text = `title. clean_body` (the production recipe, `qk_kaggle.build_texts`), each side cut to its
first 3,000 characters, tokenizer pair truncation `longest_first`, total max length 512.

## Training data (train split only, nothing else)

* Pairs: exactly the R8 training pairs (`r8_dup_finetune.train_pairs`): BOTH ends train-split
  issues (created < 2025-07-01, the #345 splits), canonical strictly older than the source
  (2,069 pairs, 2,063 distinct sources).
* Hard negatives: for each pair, the frozen R8 retriever (same code path as above) is run for the
  pair's source over the time-respecting index, the top-40 are taken, every member of the
  anchor's duplicate component (union-find over train pairs, as R8) is removed, and **7 negatives**
  are drawn at random (seed 42, without replacement) from the first 30 remaining. For a train
  source the time-respecting prefix contains only train issues (the train split is every issue
  created before the cut-off), asserted. Anchors with < 7 eligible negatives are dropped and
  counted. Known risk: an unlabeled duplicate can sit among the negatives (R8 avoided this with a
  rank 10-50 band; here the reranker must separate the very top ranks, so ranks 1-30 are used).
  The F4 retriever was trained on these same pairs, so on train queries the canonical ranks
  higher than it will on held-out queries; this affects only which negatives are mined, not the
  group construction.
* Loss: listwise (localized contrastive) cross-entropy over each group of 8 pairs (1 positive,
  7 negatives; positive at index 0); 4 groups = 32 sequences per step; 2 epochs (~1,030 steps).
* Leakage guard (code, asserted in the local build and again in the kernel from a shipped split
  table): every anchor, positive and negative is a train-split issue and none is in the
  validation or test set; the build fails otherwise. The kernel input contains no validation or
  test duplicate label. The kernel does receive the validation queries' candidate lists (query ids
  and candidate ids, needed to score them) but no canonical. **The kernel does not receive or
  score any test query**; the test stage will run a separate scoring-only kernel with the frozen
  weights, after a freeze commit (as R8).

## Grid (4 configs, fixed; no additions after seeing numbers)

| id | model | lr | epochs | groups x negatives |
|---|---|---|---|---|
| X1 | M | 2e-5 | 2 | 4 x (1+7) |
| X2 | M | 6e-5 | 2 | 4 x (1+7) |
| X3 | B | 1e-5 | 2 | 4 x (1+7) |
| X4 | B | 3e-5 | 2 | 4 x (1+7) |

The final weights of each config are used (no checkpoint picking on validation inside a run).
Diagnostic, not selecting: the two untouched pretrained models (Z-M, Z-B) score the same
validation candidate lists, to show how much of any gain is fine-tuning.

## Candidates and selection (validation only)

Per config, three candidates over the top-50 list of each validation query: **(a) pure**: order by
cross-encoder score (ties -> better retriever rank); **(b) RRF-10** and **(c) RRF-60**: reciprocal
rank fusion of the cross-encoder rank (1..50 within the list) and the retriever rank (1..50),
`1/(k + r_x) + 1/(k + r_ret)` with k in {10, 60}, ties -> better retriever rank. That is 12
candidates; no other fusion (no learned weights, no score interpolation). Reference, recomputed in
the same run from the frozen artifacts and required to reproduce **524 / 1,156 = 45.33%** exactly
(abort otherwise): the frozen R8 winner. Selection: highest validation R@5 among the 12; ties ->
higher MRR -> simpler (pure before fusion, larger k before smaller, M before B, smaller lr).

Checks before a result is used: (1) the score file covers exactly the (query, candidate) pairs of
the candidate file, no NaN; (2) kernel-vs-local parity: 300 pairs (seed 42) re-scored on CPU fp32
with the downloaded weights of the selected config must match the kernel scores at Spearman >=
0.999 (max abs difference reported); if not, the run is not used.

## Success criterion

Primary: paired validation difference (selected candidate - frozen R8 winner) in R@5 over the same
1,156 queries, percentile bootstrap 95% CI (2000, seed 42; shared resamples). **SUCCESS iff the
point gain is >= +2.0 points AND the CI lower bound is > 0.** Selection over 12 candidates makes
the validation gain optimistic, so success only licenses ONE test evaluation of the selected
candidate vs the frozen R8 winner on the same test queries, with its config frozen and hashed in
a committed file first, as R8; the test criterion is the paired R@5 CI excluding zero, and
ADR-0066's 80% is reported as met / not met.

NEGATIVE RESULT: any of (a) point gain < +2.0, (b) CI includes zero, (c) reference not
reproduced or parity fails. Then the test split is not touched, the result is written up as is
(all 12 candidates plus the zero-shot diagnostics), and a fine-tuned cross-encoder over the R8
top-50 is recorded as not paying off under this recipe. Not retried under this registration.

## Reported (validation)

R@1/5/10/50 (within the reranked full list) with Wilson 95% CIs, MRR, ceiling (R@50 = 64.9%,
share of queries with an indexed canonical as in R8), the paired bootstrap vs the R8 winner for
every metric, per-config R@5 for all 12 candidates, top-30 variant, zero-shot diagnostics, and
the per-query CPU latency of the rerank stage (fp32, 4 threads, `CUDA_VISIBLE_DEVICES=-1`, the
selected config and the other model class, N = 30 and 50, median and p95 over 40 validation
queries, tokenization included, ms) against the spec's 300 ms budget. Serving would be CPU, so a
GPU-only gain is reported as such.

## Compute, reproducibility

Kaggle private kernel, one T4 (or P100), no paid resource, internet on only for the pinned HF
model downloads. Budget (ESTIMATE, not measured): M ~5 min train + ~3 min scoring per config; B
~20-25 min train + ~10 min scoring per config; zero-shot scoring ~13 min; ~1.5-2 h total. The
kernel stops starting new configs after 4 h elapsed and records which were skipped (a partial
grid is then reported as partial). Account weekly GPU quota is shared (30 h, D85): the push is
attempted ONCE; a quota error is recorded and the run stops. Datasets and kernel private
(asserted before upload); kernel and driver sha256 recorded with the results. Outputs
(scores, weights) go to `D:\triageiq-data\models\microsoft__vscode\xenc_rerank\`, not into git.
No serving change, no index build, no cutover (GG's). No Groq, no cassette or baseline change.
