# Q5a part 2 results: vscode duplicate retrieval (2026-10-29)

Design and selection rule: `docs/Q5A_DENSE_PREREG_2026-10.md` (committed before any number here).
Code: `scripts/qp2_vscode_dup_dense.py` (script sha256 in the frozen file). Artifacts:
`reports/q5a_vscode_dup_dense_{validation,frozen,test}_2026-10.json`. Target (ADR-0066): test
Recall@5 >= 80% as a point estimate. **Not met.**

## Validation (1,156 queries, ceiling 92.8% = 1,073 queries with an indexed canonical)

| candidate | R@5 (Wilson 95%) | R@10 | R@50 | MRR |
|---|---|---|---|---|
| B1 TF-IDF | 27.3% [24.8, 30.0] | 31.4% | 40.9% | 0.231 |
| B2 BM25 (baseline, = #309) | 29.2% [26.7, 31.9] | 33.1% | 40.9% | 0.250 |
| best BM25 variant (title x3 + error lines) | 35.0% [32.3, 37.7] | 39.7% | 49.8% | 0.284 |
| D1 dense BGE | 37.5% [34.8, 40.4] | 42.0% | 53.5% | 0.303 |
| H RRF k=10 (best BM25 + D1), selected | 40.3% [37.5, 43.2] | 47.1% | 57.6% | 0.319 |

Findings (VERIFIED from the validation JSON): title weight is the main BM25 lever (R@5 29.2 -> 34.0
at w=3 without extras); error-line extraction adds 0.9 to 1.8 points at every w; code-identifier extraction
does not help (29.7 vs 29.2 at w=1, and below the no-extras variant at w=3). Title boost was still
rising at w=3, the edge of the pre-registered grid; it was not extended after seeing numbers
(BELIEVED: w=4-5 would add a little; untested). RRF k=10 > 30 > 60. Selection on 16 candidates
makes the validation number optimistic; the test number below is the honest one.

## Test, single frozen evaluation (1,705 queries, ceiling 95.8%)

| | R@1 | R@5 (Wilson 95%) | R@10 | R@50 | MRR |
|---|---|---|---|---|---|
| B2 BM25 baseline | 13.3% | 21.5% [19.6, 23.5] | 24.7% | 34.8% | 0.174 |
| frozen winner H(k=10) | 17.8% | **33.0% [30.8, 35.3]** | 40.5% | 55.4% | 0.252 |
| paired difference | +4.6 [+3.2, +5.9] | **+11.5 [+9.9, +13.2]** | +15.8 [+13.9, +17.7] | +20.6 [+18.5, +22.6] | +0.078 [+0.066, +0.090] |

The gain over BM25 is clear (all paired CIs exclude zero), but 33.0% is far below the 80% target
and below the 95.8% ceiling. The test baseline is lower than validation (21.5% vs 29.2%) because
the splits differ (BELIEVED: time drift; not tested), with no code difference; the winner and baseline are scored on the same rows.
Dense-only was not evaluated on test (not part of the pre-registered test set).

## Reading

Lexical + generic dense retrieval raises the share of duplicates found in the top 50
(34.8% -> 55.4%) but most true canonicals still rank outside the top 5. Reaching 80% at k=5 is
not plausible with these off-the-shelf signals (BELIEVED); candidate next levers, none run: a
fine-tuned or cross-encoder reranker over the top 50 (recall@50 55% bounds it), query-time
title weighting beyond 3, and a duplicate-aware fine-tuned encoder (H2 in the #309 design).
No serving change; any index build or cutover is GG's.
