# Q5b results: cross-encoder reranker for vscode duplicates (R9, 2026-10-30)

Design and selection rule: `docs/Q5B_XENC_RERANK_PREREG_2026-10.md` (committed before any run).
Reference: the frozen R8 winner (F4 fine-tuned bge-base + RRF k=10, `docs/Q5A_FINETUNE_RESULTS_2026-10.md`).
Target (ADR-0066): test Recall@5 >= 80% as a point estimate. **Not met (43.3%).** H1 (a fine-tuned
cross-encoder over the R8 top-50 beats the R8 winner) **holds on validation and on the single
test evaluation**, but only through fusion with the retriever, and the stack is **not
serving-viable on CPU** (below). Nothing is served or wired in.

All numbers are VERIFIED (read from the committed result files named under Provenance) unless
labelled BELIEVED.

## Selection (validation, 1,156 queries, R@50 ceiling 750 = 64.9%)

Four configs trained (2 epochs, 4 groups x 8 pairs per step, 2,069 train groups, seed 42), three
candidates each, plus zero-shot diagnostics of the untouched models. Selected by the preregistered
rule: **X4 (BAAI/bge-reranker-base, lr 3e-5) + RRF k=10 fusion with the retriever rank**. Source:
validation run output `r9_validation.json` (scratch, summarised in the freeze file).

| candidate | R@1 | R@5 hits = rate (Wilson 95%) | R@10 | R@50 | MRR |
|---|---|---|---|---|---|
| R8 winner (reference) | 26.3% | 524 = 45.3% [42.5, 48.2] | 52.9% | 64.9% | 0.350 |
| X1_pure (MiniLM, 2e-5) | 22.6% | 469 = 40.6% [37.8, 43.4] | 48.4% | 64.9% | 0.311 |
| X1_rrf10 | 26.4% | 543 = 47.0% [44.1, 49.9] | 54.0% | 64.9% | 0.358 |
| X1_rrf60 | 26.2% | 545 = 47.1% [44.3, 50.0] | 54.1% | 64.9% | 0.357 |
| X2_pure (MiniLM, 6e-5) | 22.7% | 467 = 40.4% [37.6, 43.3] | 48.8% | 64.9% | 0.313 |
| X2_rrf10 | 26.3% | 551 = 47.7% [44.8, 50.5] | 54.3% | 64.9% | 0.361 |
| X2_rrf60 | 26.3% | 551 = 47.7% [44.8, 50.5] | 54.9% | 64.9% | 0.359 |
| X3_pure (bge-reranker-base, 1e-5) | 22.6% | 443 = 38.3% [35.6, 41.2] | 47.1% | 64.9% | 0.308 |
| X3_rrf10 | 26.6% | 536 = 46.4% [43.5, 49.2] | 54.8% | 64.9% | 0.359 |
| X3_rrf60 | 26.8% | 538 = 46.5% [43.7, 49.4] | 53.8% | 64.9% | 0.359 |
| X4_pure (bge-reranker-base, 3e-5) | 24.1% | 477 = 41.3% [38.5, 44.1] | 50.5% | 64.9% | 0.327 |
| **X4_rrf10 (selected)** | 28.2% | **559 = 48.4% [45.5, 51.2]** | 54.8% | 64.9% | 0.375 |
| X4_rrf60 | 28.0% | 552 = 47.8% [44.9, 50.6] | 54.7% | 64.9% | 0.372 |

Zero-shot diagnostics (not selecting; untouched pretrained models on the same lists):

| candidate | R@1 | R@5 hits = rate | R@10 | MRR |
|---|---|---|---|---|
| Z-M_pure / rrf10 / rrf60 | 18.3 / 24.6 / 24.0% | 359 = 31.1% / 490 = 42.4% / 470 = 40.7% | 37.9 / 50.4 / 48.4% | 0.253 / 0.330 / 0.322 |
| Z-B_pure / rrf10 / rrf60 | 16.7 / 23.6 / 23.7% | 373 = 32.3% / 498 = 43.1% / 492 = 42.6% | 40.0 / 50.6 / 49.7% | 0.246 / 0.331 / 0.327 |

Paired validation, X4_rrf10 minus R8 winner: R@5 **+3.03 points [+1.38, +4.76]**, R@1 +1.90
[+0.26, +3.63], R@10 +1.90 [+0.52, +3.37], MRR +0.025 [+0.013, +0.036], R@50 exactly 0 by
construction. The success rule (gain >= +2.0 points and CI lower bound > 0) was met, which licensed
one test evaluation. Reading of the grid (VERIFIED as differences, causes BELIEVED):

* **Pure reranking underperforms the retriever** for all four configs (38.3 to 41.3% R@5 vs the
  retriever's 45.3%). Every gain over the R8 winner comes from fusing the cross-encoder rank with
  the retriever rank. Fine-tuning helps over zero-shot in both classes (Z-B_rrf10 43.1% to
  X4_rrf10 48.4%), but even zero-shot fusion alone (40.7 to 43.1%) is below the 45.3% reference,
  so the retriever is the stronger signal here.
* The small MiniLM class reaches 47.7% (X2) against 48.4% for the 278M model: the 0.7-point gap is
  8 queries and was not tested; capacity did not buy much. The selected winner was chosen by the
  preregistered tie order, not by a significant margin over X1/X2/X3 fusion candidates.

## Test, single frozen evaluation (1,705 queries, R@50 ceiling 1,102 = 64.6%)

| | R@1 | R@5 hits = rate (Wilson 95%) | R@10 | R@50 | MRR |
|---|---|---|---|---|---|
| R8 winner (F4 + RRF k=10) | 19.9% [18.1, 21.8] | 688 = 40.4% [38.0, 42.7] | 47.7% [45.4, 50.1] | 64.6% [62.3, 66.9] | 0.294 |
| X4_rrf10 (frozen) | 22.9% [21.0, 25.0] | **738 = 43.3% [41.0, 45.6]** | 52.1% [49.8, 54.5] | 64.6% [62.3, 66.9] | 0.323 |
| paired difference | +3.05 [+1.64, +4.52] | **+2.93 [+1.52, +4.40]** | +4.40 [+2.99, +5.75] | 0 (by construction) | +0.029 [+0.019, +0.039] |
| X4_rrf10, top-30 rerank only (secondary) | 22.9% | 735 = 43.1% | 51.8% | 64.6% | 0.324 |

The frozen R8 winner reproduced exactly on the same code path before anything was scored: 688 /
1,705 hits at R@5 (40.4%), equal to the count pinned in the freeze (the driver aborts otherwise).
**Verdict per the preregistered test criterion (paired R@5 CI excludes zero): met** (+2.93 points,
CI [+1.52, +4.40]). The test gain (+2.93) is close to the validation gain (+3.03), so selection
optimism is not visible here (n=1 split pair; no claim beyond that). **ADR-0066's 80% target: not
met**: 43.3% is 36.7 points short, and 21.3 points under this stack's own 64.6% ceiling (a reranker
over the top 50 cannot exceed it).

## CPU latency of the rerank stage (fp32, 4 threads, tokenization included, 40 validation queries)

Measured under concurrent machine load (a Groq recording and other sessions were running), so
absolute values are an upper-ish bound and BELIEVED to vary by tens of percent on an idle machine.

| model | N=30 median / p95 | N=50 median / p95 |
|---|---|---|
| X4 (bge-reranker-base, 278M) | 16,669 / 17,733 ms | 27,992 / 31,128 ms |
| X2 (MiniLM-L12, 33M) | 5,495 / 6,397 ms | 9,057 / 9,753 ms |

Kernel-vs-CPU parity of X4 on 300 pairs: Spearman 0.99999544, max abs score difference 0.0107
(prereg bound 0.999: met). **Not serving-viable on CPU**: the spec's budget is 300 ms; even the
small model is ~18x over it at N=30 and X4 is ~56x over (16,669 / 300). Reaching the budget would
need a GPU, quantisation or a much smaller model, none of which was tried; a GPU-only gain is not a
CPU serving gain.

## Caveats

* Best-of-12 selection (and the preregistered tie order) makes the validation number optimistic;
  the test number (43.3%, +2.93) is the honest one.
* One seed per config (42); seed variance of the fine-tune was not measured. The 0.7-point
  differences between the top fusion candidates are within what one seed could move (BELIEVED).
* About 2.1k training pairs and a single repository (vscode). No claim for other repositories.
* Recall@50 is unchanged by construction (64.6% on test): the remaining 35.4 points to 100% are
  queries whose canonical is outside the top 50, i.e. retrieval, not ranking. 80% needs a better candidate generator first.
* The leakage guard is ID-level (every training anchor, positive and negative is a train-split
  issue); no near-duplicate-text audit of test queries against train issues was run beyond it.
* The pure-rerank result suggests the cross-encoder adds signal only when combined with the
  retriever rank; this was not decomposed further (BELIEVED, untested).

## Provenance

* Freeze: PR #376 (merge 4da3a3c), `reports/r9_xenc_rerank_frozen_2026-10.json`, `config_sha256`
  2d8e91360c77960e7fb92134fef24a0043aeae86db59b4c3330962a560e2f84b (passed as
  `--frozen-sha256`). It pins X4_rrf10, base BAAI/bge-reranker-base at revision 2cfc18c9, the sha256
  of all four X4 files (model.safetensors 94625c93...8804), RRF k=10, N=50, seed 42, the R8 freeze
  (config_sha256 3ad311b4...26c6d, F4 embedding 10339bfe...bc9d) and the harness / kernel code.
* Validation: Kaggle T4 kernel `triageiq-xenc-rerank-vscode` (private), 5,438 s total; X4 training
  1,180 s. Driver at bca249a (LF sha256 3a490f41...3d3f).
* Test scoring: private Kaggle T4 kernel `triageiq-xenc-score-test-vscode` v1 (scoring only, 85,250
  pairs, 686 s = 11.4 GPU-minutes), private dataset `triageiq-xenc-x4-test` (privacy read back:
  `isPrivate: true` / `is_private: true`); kernel recorded the frozen weight shas. Final evaluation
  on CPU (`CUDA_VISIBLE_DEVICES=-1`), main at 4da3a3c, exit 0, once:
  `reports/r9_xenc_rerank_test_2026-10.json`.
* Not wired into serving; README / EVALUATION numbers are unchanged. Any further use is GG's
  decision. The pre-registered R9 stage is complete; no retry under this registration.
