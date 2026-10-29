# Q5a part 3 results: fine-tuned dense retriever for vscode duplicates (2026-10-29)

Design and selection rule: `docs/Q5A_FINETUNE_PREREG_2026-10.md` (committed before any run).
Reference: the frozen #345 winner H(k=10) (`docs/Q5A_DENSE_RESULTS_2026-10.md`). Target (ADR-0066):
test Recall@5 >= 80% as a point estimate. **Not met (40.4%).** H1 (fine-tuning beats the #345
winner) **holds on validation and on the single test evaluation.**

## Selection (validation, 1,156 queries, ceiling 92.8%)

All four configs finished (3 epochs, batch 32, 2,069 train pairs). 16 candidates, selected by the
preregistered rule: **F4 (lr 3e-5, hard negatives) + RRF k=10** with the frozen #345 BM25 variant
(title x3 + error lines). Pipeline parity (untouched base model in the kernel vs #345 vectors, 2,000
issues): min cosine 0.999993, 0 below 0.999.

| | R@1 | R@5 (Wilson 95%) | R@10 | R@50 | MRR |
|---|---|---|---|---|---|
| #345 winner H(k=10) | 24.3% | 40.3% [37.5, 43.2] | 47.1% | 57.6% | 0.319 |
| F4 + RRF k=10 (selected) | 26.3% | **45.3% [42.5, 48.2]** | 52.9% | 64.9% | 0.350 |
| paired difference | +2.0 [+0.8, +3.3] | **+5.0 [+3.5, +6.6]** | +5.8 [+4.3, +7.4] | +7.3 [+5.6, +8.9] | +0.031 [+0.022, +0.040] |

Success rule (gain >= +2.0 points and CI lower bound > 0): met. All 16 candidates, including the
weakest (F1 RRF k=60, 42.2%), exceed the 40.3% reference, so the direction of the effect does not
depend on the best-of-16 pick; the size of the selected gain does. R@5 per config (RRF k=10): F1
44.2, F2 44.4, F3 44.7, F4 45.3; dense-only: F1 42.5, F2 42.6, F3 43.0, F4 44.3. Hard negatives and
the higher learning rate each helped slightly; differences between configs are about 1 point and
were not tested for significance (n=1,156).

## Test, single frozen evaluation (1,705 queries, ceiling 95.8%)

| | R@1 | R@5 (Wilson 95%) | R@10 | R@50 | MRR |
|---|---|---|---|---|---|
| #345 winner H(k=10) | 17.8% | 33.0% [30.8, 35.3] | 40.5% | 55.4% | 0.252 |
| F4 + RRF k=10 (frozen) | 19.9% | **40.4% [38.0, 42.7]** | 47.7% | 64.6% | 0.294 |
| paired difference | +2.1 [+0.9, +3.2] | **+7.3 [+5.9, +8.9]** | +7.2 [+5.7, +8.7] | +9.3 [+7.7, +10.9] | +0.043 [+0.034, +0.052] |

The #345 winner reproduced exactly on the same code path before F4 was scored: 563 / 1,705 hits at
R@5 (33.0%), equal to the published count (the driver aborts otherwise). The preregistered test
criterion (paired R@5 CI excludes zero) is met; the 80% target is not met: 40.4% is half the
target and 55 points under the 95.8% ceiling.

## Caveats

* Best-of-16 selection inflates the validation number (45.3%); the test number is the honest one.
  Here the test gain (+7.3) is larger than the validation gain (+5.0), the opposite of what
  selection optimism predicts. BELIEVED: split differences (the #345 baselines are also lower on
  test than validation: BM25 21.5% vs 29.2%); not tested. The two paired CIs overlap
  ([+3.5, +6.6] vs [+5.9, +8.9]); no claim is made that the test gain is larger in truth.
* One seed per config (42); seed variance of the fine-tune was not measured.
* About 2.1k training pairs and a single repository (vscode). No claim for other repositories.
* Dense-only fine-tuned retrieval was evaluated on validation only (not part of the test set).
* Recall@50 of the selected stack is 64.6% on test, which bounds any reranker over the top 50.
* The leakage guard is ID-level (every training anchor, positive and negative is a train-split
  issue); no near-duplicate-text audit of test queries against train issues was run beyond it.

## Provenance

* Validation run: driver at 082c28b (`scripts/r8_dup_finetune.py`, LF sha256 c401231e...2a3a);
  output `reports/r8_vscode_dup_finetune_validation_2026-10.json`.
* Freeze: PR #354 (merge f5a94a2), `reports/r8_vscode_dup_finetune_frozen_2026-10.json`,
  `config_sha256` 3ad311b449c0b0d08f15746a8a15c382fec9e45b9b0c36667d69cb3d1c426c6d (the value passed
  as `--frozen-sha256`); it pins the F4 embedding (sha256 10339bfe...bc9d), the F4 weights
  (0593b94d...e70c) and the #345 harness. Fine-tuning: Kaggle T4, base BAAI/bge-base-en-v1.5 at
  revision a5beb1e3.
* Test run: main at f5a94a2, `reports/r8_vscode_dup_finetune_test_2026-10.json`, run once on CPU
  (`CUDA_VISIBLE_DEVICES=-1`), exit 0.
* Not wired into serving; README / EVALUATION numbers are unchanged. Any index build with the F4
  model, or a serving cutover, is GG's decision.
