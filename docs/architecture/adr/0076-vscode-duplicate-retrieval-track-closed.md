# ADR-0076 — vscode duplicate retrieval: the Recall@5 >= 80% target is closed as infeasible; best achievable 43.3%

Status: Accepted (GG decision 2026-10-30, D96.1)
Date: 2026-10-30
Decider: Gaurav Gandhi; recorded by the unattended run

Closes the target set by ADR-0066 (vscode duplicate Recall@5 >= 80% on test, point estimate) under the infeasible-target rule
of spec-quality-push §3: "If a target is shown infeasible by evidence (e.g. ceiling below target), stop that target, report the
evidence and the best achievable number, and continue."

## Context

Three pre-registered levers were each run to a single frozen test evaluation on the same 1,705 vscode test queries. Each reproduced
the previous stage's test number exactly before it was scored.

| stage | test R@5 (Wilson 95%) | step vs previous | source |
|---|---|---|---|
| BM25 baseline B2 (= #309) | 21.5% [19.6, 23.5] | - | `docs/Q5A_DENSE_RESULTS_2026-10.md` |
| BM25 tuning (title x3 + error lines) + dense, RRF k=10 (#345 winner) | 33.0% [30.8, 35.3] | +11.5 pts (difference of point estimates) | `docs/Q5A_DENSE_RESULTS_2026-10.md` |
| fine-tuned dense F4 + RRF k=10 (R8, #356) | 40.4% [38.0, 42.7] | +7.3 pts paired [+5.9, +8.9] | `docs/Q5A_FINETUNE_RESULTS_2026-10.md` |
| cross-encoder X4 fused by RRF k=10 (R9, #377) | **43.3% [41.0, 45.6]** | +2.93 pts paired [+1.52, +4.40] | `docs/Q5B_XENC_RERANK_RESULTS_2026-10.md`, `reports/r9_xenc_rerank_test_2026-10.json` |

The evidence that the target is not reachable with the means this project has:

- **Diminishing steps.** +11.5, +7.3 and +2.9 points. The last lever is the most expensive and gave the smallest gain; 80% is 36.7
  points above the best result.
- **The reranker gain comes only from fusion.** On validation every cross-encoder used alone scored below the retriever it reranks
  (best 477 vs 524 hits of 1,156); only rank fusion with the retriever beat it.
- **The best stack has a hard ceiling under the target.** Its test Recall@50 is 64.6% (1,102 of 1,705). Any reranker over its
  top 50 can reach at most 64.6% R@5, which is below 80%. The index-coverage ceiling is 95.8%, so the limit is in recall, not
  answerability.
- **The best stack cannot be served.** On CPU (the serving target), the X4 rerank stage takes 16,669 ms median per query at N=30.
  The smaller MiniLM model (X2) takes 5,495 ms. Both are against a 300 ms budget, and both were measured while a recording was
  running on the same machine (`docs/Q5B_XENC_RERANK_RESULTS_2026-10.md`).

## Decision

1. The ADR-0066 vscode duplicate Recall@5 >= 80% target is closed as infeasible by evidence. **Best achievable: 43.3% [41.0, 45.6]
   test R@5** (X4_rrf10, offline, single frozen evaluation, one seed, selected best-of-12 on validation).
2. No further lever is pursued, and there is **no distillation or quantisation** of the reranker. To serve, a distilled model would
   need to cut latency by about 56x and still keep a +2.9-point gain that is entirely a fusion effect. GG judges that it would
   likely lose that gain. This is a judgment (BELIEVED), not a measurement.
3. Nothing from R8 or R9 is served. This ADR does not change production retrieval.

## Consequences

- Public copy reports the vscode duplicate retrieval target as not met, with 43.3% [41.0, 45.6] as the best offline result.
  The served number is reported separately; R8 and R9 are offline results.
- R8 and R9 artifacts stay on D: as the reproducibility record: `models\microsoft__vscode\dup_finetune\` (~2.3 GB),
  `xenc_rerank\` (2.4 GB) and `xenc_rerank_test\`. They are also kept in the private Kaggle datasets and kernels.
- Reopening needs a new GG decision, made on new evidence, for example a recall-stage change that lifts the 64.6% R@50 ceiling.
  More reranking does not reopen it.

## Alternatives

- **Distil or quantise the cross-encoder for CPU serving:** rejected (decision 2).
- **Keep the target open and try more rerankers:** rejected, because of the R@50 ceiling and the diminishing steps.
- **Report only the served number:** rejected, because the offline best is the honest answer to "how far can this approach go"
  and is stated alongside the served number.
