# ADR-0031 — Retrieval Quality Improvement (three levers, none ship)

**Status:** Rejected (all three levers)
**Date:** 2026-07-12
**Decider:** Gaurav Gandhi

---

## Context

ADR-0030 established that product-task retrieval (issue → genuinely-related issue) is the
weakest model in the pipeline: honest Recall@5 is **k8s 23.5% [18.4, 28.5]**, **vscode 22.4%
[17.7, 28.0]** — statistically indistinguishable, both ~23%. The retriever surfaces the
related issue in the top-5 roughly one time in four. Everything previously tried on
retrieval targeted the wrong metric (PR→issue proxy, ADR-0006's reranker) or the wrong
stratum (corpus growth grew duplicates, not related pairs). This ADR tries three untried,
high-headroom, zero-training levers against the corrected metric, each gated on the same
bar (spec.md): **paired bootstrap CI on product-task R@5 must exclude zero on k8s (n=277,
primary)**; vscode also reported; magnitude stated alongside significance so a
statistically-real-but-marginal lift (the W3 fine-tune pattern, +3.5pp NO-GO'd) isn't
shipped just because it clears a CI.

## Baseline reproduction — a correction before any lever ran

Reproducing ADR-0030's numbers surfaced a discrepancy: **k8s reproduces byte-identically**
(23.47% ≈ 23.5%, n=277, CI[18.4,28.5], via the existing
`scripts/phaseC_k8s_live_product_eval.py`). **vscode's reported 22.4% (n=254, ADR-0028) does
not.** No script ever computed that number — it exists only as a hardcoded literal
(`phaseC_k8s_live_product_eval.py:55`, before this ADR's changes) traced to a one-off
ADR-0028 audit JSON with an unexplained denominator of 281, against a product stratum that
is actually 505 rows in `gold_related_v2.parquet` (unchanged since a single commit,
predating ADR-0028). Filtering by channel (`legacy_gold_v1` alone: 288 rows, all in-range),
confidence, or query dedup does not reconstruct 254/281.

`scripts/phaseC_vscode_live_product_eval.py` was written — the vscode equivalent of the k8s
script, identical method (product-stratum pairs filtered by live-index membership, same
percentile bootstrap) — and reproduces **n=292, R@5=26.71%, CI[21.6, 31.9]**. Adopted as the
working vscode baseline for this ADR's lever comparisons: it's the only vscode number that's
byte-reproducible from committed code and data. The old 254/22.44% number's provenance is
undocumented and not recoverable; flagged, not silently carried forward.

## The bar

Identical for every lever: paired bootstrap CI (2000 resamples, seed=42, same method as
`scripts/w3_t5_eval.py`'s ADR-0027-corrected paired bootstrap) on the R@5 delta vs the
reproduced baseline above. Ships only if the CI excludes zero on k8s. A lever that clears
CI but with a small magnitude (~3pp on a ~23-27% base, the W3 fine-tune's pattern) is
reported as statistically real but practically marginal — not auto-shipped.

---

## Lever 1 — Hybrid BM25 + dense (RRF)

**Hypothesis:** dense embeddings miss exact-term matches (error codes, stack traces, API
names) that BM25 catches. **Method:** BM25 (`rank_bm25.BM25Okapi`) built over the exact same
corpus text/issue-number set as the live dense index; candidate pool 100 per system; RRF
fusion, k=60 **untuned** (no held-out tuning slice exists without either shrinking the
already-thin powered eval set or leaking onto the test pairs — spec.md's stated fallback).
Weighted score-fusion (0.5/0.5 min-max normalized, also untuned) computed as a secondary
comparison.

| Variant | k8s R@5 delta | k8s CI | vscode R@5 delta | vscode CI | Ships? |
|---|---|---|---|---|---|
| RRF (primary) | +0.72pp | [-2.89, 4.69] | +2.05pp | [-1.37, 5.82] | **No** — CI crosses zero, both repos |
| Weighted fusion (secondary) | +3.25pp | [0.35, 6.5] | +4.11pp | [1.03, 7.19] | CI excludes zero, both repos — **rejected on magnitude** (see below) |

**Diagnostic (why dense fails):** BM25 recovers the target in 15/277 (k8s, 5.4%) and 19/292
(vscode, 6.5%) cases dense misses at R@5. k8s recoveries are largely genuine — shared flaky
test names (`TestReadFromFile`), exact API/config terms (`tokencontroller`, `kubeconfig`).
vscode recoveries are mixed — some genuine (shared stack-trace hashes, error strings), several
are boilerplate-template overlap between AI-chat bug reports (shared tokens like
`sonnet`/`gpt`/`claude` from report scaffolding, not topical relatedness) — a real but
partly noisy signal.

**Decision:** RRF (the spec's stated primary method) rejected outright — CI crosses zero on
the primary gate. Weighted fusion technically clears CI on both repos, but the k8s lower
bound (0.35pp) is barely above zero and the magnitude (+3-4pp) matches the already-NO-GO'd
W3 fine-tune pattern. Escalated to the user; **decision: reject both variants** — not worth
the added BM25 build/query cost (~270ms/query k8s, ~72ms/query vscode) for a marginal,
borderline-significant gain.

Raw results: `reports/lever1_hybrid_bm25_rrf.json`. Script: `scripts/lever1_hybrid_bm25_rrf.py`.

---

## Lever 2 — Pretrained cross-encoder reranker (the ADR-0006 retry)

**Hypothesis:** ADR-0006 rejected 7 off-the-shelf cross-encoders, but against the PR→issue
proxy metric — now known to be the wrong task (ADR-0030). Retrying the strongest prior
candidate against the honest product-task metric might reverse the verdict.

**Method:** `BAAI/bge-reranker-v2-m3` (the only ADR-0006 candidate that improved either repo
at all, later killed on its own robustness recheck) reranking the top-30 dense candidates
(best available first stage — Lever 1 was rejected). No training, zero-leakage reasoning
unchanged. Recall computed on GPU (correctness is device-independent); latency measured
separately on a CPU-forced 30-query subsample per repo, since the deployment target is
CPU-only inference — confirmed with an isolated, uncontended CPU-only timing check to rule
out a measurement artifact from running two model instances concurrently.

| Repo | Dense R@5 | Reranked R@5 | Delta | CI | Ships? |
|---|---|---|---|---|---|
| k8s | 23.47% | 21.30% | **-2.17pp** | [-6.14, 1.81] | No |
| vscode | 26.71% | 22.95% | **-3.77pp** | [-9.25, 1.71] | No |

**Latency:** CPU rerank adds **~18-60 seconds per query** (190-330x the ~50-320ms dense-only
baseline) for a 30-candidate pool — independently disqualifying even if quality had improved.

**Decision:** Rejected on both grounds. Quality moves negative on both repos (CI crosses
zero, so no significant effect either direction, but the point estimate is a regression).
Consistent with ADR-0006's own T2 finding that this exact model's k8s gain (+6pp at n=100)
vanished at n=300 (CI[-0.037,+0.053]) — pretrained cross-encoders do not generalize to this
corpus, on either the proxy or the corrected honest metric.

Raw results: `reports/lever2_reranker.json`. Script: `scripts/lever2_reranker.py`.

---

## Lever 3 — Stronger pretrained embedder (no fine-tuning)

**Hypothesis:** cheapest experiment — does a larger, same-family pretrained embedder
(`BAAI/bge-large-en-v1.5`, 335M params/1024-dim, vs the deployed `bge-base-en-v1.5`, 109M
params/768-dim) move the base rate without any training?

**Method:** corpus re-embedded verbatim (same texts/issue-numbers copied from the loaded
baseline retriever, not re-derived) so the embedding model is the only variable changed.

| Repo | BGE-base R@5 | BGE-large R@5 | Delta | CI | Ships? |
|---|---|---|---|---|---|
| k8s | 23.47% | 24.55% | +1.08pp | [-2.17, 4.33] | No |
| vscode | 26.71% | 30.48% | +3.77pp | [-0.34, 7.88] | No — CI lower bound negative |

**Cost if it had shipped:** 33% larger embedding dim, ~2x index size (58.6MB k8s / 27.5MB
vscode at float32), ~2min one-time corpus re-embed, +~75% query-encode latency (small in
absolute terms: ~30-37ms → ~50-65ms).

**Decision:** Rejected — CI crosses zero on both repos. A bigger same-family pretrained
embedder does not move the base rate materially.

Raw results: `reports/lever3_stronger_embedder.json`. Script: `scripts/lever3_stronger_embedder.py`.

---

## Decision — no lever ships

All three levers, and Lever 1's secondary weighted-fusion variant, are rejected. Summary:

| Lever | k8s delta (CI) | vscode delta (CI) | Verdict |
|---|---|---|---|
| 1a — RRF (primary) | +0.72pp [-2.89, 4.69] | +2.05pp [-1.37, 5.82] | CI crosses zero — rejected |
| 1b — weighted fusion | +3.25pp [0.35, 6.5] | +4.11pp [1.03, 7.19] | CI excludes zero, magnitude marginal — rejected on judgment |
| 2 — reranker | -2.17pp [-6.14, 1.81] | -3.77pp [-9.25, 1.71] | Quality regression + 190-330x latency — rejected |
| 3 — stronger embedder | +1.08pp [-2.17, 4.33] | +3.77pp [-0.34, 7.88] | CI crosses zero — rejected |

**Headline: the ~23-27% product-task base rate does not move with any pretrained,
zero-training lever tried here.** Retrieval quality remains the weakest model in the
pipeline, unchanged by this iteration. No index/model cutover is needed (nothing ships) —
the live v1 index (BGE-base FAISS) stays as-is.

## Consequences

**Positive (negative results are still results, per the ADR-0006/W3/selective-prediction
pattern):**
- Three real, previously-untried levers are now measured and closed out, not open questions.
- The vscode baseline provenance gap is fixed going forward: `phaseC_vscode_live_product_eval.py`
  is now a committed, reproducible script (there was none before this ADR).
- The Lever 1 diagnostic gives a concrete, evidenced answer to "why does dense retrieval
  fail here" (exact-term/API/test-name matches dense blurs) even though the fix (BM25 hybrid)
  doesn't clear the bar — useful signal for any future retrieval work.
- Lever 2 independently confirms ADR-0006's conclusion under the corrected metric: pretrained
  cross-encoders do not generalize to this corpus, closing that question more firmly than
  before (previously only tested on the wrong task).

**What changes:** none of the shipped artifacts change. The live index, the served model,
and the API surface are untouched.

**What becomes harder:** the "easy" levers (no training, no new data) are now exhausted.
Per spec.md's out-of-scope list, the two remaining paths — in-domain fine-tuning (reopens
the leakage question ADR-0030 deliberately kept closed) or new related-pair mining (already
NO-GO'd by ADR-0030 on value grounds, a ~23% base rate doesn't justify the mining cost) —
are both harder, slower, and were explicitly deferred by prior ADRs. Closing the retrieval
gap now requires picking up one of those deferred, harder paths, or accepting the ~23%
base rate as a durable product characteristic rather than a fixable bug.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Ship weighted score-fusion (Lever 1b) anyway, given CI excludes zero on both repos | Escalated explicitly; user chose reject. Magnitude (+3-4pp) matches the already-NO-GO'd W3 fine-tune pattern; k8s CI lower bound (0.35pp) is barely above zero — a fragile ship. |
| Tune Lever 1's fusion weight before rejecting | Would require carving a tuning slice out of an already-thin powered eval set, or tuning on the test pairs (the leak spec.md rules out). Not attempted. |
| Screen additional cross-encoders for Lever 2 (a third ADR-0006 slate) | Time-boxed: the other 6 candidates already failed both repos by wide margins across two prior slates; re-running them on the honest metric was judged low-value versus retrying only the one candidate whose failure could plausibly be metric-attributable. |
| Screen additional embedders for Lever 3 | Spec framed this as the cheapest, single-swap experiment; a full embedder bake-off is a separate, larger effort not scoped here. |

## Reproduce

```
python scripts/phaseC_k8s_live_product_eval.py
python scripts/phaseC_vscode_live_product_eval.py
python scripts/lever1_hybrid_bm25_rrf.py
python scripts/lever2_reranker.py
python scripts/lever3_stronger_embedder.py
python scripts/assemble_retrieval_quality_report.py
```

Master report: `reports/retrieval_quality.json`.
