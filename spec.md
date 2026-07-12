# Project Spec: TriageIQ — Retrieval Quality Improvement (attack the ~23% base rate)

## Goal

Product-task retrieval (issue → genuinely-related issue) is the **weakest model in the pipeline**:
honest Recall@5 is **k8s 23.5% [18.4, 28.5], vscode 22.4% [17.7, 28.0]** — statistically
indistinguishable, both ~23%. The retriever surfaces the related issue in the top-5 roughly one time
in four. This was invisible until ADR-0028/0030 because k8s was measured on a PR→issue proxy and
vscode on a proxy-inflated number.

Everything previously tried on retrieval aimed at the wrong target: the W3 fine-tune (+3.5pp,
marginal, HELD), corpus growth (grew the *duplicate* stratum, not related pairs), and a cross-encoder
reranker (ADR-0006 — **rejected on the PROXY metric**, which is now known to be the wrong task).

The techniques with real headroom on **technical corpora** are untried. This iteration tries them,
in leverage order, each measured honestly and gated on the same bar.

## The bar (identical for every lever — no exceptions)

- Metric: **product-task Recall@5** on the honest product-task pairs, per repo, vs the current live
  v1 baseline (k8s 23.5% / vscode 22.4%).
- Also report R@1 and R@10 for shape, but **R@5 is the gate** (it's what the product surfaces).
- **Bootstrap CI (paired, same method as ADR-0027/phaseC)**. A lever SHIPS only if its CI on the
  improvement **excludes zero** — on **k8s** (n=277, powered) as the primary gate. vscode (n=254) is
  also powered here — report it; if a lever clears **both**, that's a strong ship. If it clears k8s
  only, that's a k8s-conditional ship (state it honestly, like the resolution bucket decision).
- A lever that doesn't clear the bar is **rejected and documented** — same as the reranker, the W3
  fine-tune, selective prediction. A negative is a valid outcome. **Do not p-hack, do not lower the
  bar, do not ship a marginal gain.**
- **Magnitude matters too**: at a ~23% base, a +2-3pp lift is polish (that's the reason the W3
  fine-tune was NO-GO'd). State the effect size, not just significance. A lever that clears the CI
  but only adds ~3pp should be reported as *statistically real but practically marginal* — the human
  decides if it's worth shipping.

## Eval data (already exists — no new mining)

- k8s: **277** in-live-index product-task pairs (`scripts/phaseC_k8s_live_product_eval.py` produces
  the baseline; reuse its pair-selection exactly).
- vscode: **254** product-task pairs (the honest set from ADR-0028's correction).
- Zero-leakage reasoning (ADR-0030): these evaluate an **untrained pretrained** embedder, so
  train/test split labels don't apply. **This reasoning holds ONLY for untrained models.** If a lever
  **trains** anything (reranker fine-tune, embedder fine-tune), the leakage question **returns** —
  a trained model MUST be evaluated on pairs disjoint from its training data. **Escalate before
  training anything on these pairs.**

## Levers (in this order — each gated before proceeding)

### Lever 1 — Hybrid BM25 + dense (highest value, lowest risk, no training)

**Hypothesis**: dense embeddings (BGE) systematically miss **exact-term matches** — error codes
(`ImagePullBackOff`, `CrashLoopBackOff`), stack traces, API names, file paths, CLI flags — which is
exactly what GitHub issue text is made of. BM25 catches lexical overlap that dense retrieval blurs
away. Hybrid fusion routinely gives large gains on technical corpora.

- Build BM25 over the same corpus the dense index covers (per repo, same document text).
- Fuse: **Reciprocal Rank Fusion (RRF)** as the default (robust, no score-normalization pitfalls).
  Optionally also try weighted score-fusion (normalized) and report both — RRF is the primary.
- **Tune the fusion parameter (k / weight) on a held-out slice, NOT on the test pairs** — tuning on
  the eval set is the classic leak. Split the product pairs into tune/test, or tune on a separate
  slice; **state exactly what was tuned on what**. If there's no clean tuning slice, use RRF's
  standard k=60 untuned and say so.
- Measure product-task R@5 (+R@1/R@10) per repo vs baseline, with paired bootstrap CI.
- **Diagnostic (do this regardless of outcome)**: on the pairs BM25 gets right that dense misses,
  what characterizes them? (Exact error codes? Rare API names?) This tells you *why* dense fails and
  informs everything downstream. Report examples.

### Lever 2 — Reranker on the PRODUCT task (the ADR-0006 retry — different metric this time)

**Only if Lever 1 is done and reported** (hybrid may change the candidate set the reranker sees —
rerank the *best available* first-stage).

- ADR-0006 rejected a cross-encoder reranker — **but on the PROXY metric** (PR→issue), which is now
  known to be the wrong task. Retry it against **product-task R@5**, over the best first-stage
  (hybrid if Lever 1 ships, else dense).
- Use an **off-the-shelf pretrained cross-encoder** (no training → the zero-leakage reasoning still
  holds, no escalation needed). If a *trained* reranker is proposed, **STOP and escalate** (leakage
  question returns).
- Rerank top-k (k=20-50) candidates → measure R@5 with paired bootstrap CI.
- **Report latency**: a reranker adds inference cost per query. State the added latency; a lever that
  doubles response time for +3pp is a bad trade. The human decides on the quality/latency curve.

### Lever 3 — Stronger base embedder (cheapest experiment, do if 1-2 underdeliver)

- Swap BGE for a stronger modern retrieval embedder, **pretrained, no fine-tuning** (keeps the
  zero-leakage reasoning). Re-embed the corpus, re-measure product-task R@5.
- Report: does the base rate move materially, and at what index-build/storage cost?
- Zero-cost constraint: use a model that runs locally / is freely available. No paid APIs.

## Scope

### In scope
- Levers 1-3 as above, each measured + gated + documented.
- The Lever-1 diagnostic (why does dense fail — what does BM25 recover?).
- If a lever ships: full deliberate index/model cutover (artifacts + MANIFEST + drift guard +
  rollback anchor + live verify) — a retrieval change alters `similar_issues` → it changes the frozen
  eval-set retrieval → **re-record + re-baseline** (escalate; this is the ADR-0010/Option-C path).
- ADR-0031: hypothesis, per-lever result (effect + CI + latency), ship/reject decision per lever,
  and the honest headline (did we move the ~23% base rate, and by how much).

### Out of scope
- **No training/fine-tuning of anything on the product-task pairs without escalation** (leakage).
- No new data mining (Phase C is NO-GO'd — this uses existing pairs).
- No reviving the held W3 fine-tune (it's marginal; that decision stands).
- No paid APIs / no ANTHROPIC_API_KEY (zero-cost).
- No changing the product-task metric to make a lever look better.

## Autonomy & escalation

CC runs each lever autonomously. **Escalate ONLY:**
1. **Each lever's result** (effect size + CI + latency where relevant) **before proceeding to the
   next lever** — so the human can stop early if a lever wins big or all are duds.
2. **Any proposal to TRAIN a model on the product pairs** (leakage question returns — do not proceed).
3. **The cutover** if a lever ships (index change → re-record + re-baseline + deploy).

## Hard rules

- **Ship only on CI-excludes-zero (k8s primary), and state the effect size** — at a ~23% base, a
  +3pp "significant" lift is *practically marginal*; report it as such and let the human decide.
- **Never tune on the test pairs.** State exactly what was tuned on what.
- **No training on the eval pairs without escalation** — the zero-leakage reasoning holds only for
  untrained models (ADR-0030).
- A rejected lever is a **valid, documented finding** (ADR-0006 / W3 / selective-prediction pattern).
- Report **latency** for any lever that adds inference cost.
- Branch only (`feat/retrieval-quality`); human merges. Zero-cost, local only.
  Claude Max — never ANTHROPIC_API_KEY. Don't touch aetherart-497918.

## Success criteria

- Baseline reproduced (k8s 23.5% / vscode 22.4%) before any lever — confirm the harness agrees.
- Each lever: product-task R@5 (+R@1/R@10) per repo, paired bootstrap CI vs baseline, effect size,
  latency where relevant. Tuning provenance stated.
- Lever-1 diagnostic: what BM25 recovers that dense misses (with examples) — the *why*.
- Ship/reject per lever on the stated bar; rejections documented as findings.
- ADR-0031 + `reports/retrieval_quality.json` (reproducible byte-identically).
- If shipping: full cutover plan (escalated) incl. re-record + re-baseline.

## Build order (CC autonomous, escalate at each lever's result)

1. Reproduce the baseline on the existing harness (sanity: k8s 23.5% / vscode 22.4%).
2. **Lever 1 — Hybrid BM25 + RRF.** Measure, CI, diagnostic. **ESCALATE the result.**
3. **Lever 2 — Pretrained cross-encoder reranker on product task** (over the best first-stage).
   Measure, CI, latency. **ESCALATE.**
4. **Lever 3 — Stronger pretrained embedder.** Measure, CI, cost. **ESCALATE.**
5. ADR-0031 + the cutover plan for whatever ships (escalate the cutover).
```

