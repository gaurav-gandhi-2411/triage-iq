# ADR-0016 — W3: Fine-tune BGE bi-encoder for similar-issue retrieval

**Status:** Proposed — re-verification in progress on current main
**Date:** 2026-07-04 (rebased from a draft originally numbered ADR-0010 on `feat/w3-finetune`, 2026-05-31)

---

## Rebase note (2026-07-04)

This ADR was originally drafted as `0010-w3-biencoder-finetune.md` on `feat/w3-finetune`, a branch
that diverged from main 47 commits ago (before CQR, the eval-regression-gate, the model-artifact
drift guard, and the grounding verifier landed). That branch's number collided with main's current
`0010-conformal-quantile-regression.md`. Renumbered to 0016, the next free slot.

More importantly: the branch's fine-tuned model weights were never committed
(`data/models/**` is gitignored except the manifest), so the original +13.16pp k8s / +13.33pp vsc
R@5 result cannot simply be carried over — it must be reproduced end-to-end on current main before
it is claimed or shipped. The scripts (`scripts/w3_t2_mine_negatives.py`, `w3_t3_split.py`,
`w3_t4_train.py`, `w3_t5_eval.py`) and their tracked inputs (`data/w3_hard_negatives.parquet`,
`data/w3_split.parquet`) were ported as-is — `data/gold_related.parquet` was confirmed unchanged
(1,435 pairs) since the original run, so those mining/split outputs remain valid. The model itself
is being retrained and re-evaluated fresh; **the Results section below is pending until that
re-verification completes.**

---

## Context

Baseline BAAI/bge-base-en-v1.5 (zero-shot) achieved R@5 of 0.4102 (k8s) and 0.3674 (vsc) on the full
gold similar-issue pairs (ADR-0008). ADR-0006 rejected cross-encoder reranking as a latency-neutral path
to improvement. The next hypothesis: domain-specific fine-tuning of the bi-encoder itself should
lift recall with zero added inference latency.

**Gold dataset**: 1,435 positive pairs (`data/gold_related.parquet`) built from PR→issue references
and text-similarity labelling across two repos (1,024 k8s, 411 vsc). Sources: body_ref (4), body_related
(1,010), title_sim (421). Confidence: 1,431 medium, 4 high.

## Decision

Fine-tune BAAI/bge-base-en-v1.5 on domain gold pairs using MultipleNegativesRankingLoss with
explicit hard negatives.

**T2 — Hard-negative mining**: For each positive pair, query the pre-built BGE FAISS top-50 and
collect the rank-1 non-positive result as the hard negative. 1,028 training triplets after
connected-component split (35 cross-repo-contaminated pairs dropped). Spot-check: 15% soft-positive
rate — below the 20% abort threshold.

**T3 — Split**: Connected-component temporal split (networkx). Components sorted chronologically
by earliest created_at. Greedy state-machine assignment at 70/15/15. k8s: 704/150/152,
vsc: 324/10/60. Zero-leakage verified (no issue_id in multiple splits). vsc val=10 due to
a large late-period component cluster — acceptable for the proxy early-stop metric; real test uses 60 pairs.

**T4 — Training**: Direct HuggingFace training loop (sentence-transformers 2.7.0's `model.fit()` had
130s/step overhead due to per-sequence CUDA kernel recompilation with smart batching). Pre-tokenised
TripletDataset, AdamW lr=2e-5, warmup 10%, batch=16, epochs=5, temp=0.05, max_len=128, seed=42.

**T5 — Evaluation**: Full-corpus FAISS rebuild with fine-tuned model. Baseline computed on the same
test-split pairs using the pre-built dup_index_*_bge index. Self-exclusion applied (query issue excluded
from retrieval, matching baseline methodology). Bootstrap 95% paired CI (n=2,000) on delta.

## Consequences

**Results (test split, zero training overlap):** PENDING re-verification on current main — see
Rebase note above. The prior draft's numbers (kept here for provenance only, NOT a current claim):

| Repo | n | Baseline R@5 | Fine-tuned R@5 | Delta | CI 95% | Verdict |
|---|---|---|---|---|---|---|
| kubernetes_kubernetes | 152 | 0.5263 | 0.6579 | +13.16 pp | [+6.58, +19.74] | PASS (stale — pre-rebase) |
| microsoft_vscode | 60 | 0.6833 | 0.8167 | +13.33 pp | [+5.00, +23.33] | PASS (stale — pre-rebase) |

This table will be replaced with the re-established numbers once T4/T5 complete on current main.
Decision gate (per ADR-0006's bar, which rejected the cross-encoder reranker): CI must exclude zero
on both repos to ship.

**CPU latency**: No regression expected. Same architecture (BGE-base 86M params), same FAISS
IndexFlatIP; only weights change.

**What worked (from the prior run, being re-verified)**:
- Hard negatives from the existing FAISS top-50 were effective (rank-1 negatives, 15% soft-positive rate tolerated)
- Even 1,028 training triplets (1 neg/pair, epoch 0 only) gave substantial improvement
- Combined model generalises across both repos despite vocabulary differences

**What didn't work / open questions (from the prior run)**:
- Per-repo training blocked by GPU state leak in sequential training (same process); fix documented in T4 script
- Proxy val R@5 peaked at epoch 0, declining afterwards — small dataset (1,028 triplets) overfits quickly

**Alternatives considered**:
- Track B (ms-marco-MiniLM-L-6-v2 CE reranker, 22M): not attempted — Track A sufficient
- Per-repo fine-tuning: not completed — combined model covers both repos, per-repo deferred
- More hard negatives per pair (10 instead of 1): not attempted — single hardest negative was sufficient

---

## Correction note (2026-05-31, carried over from the original draft)

**Pre-merge verification caught eval contamination in the original T5 run.**

The initial T5 eval script used `sample_gold(gold, repo, n=100, seed=42)` to draw the evaluation
sample. This function sampled from `data/gold_related.parquet` — the **full gold corpus**, not
filtered to held-out splits. Since the 70/15/15 temporal split assigned ~70% of gold pairs to
training, any random draw from full gold is ~70% training data.

Measured contamination: 66/100 k8s eval pairs and 71/100 vsc eval pairs were in-training, inflating
the reported delta to ~2x the true value (originally reported +25.98 pp k8s / +13.26 pp vsc).

**Fix applied**: `sample_gold` removed from T5. Canonical eval is test-split pairs only
(k8s n=152, vsc n=60, confirmed zero overlap with training data). Baseline computed live on the
same query set using the pre-built `dup_index_*_bge` FAISS indexes. An `assert_eval_disjoint_from_train`
guard prevents silent reintroduction of this bug — kept intact through the rebase.

This correction is kept permanently — it is evidence of the verification discipline applied before
merge, not an embarrassment to hide.

---

## Scripts

- `scripts/w3_t2_mine_negatives.py` — hard-negative mining
- `scripts/w3_t3_split.py` — connected-component temporal split
- `scripts/w3_t4_train.py` — fine-tuning with direct HF loop
- `scripts/w3_t5_eval.py` — evaluation + bootstrap CI (leakage-guarded)

## Data / model artifacts

- `data/w3_hard_negatives.parquet` — 14,350 hard-negative records (ported, gold_related.parquet unchanged)
- `data/w3_split.parquet` — 1,400 pair split assignments (ported, gold_related.parquet unchanged)
- `data/models/bge_finetuned_combined/` — fine-tuned SentenceTransformer (regenerated on current main, not committed — gitignored)
- `data/models/bge_finetuned_k8s_index/` — FAISS index built on fine-tuned model (k8s, regenerated)
- `data/models/bge_finetuned_vsc_index/` — FAISS index built on fine-tuned model (vsc, regenerated)
- `reports/w3_t4_val_results.json` — T4 variant comparison (regenerated)
- `reports/w3_t5_eval_results.json` — full T5 eval table (regenerated)
