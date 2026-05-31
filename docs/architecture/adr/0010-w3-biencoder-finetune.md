# ADR-0010 — W3: Fine-tune BGE bi-encoder for similar-issue retrieval

## Context

Baseline BAAI/bge-base-en-v1.5 (zero-shot) achieved R@5 of 0.4102 (k8s) and 0.3674 (vsc) on the gold
similar-issue pairs (ADR-0008). ADR-0006 rejected cross-encoder reranking as a latency-neutral path
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
Proxy val R@5 peaked at epoch 0 (0.8500 on small distractor corpus), slight overfit in epochs 2-5 —
best checkpoint saved at step 65.

Combined model (both repos jointly) chosen over per-repo split: vsc-only training could not complete
due to GPU state degradation from sequential training in the same process. Per-repo would have required
separate processes. Given the combined model's strong performance, the per-repo experiment was deprioritised.

**T5 — Evaluation**: Full-corpus FAISS rebuild with fine-tuned model. Test split (152 k8s + 60 vsc pairs)
+ n=100 seed=42 sample (W1.3 comparable protocol). Self-exclusion applied (query issue excluded from
retrieval, matching baseline methodology). Bootstrap 95% CI (n=2,000) on delta.

## Consequences

**Results (n=100 seed=42 sample, self-excluded)**:

| Repo | Baseline R@5 | Fine-tuned R@5 | Delta | CI 95% | Verdict |
|---|---|---|---|---|---|
| kubernetes_kubernetes | 0.4102 | 0.6700 | +25.98 pp | [+15.98, +34.98] | PASS |
| microsoft_vscode | 0.3674 | 0.5000 | +13.26 pp | [+3.26, +23.26] | PASS |

**Test split (152 k8s, 60 vsc)**:

| Repo | R@5 | R@1 |
|---|---|---|
| kubernetes_kubernetes | 0.6579 | 0.4737 |
| microsoft_vscode | 0.8167 | 0.5000 |

**Overall verdict: TRACK_A_SUCCESS** — ≥3 pp R@5 BOTH repos, CI lower bound > 0 on both.

**CPU latency**: No regression. Same architecture (BGE-base 86M params), same FAISS IndexFlatIP;
only weights changed.

**What worked**:
- Hard negatives from the existing FAISS top-50 were effective (rank-1 negatives, 15% soft-positive rate tolerated)
- Even 1,028 training triplets (1 neg/pair, epoch 0 only) gave massive improvement
- Combined model generalises across both repos despite vocabulary differences

**What didn't work / open questions**:
- Per-repo training blocked by GPU state leak in sequential training (same process); fix documented in T4 script
- k8s-only model (epoch 0 only due to kill) showed proxy val R@5=0.8533 vs combined 0.8500 — slightly higher on k8s val, but not deployed
- Proxy val R@5 peaked at epoch 0, declining afterwards — small dataset (1,028 triplets) overfits quickly

**Alternatives considered**:
- Track B (ms-marco-MiniLM-L-6-v2 CE reranker, 22M): not attempted — Track A sufficient
- Per-repo fine-tuning: not completed — combined model covers both repos, per-repo deferred
- More hard negatives per pair (10 instead of 1): not attempted — single hardest negative was sufficient

## Scripts

- `scripts/w3_t2_mine_negatives.py` — hard-negative mining
- `scripts/w3_t3_split.py` — connected-component temporal split
- `scripts/w3_t4_train.py` — fine-tuning with direct HF loop
- `scripts/w3_t5_eval.py` — evaluation + bootstrap CI

## Data / model artifacts

- `data/w3_hard_negatives.parquet` — 14,350 hard-negative records
- `data/w3_split.parquet` — 1,400 pair split assignments
- `data/models/bge_finetuned_combined/` — fine-tuned SentenceTransformer (best val checkpoint)
- `data/models/bge_finetuned_k8s_index/` — FAISS index built on fine-tuned model (k8s)
- `data/models/bge_finetuned_vsc_index/` — FAISS index built on fine-tuned model (vsc)
- `reports/w3_t4_val_results.json` — T4 variant comparison
- `reports/w3_t5_eval_results.json` — full T5 eval table
