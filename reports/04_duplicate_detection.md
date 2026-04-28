# System 2 — Duplicate Issue Detection

**Version:** Day 5  
**Last updated:** 2026-04-28  
**Maintainer:** Gaurav Gandhi

---

## 1. Architecture

**Task:** Given a new GitHub issue, retrieve the top-K most similar existing issues. Enables duplicate detection, related-issue surfacing, and context provision to triagers.

**Approach:** Sentence embeddings + FAISS inner-product (cosine) retrieval. One index per repo (different vocabulary domains).

```
Issue text (title + body[:512])
→ SentenceTransformer encoder (BGE-base or MiniLM-L6)
→ L2-normalized embedding
→ FAISS IndexFlatIP (exact cosine search)
→ Top-K results with similarity scores
```

**Models evaluated:**

| Model | Dimensions | Size on disk | Use case |
|---|---|---|---|
| `BAAI/bge-base-en-v1.5` | 768 | 24–50 MB/repo | Higher accuracy, technical-text optimized |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 13–27 MB/repo | Faster, smaller footprint |

**Infrastructure:** CPU inference. GPU used for index construction only (speed). Retrieval via `faiss.IndexFlatIP` — exact search, no quantization.

---

## 2. Gold Standard Construction

### 2.1 Methodology

No comments data was scraped — GitHub's "duplicate of #X" annotations live in comments, not issue bodies. Gold pairs were built from three signals in priority order:

| Source | Signal | Repos | Count | Confidence |
|---|---|---|---|---|
| `body_ref` | Explicit `"duplicate of #N"` in issue body | vscode | 1 | High |
| `body_ref` | Explicit `"duplicate of #N"` in issue body | kubernetes | 3 | High |
| `body_related` | `"Fixes #N"`, `"Closes #N"`, `"See #N"` in body | vscode | 112 | Medium |
| `body_related` | `"Fixes #N"`, `"Closes #N"`, `"See #N"` in body | kubernetes | 898 | Medium |
| `title_sim` | TF-IDF cosine ≥ 0.45 on title + body[:200] | vscode | 298 | Medium |
| `title_sim` | TF-IDF cosine ≥ 0.45 on title + body[:200] | kubernetes | 123 | Medium |

**Filters applied:**
- Referenced issue must exist in scraped corpus
- Query issue must be NEWER than original (temporal ordering)
- Both issues must have body length > 10 characters

**Final counts:** 411 vscode pairs, 1,024 kubernetes pairs → **1,435 total**

### 2.2 Gold Standard Caveats

**`body_related` pairs are related, not strict duplicates.** "Fixes #X" means the query issue references #X as the thing being addressed — often a PR closing an issue, or a follow-up. These pairs are legitimate "related issue" signal but weaker than explicit duplicate annotations.

**`title_sim` pairs include noise.** A cosine threshold of 0.45 on short titles produces some false positives (e.g., sequential issues in the same sprint with similar vocabulary). Raising the threshold to 0.6 would increase precision but reduce coverage.

**Interpretation:** The Recall@K metrics below measure *related issue retrieval*, not strict duplicate detection. True duplicate detection would require scraping issue comments — feasible in a future iteration.

---

## 3. Results

### 3.1 Full Evaluation Table

| Repo | Model | R@1 | R@5 | R@10 | R@20 | MRR | p50 (ms) | p95 (ms) | Index |
|---|---|---|---|---|---|---|---|---|---|
| microsoft/vscode | **BGE-base** | **0.197** | **0.367** | **0.521** | 0.735 | **0.294** | 26.4 | 33.3 | 24.3 MB |
| microsoft/vscode | MiniLM-L6 | 0.165 | 0.353 | 0.506 | **0.740** | 0.267 | 23.2 | 28.6 | 13.5 MB |
| kubernetes/kubernetes | **BGE-base** | **0.232** | **0.410** | **0.474** | **0.541** | **0.316** | 35.3 | 41.8 | 50.1 MB |
| kubernetes/kubernetes | MiniLM-L6 | 0.187 | 0.356 | 0.435 | 0.507 | 0.266 | 22.4 | 25.6 | 27.1 MB |

**BGE wins on all metrics except vscode R@20 (essentially tied: 0.735 vs 0.740).**

### 3.2 Model Comparison: BGE vs MiniLM

| Metric | BGE advantage (vscode) | BGE advantage (kubernetes) |
|---|---|---|
| R@1 | +3.2pp | +4.5pp |
| R@5 | +1.4pp | +5.4pp |
| R@10 | +1.5pp | +3.9pp |
| R@20 | −0.5pp (MiniLM wins) | +3.4pp |
| MRR | +0.027 | +0.050 |
| Latency p50 | 3.2ms slower | 12.9ms slower |

**BGE is consistently more accurate at all cut-offs below @20.** At @20 the gap closes because MiniLM retrieves a broader set of weakly-similar candidates that happen to include the relevant item. For production use (top-5 deduplication suggestion), BGE is the clear choice.

The latency penalty (3–13ms) is modest — both models are in the 22–35ms range on CPU, well within acceptable UX bounds for a background enrichment task.

### 3.3 Latency Benchmark

| Model | vscode p50 | vscode p95 | kubernetes p50 | kubernetes p95 |
|---|---|---|---|---|
| BGE-base | 26.4ms | 33.3ms | 35.3ms | 41.8ms |
| MiniLM-L6 | 23.2ms | 28.6ms | 22.4ms | 25.6ms |

kubernetes is slower than vscode with BGE because the index has 15K vs 7K vectors. MiniLM is nearly constant-time (lower dimensional, FAISS search scales with dim × n). At 100K issues, BGE would be ~75ms, MiniLM ~45ms.

### 3.4 Index Size

| Repo | BGE index | MiniLM index |
|---|---|---|
| microsoft/vscode (7K issues) | 24.3 MB | 13.5 MB |
| kubernetes/kubernetes (15K issues) | 50.1 MB | 27.1 MB |
| Per-issue (BGE) | ~3.2 KB | ~1.8 KB |

At 100K issues: BGE ~320 MB, MiniLM ~180 MB. Both fit comfortably in memory.

---

## 4. Result Interpretation

### 4.1 What R@5 = 0.367–0.410 Means

In 36–41% of cases, the correct related issue appears in the top-5 results. For production duplicate detection:
- A triager reviewing the top-5 suggestions will find the true duplicate in roughly 1 in 3 issues
- At R@10 (0.474–0.521), the hit rate rises to ~50%
- At R@20, the numbers are 0.541–0.735 — showing the signal is present but buried

**Context on gold standard quality:** The `body_related` pairs (Fixes/Closes patterns) are the majority of gold pairs and represent weaker signal than explicit duplicates. On strict duplicate pairs only (4 high-confidence examples), all models retrieve the correct issue at rank 1 — but that sample is too small for reliable conclusions.

### 4.2 Why R@5 Isn't Higher

1. **Gold standard noise:** Many `body_related` pairs are "related issues" not true duplicates — the semantic embedding finds the right related issue but not necessarily the specific one referenced via `Fixes #X`.

2. **Body text quality:** Many issues have sparse bodies (one-liner bugs, minimal reproduction steps). The embedding has little signal beyond the title.

3. **Era mismatch (kubernetes):** 2014–2015 language patterns differ from current developer vocabulary in pretraining corpora. BGE was pretrained on modern technical text.

4. **Index contamination:** The query issue is in the index (excluded during retrieval), but conceptually similar issues from the same sprint cluster together, creating noise at the top of the ranking.

### 4.3 Actionable Improvement Paths

| Improvement | Expected ΔR@5 | Effort |
|---|---|---|
| Scrape comments → extract explicit `duplicate of #N` annotations | +10–15pp (cleaner gold) | Medium |
| Add cross-encoder re-ranking (top-20 → top-5) | +8–12pp | High |
| Fine-tune BGE on domain-specific pairs | +5–10pp | High |
| Use title-only index + body-only index separately, combine scores | +2–5pp | Low |
| Increase `body[:512]` to `body[:1024]` for longer issues | +1–3pp | Trivial |

---

## 5. Production Design

### 5.1 Recommended Configuration

```
Model: BGE-base-en-v1.5 (higher accuracy, 26–35ms on CPU)
K=10 for triage queue enrichment (R@10 ≈ 50%)
K=5 for UI "possible duplicates" panel (R@5 ≈ 39%)
Minimum similarity threshold: 0.6 (filter out low-confidence retrievals)
```

### 5.2 Integration with System 1

The duplicate detector complements the component classifier:

```
Incoming issue
├── System 1: TF-IDF classifier → component label + confidence
│   └── If confidence < threshold → DistilBERT secondary
└── System 2: BGE retrieval → top-5 similar past issues
    ├── If top-1 similarity > 0.85 → flag as likely duplicate
    └── Otherwise → surface as "related issues" for context
```

This gives triagers both a label and context (prior art) in a single pass.

### 5.3 Index Refresh Strategy

- Full rebuild: weekly (new issues accumulate; takes ~50s for 15K issues)
- Incremental: append new embeddings to index daily (`faiss.IndexFlatIP.add` is O(1) per vector)
- No retraining required — embedding model is frozen

---

## 6. Reproducibility

```bash
# Step 1: Extract gold pairs
python scripts/07_extract_duplicates.py
# Output: data/gold_duplicates.parquet (1,435 pairs)

# Step 2: Build indices + evaluate
python scripts/08_build_duplicate_index.py
# Output: data/models/dup_index_{repo}_{model}/
#         reports/duplicate_results.json
#         reports/charts/recall_at_k_curve.png
#         reports/charts/duplicate_score_dist_{repo}_{model}.png

# Optional: single model / repo
python scripts/08_build_duplicate_index.py --repos microsoft_vscode --models bge
```

**Index build times (GPU encoding, CPU search):**
- vscode BGE: ~28s encoding, cached on subsequent runs
- vscode MiniLM: ~10s
- kubernetes BGE: ~47s
- kubernetes MiniLM: ~14s

Charts generated:
- `reports/charts/recall_at_k_curve.png` — R@K per repo per model
- `reports/charts/duplicate_score_dist_{repo}_{model}.png` — similarity score distribution (true positive vs random negative)
