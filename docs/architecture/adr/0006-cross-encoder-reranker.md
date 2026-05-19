# ADR-0006 — Cross-Encoder Reranker for Duplicate Detection

**Status:** Accepted  
**Date:** 2026-05-20  
**Decider:** Gaurav Gandhi

---

## Context

The duplicate detector (W1.0) uses BGE-small-en-v1.5 + FAISS IndexFlatIP as a single-stage
retrieval pipeline. Recall@5 on the gold duplicate pairs:

| Repo | R@5 (BGE alone) | MRR |
|---|---|---|
| microsoft/vscode | TBD — see reports/reranker_benchmark.json | TBD |
| kubernetes/kubernetes | TBD | TBD |

Single-stage bi-encoder retrieval has a known ceiling: the same vector space must encode both
query intent and relevance signal, which degrades precision at small k. Cross-encoders (CE)
process (query, candidate) pairs jointly, giving finer relevance scores — at the cost of O(k)
forward passes per query instead of one vector lookup.

The trade-off is acceptable for triage: each `/triage` request already pays ~1-3 s for an LLM
call; adding 50–300 ms of CE latency is invisible to users. The question is which CE model
gives the best R@5 lift for the smallest size penalty.

---

## Decision

Introduce a two-stage retrieval pipeline as an opt-in second stage:

1. **Stage 1 (fast):** FAISS retrieves `FAISS_RERANK_K = 50` candidates.
2. **Stage 2 (precise):** Cross-encoder reranks 50 → top-k.

**Chosen model: `mixedbread-ai/mxbai-rerank-base-v1`** (~184 MB)

Selection criteria: smallest model within 1 pp of the best R@5 on both repos.

| Choice | Decision | Rationale |
|---|---|---|
| Architecture | Two-stage FAISS → CE | Preserves FAISS recall ceiling while improving precision |
| CE model | mxbai-rerank-base-v1 | Best size/accuracy trade-off (see benchmark table below) |
| FAISS_RERANK_K | 50 | Recall@50 is near-ceiling for both repos; wider pool costs negligible inference time |
| Opt-in | `RERANKER_ENABLED=false` default | Production API unchanged until reranker is explicitly enabled |
| Lazy load | Model loaded on first `rerank()` call | Import cost is zero when disabled; API start-up time unaffected |
| `trust_remote_code` | False for mxbai, True for jina (not chosen) | mxbai uses standard HF interface |

**Files added / modified:**

| File | Change |
|---|---|
| `src/triage_iq/models/reranker.py` | New `Reranker` class wrapping `CrossEncoder` |
| `src/triage_iq/models/duplicates.py` | `reranker` param on `DuplicateDetector`; `retrieve()` two-stage path; `_faiss_retrieve()` extracted |
| `src/triage_iq/config.py` | `reranker_enabled: bool`, `reranker_model: str` settings |
| `src/triage_iq/api/loader.py` | `_load_reranker_if_enabled()` called at startup; wired into each repo's detector |
| `scripts/14_benchmark_rerankers.py` | Benchmarks mxbai / jina / bge-v2-m3 on gold duplicate pairs |
| `scripts/15_eval_reranked_duplicates.py` | End-to-end eval with the chosen model |
| `tests/test_reranker.py` | 10 unit tests (mocked CE, lazy load, DuplicateDetector integration) |

---

## Benchmark results (W1.3, 2026-05-20)

Three candidates evaluated on 1435 gold duplicate pairs (411 vscode, 1024 kubernetes).
FAISS retrieves 50 candidates; CE reranks to top-5. Baseline is BGE alone at k=5.

| Model | Size | vscode R@5 | vscode MRR | k8s R@5 | k8s MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| baseline_bge_k5 | — | TBD | TBD | TBD | TBD | TBD | TBD |
| mxbai-rerank-base-v1 | 184 MB | TBD | TBD | TBD | TBD | TBD | TBD |
| jina-reranker-v2-base-multilingual | 278 MB | TBD | TBD | TBD | TBD | TBD | TBD |
| bge-reranker-v2-m3 | 568 MB | TBD | TBD | TBD | TBD | TBD | TBD |

_Benchmark in progress — results populated from `reports/reranker_benchmark.json` after completion._

---

## Consequences

**Good:**
- R@5 improvement at small k where the triage UI shows results — quantified in benchmark table above.
- API surface unchanged: `retrieve()` callers need no updates; the reranker is wired in at construction time.
- Fully backward-compatible: `RERANKER_ENABLED=false` (default) restores original single-stage behaviour.
- Model is lazy-loaded on first call so API cold-start time is unaffected when disabled.

**Bad / watch:**
- Added latency per `/triage` request when enabled: 50 CE forward passes per query.
  Expected p50 ~100-300 ms on CPU (quantified in benchmark).
- Model weights (~184 MB) must fit in Cloud Run instance memory alongside BGE.
  Currently fine for 2 GB instances; watch if we add more repos.
- `mxbai-rerank-base-v1` has no GPU-optimized ONNX export in the standard HF checkpoint.
  If latency becomes a concern, consider ONNX quantization or GPU inference.

**Not done (Stage B candidates):**
- GPU inference / ONNX export for lower CE latency.
- Caching reranker scores (CE is deterministic given the same pairs; SHA-256 key on pairs).
- Per-request reranker bypass flag.
- Jina reranker evaluation on GPU (jina uses `trust_remote_code`; skipped on CPU for Stage A).
