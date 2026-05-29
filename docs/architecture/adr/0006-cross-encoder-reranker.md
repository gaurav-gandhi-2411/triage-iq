# ADR-0006 — Cross-Encoder Reranker for Duplicate Detection

**Status:** Rejected  
**Date:** 2026-05-30  
**Decider:** Gaurav Gandhi

---

## Context

The duplicate detector (W1.0) uses BAAI/bge-base-en-v1.5 + FAISS IndexFlatIP as a single-stage
retrieval pipeline. Recall@5 on the full gold duplicate set (canonical baseline from `duplicate_results.json`):

| Repo | R@5 (BGE alone) | MRR | n pairs |
|---|---|---|---|
| microsoft/vscode | 0.367 | 0.294 | 411 |
| kubernetes/kubernetes | 0.410 | 0.316 | 1024 |

Single-stage bi-encoder retrieval has a known ceiling: the same embedding space must capture both
broad recall and fine-grained relevance. Cross-encoders (CE) process (query, candidate) pairs
jointly, which should in theory give better discrimination — at the cost of O(k) forward passes per
query instead of one vector lookup.

The hypothesis was: retrieve FAISS top-50, rerank to top-5 with a CE, and gain R@5 lift with
acceptable latency addition.

---

## Investigation arc

### Slate 1 — General-purpose search-relevance cross-encoders (2026-05-30)

All three candidates screened on GPU with n=100 random queries per repo (seed=42).
Baseline anchor for the same 100 queries: vscode R@5=0.470, k8s R@5=0.430.

| Candidate | Size | License | vscode R@5 | Δ | k8s R@5 | Δ | Result |
|---|---|---|---|---|---|---|---|
| Baseline (BGE FAISS k=20) | — | — | 0.470 | — | 0.430 | — | — |
| mixedbread-ai/mxbai-rerank-base-v1 | 184 MB | Apache-2.0 | 0.310 | −16pp | 0.430 | ±0pp | **FAIL** |
| jinaai/jina-reranker-v2-base-multilingual | 278 MB | CC-BY-NC-4.0 | — | — | — | — | **ELIMINATED** |
| BAAI/bge-reranker-v2-m3 | 568 MB | Apache-2.0 | 0.390 | −8pp | 0.490 | +6pp | **PARTIAL** |

jina eliminated: (1) CC-BY-NC-4.0 license prohibits production use; (2) `Got unsupported ScalarType BFloat16`
hardware error on this machine.

mxbai result: actively hurt vscode (−16pp). This alone triggered the STOP condition.

bge-v2-m3 result: split — improved k8s by +6pp, hurt vscode by −8pp. Not deployable as a global
reranker (hurts one corpus). Would qualify for CASE C (repo-gated: k8s only) but led to a deeper question.

**Diagnosis after Slate 1:** These models are trained on MS MARCO / web-search relevance tasks. The
signal they learn is "document B answers query A," not "document B is a duplicate of document A."
For vscode issues (diverse types: bugs, feature requests, questions, docs), high relevance ≠ duplication
— so the CE actively re-orders the FAISS top-5 away from true duplicates. For k8s issues (more
uniform technical infrastructure vocabulary), relevance ≈ duplication by coincidence, explaining the
bge-v2-m3 k8s improvement.

### Slate 2 — Duplicate/paraphrase/STS-trained cross-encoders (2026-05-30)

Pivoted to models specifically trained for semantic equivalence detection rather than search relevance.
Same evaluation protocol: n=100 queries per repo, seed=42.

| Candidate | Size | Training | vscode R@5 | Δ | k8s R@5 | Δ | Result |
|---|---|---|---|---|---|---|---|
| Baseline (BGE FAISS k=20) | — | — | 0.470 | — | 0.430 | — | — |
| cross-encoder/quora-distilroberta-base | 82 MB | Quora Dup. Questions | 0.430 | −4pp | 0.290 | **−14pp** | **FAIL** |
| cross-encoder/quora-roberta-base | 125 MB | Quora Dup. Questions | 0.380 | −9pp | 0.330 | **−10pp** | **FAIL** |
| cross-encoder/stsb-distilroberta-base | 82 MB | STS-Benchmark | 0.410 | −6pp | 0.260 | **−17pp** | **FAIL** |
| BAAI/bge-reranker-base | 278 MB | Mixed retrieval | 0.420 | −5pp | 0.380 | −5pp | **FAIL** |

All four candidates FAIL k8s even more severely than the search-relevance models from Slate 1.

The Quora Duplicate Questions dataset contains informal natural language Q&A pairs. GitHub issues —
especially k8s infrastructure tickets about YAML configurations, controller behavior, and network
policies — bear no textual resemblance to Quora questions. The models learned "are these two informal
questions asking the same thing?" and apply that signal to technical infrastructure discussions,
producing nonsense rankings.

Notable exception in Slate 2: `stsb-distilroberta-base` improved vscode R@10 (+7pp, from 0.550 to
0.620), suggesting it can find duplicates in a wider candidate pool, but cannot rank them into the
top-5 reliably. This could be explored with a larger final-k, but is out of scope for W1.3.

### Cross-slate finding: why bge-v2-m3 was the outlier

bge-reranker-base (same family, smaller) gives k8s −5pp while bge-v2-m3 gives k8s +6pp. This 11pp
swing within the same model family is explained by training scale and data diversity: bge-v2-m3 uses
a much larger and more diverse multilingual retrieval training set that incidentally covers
technical/infrastructure text. The base model lacks this coverage.

This makes the k8s improvement from bge-v2-m3 fragile — it is not a principled signal, just a
coincidental training distribution overlap with k8s infrastructure vocabulary. Deploying bge-v2-m3
repo-gated for k8s would be building on this coincidence rather than a genuine quality signal.

---

## Decision

**Rejected.** No cross-encoder from either screening slate improves both corpora.

Decision tree outcome: **CASE D** — no candidate beats baseline on either repo by ≥3pp in the
final (determinative) screening slate. Drop W1.3. Close PR #1 without merging.

The implementation (`src/triage_iq/models/reranker.py`, the `DuplicateDetector.reranker` hook, and
`loader.py` plumbing) is correct and clean. The failure is not an integration bug. The hypothesis
was simply wrong: pre-trained cross-encoders without in-domain fine-tuning do not generalise to
GitHub issue duplicate detection.

---

## Root cause

Two distinct failure modes, both of which would need to be solved for W1.3 to succeed:

**Failure 1 — Corpus mismatch (vscode):** vscode has diverse issue types (bugs, feature requests,
clarification questions, documentation issues). High semantic similarity between two issues ≠
duplication. A CE trained on any relevance task will score "related" issues higher than true
duplicates.

**Failure 2 — Domain mismatch (k8s, all Quora/STS models):** Quora and STS training data contains
informal natural language. k8s issues use technical domain vocabulary (CRDs, controllers, kubelet,
etcd, YAML manifests). CEs trained on informal text cannot rank technical duplicates.

---

## Consequences

**Positive (negative result is still a result):**
- The `Reranker` class, `DuplicateDetector.retrieve()` two-stage hook, and `RERANKER_ENABLED` opt-in
  infrastructure remain in the codebase for future use. The design is correct even though no model
  currently benefits from it.
- 79/79 tests pass throughout. No regression introduced.
- Clear empirical evidence that CE reranking on this dataset requires domain-specific fine-tuning.

**What would be needed for W1.3 to succeed (future work, not current scope):**
- Fine-tune a CE on the project's own `gold_duplicates.parquet` pairs (1435 positive pairs) using
  hard negative mining from the FAISS top-50. This is the standard approach for production duplicate
  detection (see: StackExchange duplicate detection papers, GitHub's own internal dedup systems).
- Approximate upper bound on achievable improvement: the `stsb` model's R@10 gain (+7pp vscode)
  suggests the CE CAN find more duplicates — it just needs domain-tuning to rank them into top-5.

**What is NOT being done:**
- bge-v2-m3 repo-gated for k8s only. While the k8s +6pp is real, it is built on a training-set
  coincidence (BGE v2-m3's large multilingual training happens to overlap k8s vocabulary). Shipping
  a feature on this basis is fragile. One BGE model update could flip the result.

---

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| bge-v2-m3 repo-gated for k8s only (CASE C) | k8s improvement is a training coincidence, not principled. Asymmetric architecture adds complexity without reliable benefit. |
| Third CE slate (larger models, different providers) | Time-box: two slates were the agreed limit. The failure pattern is systematic, not model-specific. A third slate would not fix the root cause. |
| Reduce final-k to top-3 | Would hurt R@5 further; doesn't address the ranking quality problem. |
| Increase FAISS k beyond 50 | Increases CE cost; the bottleneck is ranking quality, not candidate pool size. |
| Use CE as a binary classifier (duplicate/not) with threshold | Would require different output layer, fine-tuning, threshold calibration — effectively the same as full fine-tuning path. |

---

## Screening data

All raw results saved to:
- `reports/bge_v2m3_benchmark.json` — Slate 1 bge-v2-m3 detailed results
- `reports/dup_trained_reranker_screening.json` — Slate 2 complete screening (all 4 candidates)
- `reports/bge_v2m3_benchmark.log` — Slate 1 GPU screening log
- `reports/dup_trained_screening.log` — Slate 2 GPU screening log
