# ADR-0006 — Cross-Encoder Reranker for Similar-Issue Retrieval

**Status:** Rejected (original W1.3 verdict) — see *Reinterpretation under ADR-0008* below  
**Date:** 2026-05-30  
**Decider:** Gaurav Gandhi

---

## Context

The similar-issue retriever (W1.0) uses BAAI/bge-base-en-v1.5 + FAISS IndexFlatIP as a single-stage
retrieval pipeline. *(Note: at the time of W1.3 screening this module was named `DuplicateDetector`;
ADR-0008 corrects the task framing to "related-issue retrieval" — see Reinterpretation section below.)*

Recall@5 on the full gold set (canonical baseline from `related_issue_results.json`):

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
signal they learn is "document B answers query A," not "document B is related to document A."
For vscode issues (diverse types: bugs, feature requests, questions, docs), high relevance ≠ relatedness
— so the CE actively re-orders the FAISS top-5 away from true related issues. For k8s issues (more
uniform technical infrastructure vocabulary), relevance ≈ relatedness by coincidence, explaining the
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
0.620), suggesting it can find related issues in a wider candidate pool, but cannot rank them into
the top-5 reliably.

### Cross-slate finding: why bge-v2-m3 was the outlier

bge-reranker-base (same family, smaller) gives k8s −5pp while bge-v2-m3 gives k8s +6pp. This 11pp
swing within the same model family is explained by training scale and data diversity: bge-v2-m3 uses
a much larger and more diverse multilingual retrieval training set that incidentally covers
technical/infrastructure text. The base model lacks this coverage.

---

## Decision

**Rejected.** No cross-encoder from either screening slate improves both corpora.

Decision tree outcome: **CASE D** — no candidate beats baseline on either repo by ≥3pp in the
final (determinative) screening slate. Drop W1.3. Close PR #1 without merging.

The implementation (`src/triage_iq/models/reranker.py`, the `SimilarIssueRetriever.reranker` hook, and
`loader.py` plumbing) is correct and clean. The failure is not an integration bug. The hypothesis
was simply wrong: pre-trained cross-encoders without in-domain fine-tuning do not generalise to
GitHub issue related-issue retrieval.

---

## Root cause

Two distinct failure modes, both of which would need to be solved for W1.3 to succeed:

**Failure 1 — Corpus mismatch (vscode):** vscode has diverse issue types (bugs, feature requests,
clarification questions, documentation issues). High semantic similarity between two issues ≠
relatedness. A CE trained on any relevance task will score "topically similar" issues higher than
truly related ones.

**Failure 2 — Domain mismatch (k8s, all Quora/STS models):** Quora and STS training data contains
informal natural language. k8s issues use technical domain vocabulary (CRDs, controllers, kubelet,
etcd, YAML manifests). CEs trained on informal text cannot rank technical related issues.

---

## Consequences

**Positive (negative result is still a result):**
- The `Reranker` class, `SimilarIssueRetriever` two-stage hook, and `RERANKER_ENABLED` opt-in
  infrastructure remain in the codebase for future use. The design is correct even though no model
  currently benefits from it.
- 69/69 tests pass throughout. No regression introduced.
- Clear empirical evidence that CE reranking on this dataset requires domain-specific fine-tuning.

**What would be needed for W1.3 to succeed (future work):**
- Fine-tune a CE on the project's own `gold_related.parquet` pairs using hard negative mining from
  the FAISS top-50. This is the standard approach for production related-issue retrieval.
- Approximate upper bound on achievable improvement: the `stsb` model's R@10 gain (+7pp vscode)
  suggests the CE CAN find more related issues — it just needs domain-tuning to rank them into top-5.

**What is NOT being done (original verdict):**
- bge-v2-m3 repo-gated for k8s only. The k8s +6pp is built on a training coincidence (BGE v2-m3's
  large multilingual training happens to overlap k8s vocabulary). Shipping on this basis is fragile.

---

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| bge-v2-m3 repo-gated for k8s only (CASE C) | k8s improvement may be a training coincidence, not principled. Asymmetric architecture adds complexity. Phase 2 tests this hypothesis. |
| Third CE slate (larger models) | Time-box: two slates were the agreed limit. The failure pattern is systematic. |
| Reduce final-k to top-3 | Would hurt R@5 further; doesn't address ranking quality. |
| Increase FAISS k beyond 50 | CE cost increases; bottleneck is ranking quality, not candidate pool. |

---

## Screening data

Raw results in git history on `feat/w1.3-cross-encoder-reranker`:
- `reports/bge_v2m3_benchmark.json` — Slate 1 bge-v2-m3 detailed results
- `reports/dup_trained_reranker_screening.json` — Slate 2 complete screening (all 4 candidates)

---

## Reinterpretation under ADR-0008

**Date:** 2026-05-30  
**Trigger:** ADR-0008 (task reframing: "duplicate detection" → "related-issue retrieval")

### What changed

ADR-0008 established that the gold dataset (`data/gold_related.parquet`, formerly
`gold_duplicates.parquet`) contains primarily PR→issue fix references and text-similar pairs — not
strict GitHub-marked duplicate issue pairs. The task being measured throughout W1.3 was therefore
**related-issue retrieval** (surface historically related issues given a new issue), not duplicate
detection.

This reframing does not invalidate the screening data. The same queries, the same model outputs,
the same FAISS index — only the task label changes. All R@5 numbers above are valid measurements
of how well each model retrieves related issues.

### Re-reading the two key findings

**vscode: all cross-encoder candidates degrade (−4pp to −16pp)**

Unchanged by reframing. vscode has high issue-type diversity (bugs, features, questions, docs).
Under any task framing — duplicate detection *or* related-issue retrieval — a general-purpose CE
scores "topically similar" issues higher than issues that are actually linked by PR references or
editorial curation. BGE-base FAISS top-5 already selects for embedding similarity, which is a
reasonable proxy for relatedness. The CE reranks away from this signal. BGE-base alone is best for
vscode.

**k8s: bge-v2-m3 +6pp R@5 (0.430 → 0.490, n=100, seed=42)**

Under reframing, this becomes a candidate real win: bge-v2-m3 improves related-issue retrieval for
a technical infrastructure corpus. The original CASE D verdict assumed both corpora must pass; under
CASE C reasoning (repo-gated), k8s +6pp at n=100 is worth robustness testing before rejecting.

The original "training coincidence" concern applies, but is now a hypothesis to test rather than a
rejection reason. A principled explanation exists: k8s issues are predominantly technical retrieval
queries (YAML, controller logs, error messages) — closer to the MS MARCO retrieval training
distribution than informal Q&A. If the gain holds at n=300 with a tight CI, shipping repo-gated for
k8s is defensible.

### Phase 2 evaluation plan

The following gates decide the final ADR-0006 verdict:

| Gate | Test | Stop condition |
|---|---|---|
| T2 | n=300 k8s robustness, 1000-resample bootstrap 95% CI on delta | CI crosses zero → rejection stands |
| T3 | CPU-only rerank latency (bge-v2-m3, k8s top-50, p95) | p95 > 2.5s even at top-25 → infeasible |
| T4 | Repo-gated implementation (k8s only, vscode=None) | Only if T2+T3 pass |
| T5 | Cohere judge k8s-subset similar_issues_relevance delta | No movement → don't ship |

Final verdict updates will be appended to this ADR when Phase 2 concludes.
