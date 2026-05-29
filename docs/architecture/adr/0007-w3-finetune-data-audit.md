# ADR-0007 — W3 Fine-Tuning Data Audit

**Status:** Proposed (STOP condition triggered — awaiting GG review)  
**Date:** 2026-05-30  
**Decider:** Gaurav Gandhi

---

## Summary

T1 data audit reveals that `gold_duplicates.parquet` contains mostly **PR→issue fix pairs**
(not issue→issue duplicate pairs) and **TF-IDF title-similarity pairs** (auto-generated, circular).
Only 4/1435 pairs (0.3%) are high-confidence "duplicate of" pairs.

**W3 STOP condition triggered per spec:** "T1 reveals duplicate pair construction is unreliable
(e.g., auto-generated from text similarity rather than GitHub 'duplicate' labels) → STOP, surface.
Garbage labels → garbage model."

---

## Data Schema and Counts

```
repo               object     # microsoft_vscode | kubernetes_kubernetes
query_number        int64     # the "newer" issue (the duplicate)
original_number     int64     # the "older" issue (the original)
query_title        object
original_title     object
query_body         object
original_body      object
source             object     # body_ref | body_related | title_sim
confidence         object     # high | medium | low
```

| Repo | Rows | Unique issues |
|---|---|---|
| microsoft_vscode | 411 | 349 |
| kubernetes_kubernetes | 1024 | 1905 |
| **Total** | **1435** | **2236** |

All rows have no nulls. All query_numbers and original_numbers are present in the FAISS index
source data (issues parquets), confirming the eval is self-consistent.

---

## Pair Construction — Three Strategies, Three Reliability Levels

From `scripts/07_extract_duplicates.py`:

### Strategy 1 — Explicit body reference (source=`body_ref`, confidence=`high`)

**Pattern:** `r"[Dd]uplicate[sd]?(?: of)? #?(\d+)"` and similar "same as", "closing as dup" patterns.

| Repo | Count | % of total |
|---|---|---|
| microsoft_vscode | 1 | 0.2% |
| kubernetes_kubernetes | 3 | 0.3% |
| **Total** | **4** | **0.3%** |

Manual inspection of all 4 pairs:

| Pair | Q title | O title | Assessment |
|---|---|---|---|
| vs #4316→#909 | Typescript errors remain when running in watch mode | Errors remain even when fixed when running in watch mode | Genuine duplicate |
| k8s #5373→#5372 | Manually killing one POD container restarts all pods | Kubelet log filled with errors | Uncertain ("Maybe same as #5372?") |
| k8s #6651→#6261 | Pod status not reported promptly | Flake: mirror pod not found | Uncertain ("Probably not the same as #6261?") |
| k8s #7559→#7288 | Add a simple cache for objects stored in etcd. | Add a simple cache for objects stored in etcd | Genuine duplicate |

**2 confirmed genuine duplicates, 2 uncertain.** This is the only genuinely reliable pool, and it is far too small for any training (n=4).

---

### Strategy 2 — Weak body reference (source=`body_related`, confidence=`medium`)

**Pattern:** `r"[Cc]loses? #(\d+)"`, `r"[Ff]ixes? #(\d+)"`, `r"[Ss]ee(?: also)? #(\d+)"`

| Repo | Count | % of total |
|---|---|---|
| microsoft_vscode | 112 | 27.3% |
| kubernetes_kubernetes | 898 | 87.7% |
| **Total** | **1010** | **70.4%** |

**Critical finding:** These are predominantly **Pull Request → Issue pairs**, not Issue → Issue pairs.

- 47.3% of vscode body_related query bodies have "Closes/Fixes #N" as first line — standard PR merge description
- 12.5% explicitly contain "This PR" or "This pull request" language
- Sample k8s pairs confirmed: "Add a proxy test" (Q) → "Write an e2e test for the node proxy" (O); "Clarify comments describing GuaranteedUpdate()" (Q) → "Decide if current ETCD Update semantics is correct" (O) — these are PRs implementing the referenced issues, not duplicates

The `issues_{repo}.parquet` includes PRs (GitHub's API returns issues and PRs via the same endpoint). These PRs have bug/feature labels in the issues data and were not filtered out during preprocessing.

**Training on these pairs would teach the model "PR body is semantically related to the issue it resolves" — an entirely different signal from "Issue A is a duplicate of Issue B."**

---

### Strategy 3 — TF-IDF title similarity (source=`title_sim`, confidence=`medium/low`)

**Method:** TF-IDF bigram cosine similarity ≥ 0.45 on `title + body[:200]`. Top-300 pairs per repo.

| Repo | Count | % of total |
|---|---|---|
| microsoft_vscode | 298 | 72.5% of vscode |
| kubernetes_kubernetes | 123 | 12.0% of k8s |
| **Total** | **421** | **29.3%** |

Manual spot-check of 5 pairs:
- "Added UndeltaStore." / "Added UndeltaStore." → identical title (likely genuine dup or near-duplicate)
- "bu found" / "Timestamp" → completely unrelated (TF-IDF false positive)
- "Provide task oriented documentation around Clusters" / "...around Replication Controllers" → related docs but DIFFERENT topics
- "[Error] unhandlederror: popular — onD" / "...popular — texD" → same error, different contexts, possibly genuine
- "Correct kubectl delete's wrong synopsis" / "Kubectl delete command's wrong synopsis" → plausibly genuine duplicate

**Mixed quality: some genuine duplicates, some false positives (unrelated issues with similar surface tokens), some "related but distinct" cases.** Not a reliable training signal.

**Circular training risk:** Training a CE on pairs that were selected by TF-IDF would teach the model to predict TF-IDF similarity — which BGE already does better. No information gain.

---

## Reliability Summary

| Source | Count | Actual pair type | Training signal for dup detection |
|---|---|---|---|
| body_ref | 4 | Issue→issue duplicate | ✓ Reliable (n too small) |
| body_related | 1010 | ~75% PR→issue fix | ✗ Wrong task |
| title_sim | 421 | ~30-40% genuine dup | ✗ Circular (TF-IDF) |

**The gold_duplicates.parquet is not a "duplicate pairs" dataset. It is a "related issue pairs" dataset dominated by PR→issue fix pairs.** The naming is misleading.

---

## Implication for the W1.3 evaluation

The W1.3 baseline (vscode R@5=0.367, k8s R@5=0.410) was measured against ALL 1435 pairs, including
PR→issue and title_sim pairs. This means:

1. The metric is not strictly "find duplicate issues" — it's closer to "find related/referenced issues"
2. The strong k8s performance of bge-v2-m3 (+6pp in W1.3 screening) may have been capturing the PR→issue
   relatedness signal (which a search-relevance model is well-suited to detect), not duplicate detection
3. The Quora/STS models' catastrophic k8s failure may be because Quora questions look nothing like
   k8s PRs, even at the "relatedness" level

This retroactively partially explains the W1.3 results: the task was not pure duplicate detection —
it was mixed-quality relatedness detection where search-relevance CEs had some advantage.

---

## Connected Component Analysis (for T1.e — split strategy)

Components (connected subgraphs of duplicate/related pairs):

| Size | Count | Issues involved |
|---|---|---|
| 2 (simple pair) | 914 | 1828 |
| 3 | 89 | 267 |
| 4 | 9 | 36 |
| 5 | 5 | 25 |
| 6 | 1 | 6 |
| 7 | 1 | 7 |
| 8 | 1 | 8 |
| 16 | 1 | 16 |
| 18 | 1 | 18 |
| 25 | 1 | 25 |

109 of 1023 components have chains (>2 nodes). Largest chain: 25 nodes.

**Component-based splitting would be required** if proceeding to T2 — naive random splitting would
put the same issue on both sides of train/val, inflating val metrics.

Even with correct splitting, the usable training data is very sparse: at most ~500-800 genuine or
plausibly-genuine pairs (filtering out confirmed PR→issue cases), split across two repos.

---

## STOP Condition Assessment

Per W3 spec: *"T1 reveals duplicate pair construction is unreliable (e.g., auto-generated from text
similarity rather than GitHub 'duplicate' labels) → STOP, surface."*

This condition is met:
- 29.3% of pairs are auto-generated from TF-IDF text similarity — exactly the example in the spec
- 70.4% are mined from "Closes/Fixes" patterns — not from GitHub "duplicate" labels

**4 out of 1435 pairs (0.3%) come from explicit "duplicate of" GitHub language.**

Training on this dataset would NOT produce a duplicate detector. It would produce a PR-relevance
predictor (from body_related) and a TF-IDF similarity predictor (from title_sim) — neither of which
is a useful improvement over the existing BGE embedding retrieval.

---

## Options for GG review

### Option A — Accept STOP, declare W3 failure on data quality

**Outcome:** W3 Rejected. Document that the training data prerequisite was not met.
ADR-0007 status: Rejected.
ADR-0006 footnote: "W3 fine-tuning attempted; blocked at T1 — insufficient genuine duplicate labels.
Bottleneck confirmed: the reranker stage is not the right lever without higher-quality duplicate
labels."

No training runs needed. No compute spent.

### Option B — Acquire genuine duplicate labels (unblocks W3 cleanly)

The GitHub API reports `pull_request.merged_at` to distinguish PRs from issues, and issues marked
as "duplicate" carry a standard label `duplicate`. Re-scraping both repos with:
- `GET /repos/{owner}/{repo}/issues?labels=duplicate&state=closed` (for issues with duplicate label)
- Parsing the body for "Duplicate of #N" referents

vscode has a dedicated `*duplicate` label with known high prevalence. k8s uses `triage/duplicate`.
A targeted re-scrape could yield 200-500 genuine duplicate pairs per repo.

This is out of original W3 scope but would make fine-tuning tractable.

### Option C — Proceed with filtered subset + acknowledged noise

Use only:
- body_ref pairs (4)
- body_related pairs where query_body does NOT start with Fixes/Closes/Resolves (estimated ~500 pairs)
- title_sim pairs at confidence=medium with sim ≥ 0.6 only

Total: ~500-600 pairs. Acknowledge that quality is mixed. Proceed to T2-T5 with this caveat.

**Risk:** Model trained on mixed data evaluated on the same mixed data may show apparent improvement
that doesn't reflect genuine duplicate detection capability. Eval circularity.

---

## Recommendation

Option A (STOP). The W3 spec explicitly defined this STOP condition, and the data clearly triggers it.
Option B is the clean path to a meaningful W3 attempt but requires new data collection (a separate
workstream). Option C risks producing a meaningless positive result.

Awaiting GG decision before any T2 work begins.
