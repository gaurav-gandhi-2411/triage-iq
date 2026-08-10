# ADR-0008 — Task Reframing: "Duplicate Detection" → "Similar Issue Retrieval"

**Status:** Accepted  
**Date:** 2026-05-30  
**Decider:** Gaurav Gandhi

---

## Context

A T1 data audit (ADR-0007) of `data/gold_related.parquet` (formerly `gold_duplicates.parquet`)
revealed that the training/evaluation dataset contains:

| Source | Count | % | Actual pair type |
|---|---|---|---|
| `body_ref` — "duplicate of #N" language | 4 | 0.3% | Genuine duplicate issue pairs |
| `body_related` — "Closes/Fixes/See #N" patterns | 1010 | 70.4% | PR→issue fix pairs + issue cross-references |
| `title_sim` — TF-IDF cosine ≥ 0.45 | 421 | 29.3% | Text-similar pairs (mixed quality) |

The user-facing Pydantic schema (`SimilarIssue`, `TriagePlan.similar_issues`) and the judge rubric
dimension (`similar_issues_relevance`) were already correctly named. The internal codebase had
drifted: module name (`duplicates.py`), class name (`DuplicateDetector`), eval scripts
(`08_build_duplicate_index.py`), data file (`gold_duplicates.parquet`), and README all used
"duplicate detection" as the task name.

The user-facing feature is: *"Given an issue, surface the most semantically related historical
issues from the same repository."* This is what the data captures and what the eval measures.
The data was never exclusively "GitHub-marked duplicate issue pairs."

---

## Decision

Rename all internal code references from "duplicate detection" to "similar issue retrieval."
No behavior change, no metric change, no API change.

**Rename map applied:**

| From | To | Notes |
|---|---|---|
| `src/triage_iq/models/duplicates.py` | `similar_issues.py` | Module renamed |
| `class DuplicateDetector` | `class SimilarIssueRetriever` | Class renamed |
| `scripts/07_extract_duplicates.py` | `07_extract_related_pairs.py` | Script renamed |
| `scripts/08_build_duplicate_index.py` | `08_build_similar_issue_index.py` | Script renamed |
| `data/gold_duplicates.parquet` | `data/gold_related.parquet` | Data file renamed |
| `reports/duplicate_results.json` | `reports/related_issue_results.json` | Report renamed |

**Left unchanged (intentionally):**
- `SimilarIssue` Pydantic class — always correct, no rename needed
- `similar_issues_relevance` judge dimension — always correct
- `duplicate_count` API response field — user-facing, breaking change to rename; deferred
- `_has_duplicate_label()` function — checks GitHub's own `duplicate` label by name; semantically correct
- `dup_index_microsoft_vscode_bge` / `dup_index_kubernetes_kubernetes_bge` — GCS artifact names; renaming requires a coordinated GCS → local rename; deferred as a follow-up. **Update:** completed in issue #3 — renamed to `similar_issue_index_*`.

**GCS artifact deferral:** The FAISS index directories on GCS and in `data/models/` retain the `dup_index_*` naming. `loader.py` contains a comment marking the path strings as deferred-rename. Once GCS artifacts are renamed (a separate operational step), update `deploy.yml`, `Dockerfile.prod`, and `loader.py` in a single follow-up commit.

---

## Consequences

**Good:**
- Internal code now matches what the system actually does and what the user sees
- ADR-0007 negative result (W3 blocked at T1) is reframed: the data is appropriate for the task,
  only the naming was misleading
- W3 fine-tuning path (deferred to Phase 3) is now coherent: fine-tuning on "related issue pairs"
  to improve "similar issue retrieval" is a sensible task; fine-tuning for "duplicate detection"
  on non-duplicate data was the conceptual mismatch

**W1.3 reinterpretation (see Phase 2):**
W1.3 screening measured cross-encoder rerankers on related-issue retrieval, not strict duplicate
detection. ADR-0006 is updated with a footnote referencing ADR-0008. Phase 2 reinterprets the
W1.3 results under the corrected task framing and decides whether bge-reranker-v2-m3 ships
repo-gated for kubernetes.

**Not changed:**
- 69/69 tests pass throughout — no regression
- `duplicate_count` API field rename is a follow-up item (tracked here, not blocking)
- GCS artifact rename is a follow-up item (tracked here)
