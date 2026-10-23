"""vscode duplicate retrieval evaluation (runbook Q5a; ADR-0066 item 1, D79 item 3).

PRE-REGISTERED DESIGN (2026-10-23; committed before any number is computed).
 * Target (ADR-0066 decision 1, verbatim): "vscode duplicate Recall@5 >= 80% stays the target"
   (spec section 7, unchanged).
 * Task (as scripts/qp0_ceiling_retrieval.py): query = the LATER issue of a duplicate pair
   (``source``), target = its duplicate canonical, from dup_task_pairs.parquet
   (D:/triageiq-data/splits/microsoft__vscode, derived_v2 issues). A query with several pairs has
   several canonicals; it is a hit at k when ANY indexed canonical ranks <= k.
 * Index per query: every vscode issue in derived_v2 issues.parquet (all splits) with created_at
   STRICTLY before the query's created_at (time-respecting), so the query itself is excluded.
 * Text: production recipe, title.strip() + ". " + clean_text(body)[0].strip()
   (scripts/qk_kaggle.build_texts). Sparse vocabulary / IDF / avgdl are fitted on train-split
   issues only (the ceiling script fitted TF-IDF on all issues; this is a deliberate change).
 * Ranking: candidates ordered by descending score; ties go to the MORE RECENT candidate (the
   qp2 prior-PR convention). Rank of a canonical = 1 + #{earlier issues scoring strictly higher}
   + #{tied issues created after the canonical}. Query rank = min over its indexed canonicals.
 * Primary metric: Recall@5 over ALL validation/test queries, a query whose canonicals are all
   unindexed (not in the table, or not created before the query) counts as a MISS. Reported
   also: indexed-only Recall@5 (denominator = queries with >= 1 indexed canonical), Recall@1/10/50,
   MRR (reciprocal query rank, 0 for unindexed), and the ceiling (share of queries with an indexed
   canonical = the maximum attainable recall). Pair-level variants (each pair its own sample)
   are reported alongside. CIs: Wilson 95% for rates; percentile bootstrap (2000 resamples, seed
   42, resampling queries) for MRR and all paired differences.
 * Candidates: B1 TF-IDF (sublinear tf, English stop words, min_df 2, cosine); B2 BM25 (k1 1.2,
   b 0.75, query term counts, idf = ln(1 + (N - df + 0.5) / (df + 0.5)), the qp2 definition);
   D1 BGE dense on the production recipe (Kaggle embeddings; NOT run in this task); H1 RRF(B2, D1)
   k = 60 (needs D1); H2 a learned / fine-tuned retriever if time allows (later).
 * Selection: on VALIDATION Recall@5 (all-queries denominator); ties -> MRR -> the simpler
   candidate (B1 < B2 < D1 < H1 < H2). The strongest baseline is the better of B1/B2 by the same
   rule. The test split is evaluated ONCE, only after every candidate exists and the selection is
   frozen; this script refuses test without --final-test.
 * Success: test Recall@5 of the selected candidate >= 0.80 as a POINT estimate, reported with its
   Wilson 95% CI and the paired bootstrap difference vs the strongest baseline (2000, seed 42);
   if the ceiling is below 0.80 the target is infeasible by construction and is reported as such.
 * This task (Q5a part 1): B1 and B2 on VALIDATION only; no test number is computed.
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError("implementation follows the design commit")


if __name__ == "__main__":
    sys.exit(main())
