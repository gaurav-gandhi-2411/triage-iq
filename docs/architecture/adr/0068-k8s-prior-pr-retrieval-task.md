# ADR-0068 — k8s prior-PR retrieval task (replaces the same-PR fix-linked headline of ADR-0066)

Status: Accepted (frozen 2026-10-20, before any model is scored on it)
Date: 2026-10-20
Decider: Gaurav Gandhi (decision of 2026-10-20, option A, D83)

## Context

ADR-0066 named same-PR fix-linked retrieval as the k8s retrieval headline. The ceiling analysis showed it cannot
support a scored result: 22 groups of issues fixed by the same merged PR (28 under the union of every definition
tried), and 2 answerable queries in the frozen test split. GG chose a new, time-respecting task (option A) that also
feeds Phase 3b ("likely code area / recent changers" as grounded evidence). Fix-link precision for k8s was audited
independently at 118/120 = 98.3% (Wilson 94.1-99.5).

## Decision: the task

**Query.** A kubernetes issue q created at t_q whose audited fix PR F(q) exists (derived `fix_links`, method
`closed_event_closer_pr`, same repository, merged; if several, the earliest merged). Only issues of the evaluation
splits are scored; the splits are the frozen issue splits (train <= 2025-06-30, validation 2025-07-01..2025-12-31,
test >= 2026-01-01).

**Index.** INDEX(q) = merged PRs with base branch `master`, created on or after 2023-01-01 (the scrape window; PRs
older than that are unavailable, a stated limitation), and merged STRICTLY before t_q. Cherry-picks into `release-*`
branches (1,615 of 12,659 scraped merged PRs) are excluded. F(q) is never in the index (it merges after t_q for every
one of the 1,892 fix-linked queries; asserted in the pipeline). The issue's own fix PR is never a target.

**Positives.** A prior PR p in INDEX(q) is a positive if files(p) intersects INFORMATIVE_FILES(F(q)), where
INFORMATIVE_FILES(F) = files(F) minus EXCLUDED files. Excluded files, declared up front:
(a) patterns: `go.mod`, `go.sum`, `go.work`, `go.work.sum`, `OWNERS`, `OWNERS_ALIASES`, `generated.proto`, any
`zz_generated*`, `*.pb.go`, anything under `vendor/` or `staging/**/vendor/`, `api/openapi-spec/`, top-level
`CHANGELOG/`; (b) frequency: any file changed by MORE than 1.0% of the merged `master` PRs merged on or before
2025-06-30 (train period only). The frequency list (87 files, 6,779 train-period PRs, sha256
`a05445d6129f358e9a135ab1b5df724f828672afd591c0d70469686da51743ee`) is committed in
`reports/qp2_prior_pr_common_files_kubernetes__kubernetes_2026-10.json` and is FROZEN. PRs with more than 300 files
use their first 300 (54 PRs, flagged). A query is **answerable** iff INFORMATIVE_FILES(F(q)) is non-empty and it has
at least one positive.

**Document and query features allowed.** Document: PR title, body, changed file paths, merge time, author
association. Query: issue title and body only. Gold labels (the `sig/*` label) are not query features; a predicted
label from a frozen model may be tested as a separate, named variant.

**Metrics** (answerable queries): primary hit rate at 10 (Recall@10 = share of queries with at least one positive in
the top 10), MRR (first positive), nDCG@10 (binary relevance); also Recall@5 and precision@10. A stricter secondary
slice: the **non-trivial** positives, i.e. positives whose overlap with F(q) includes at least one informative file
outside the 100 most frequently changed informative files of the train period (ties by path; the list is committed
with its hash when the first scoring script is built, from train-period data only). Roughly 46% of positive pairs
are "trivially popular" (overlap only in those 100 files).

**Baselines always reported:** (1) random ranking (analytic; hit@10 about 0.054 on test), (2) most recently merged
first, (3) a query-independent file-popularity prior (rank PRs by the train-period popularity of their files),
(4) BM25 / TF-IDF on issue text vs PR title+body(+paths). Success is a lift over the STRONGEST baseline with a paired
cluster-bootstrap CI that excludes zero on hit@10 and nDCG@10; there is no absolute target number.

**Evaluation discipline.** Training uses train-split queries (1,066 answerable) with negatives drawn from INDEX(q)
at t_q; hyperparameters and model selection use validation (230); test (355 answerable) is evaluated once per model
id after selection, exactly as in Phase 1. CIs: Wilson for rates, bootstrap over queries (353 distinct fix PRs among
the 355 test queries, so clustering is minor; Kish effective n 351). Expected half-width at n = 355: +-5.2 points
(hit rate near 0.5), +-4.8 (near 0.3).

## Numbers behind the go decision (`reports/qp2_prior_pr_counts_kubernetes__kubernetes_2026-10.json`, independently recomputed)

| | Train | Validation | Test |
|---|---|---|---|
| Issues with a fix link | 1,279 | 238 | 375 |
| Fix PR has informative files | 1,227 | 232 | 360 |
| Answerable queries | 1,066 | 230 | **355** |
| Median index size at query time | 3,271 | 7,553 | 9,431 |
| Positives per answerable query (median / mean / max) | 9 / 21.6 / 481 | 24 / 47.4 / 352 | 26 / 56.2 / 959 |

The pre-registered stop rule was: stop if answerable test queries < 200. 355 >= 200, so the task proceeds.
Sensitivity (context only, not used to choose): frequency threshold 0.5% -> 340 answerable test queries, 2% -> 368;
including cherry-picks -> 355.

## Consequences

- ADR-0066 item 3 (same-PR headline) is superseded; items 1, 2 and 4 stand (vscode duplicate Recall@5 >= 80% is
  unchanged; k8s duplicates are reported without a target).
- The task is time-respecting but not a deployment simulation: PRs before 2023 are absent, and it measures "earlier
  PRs touching the files the eventual fix touched", a proxy for "likely code area / recent changers" for Phase 3b.
- About half of the positive pairs are explained by file popularity, which is why the popularity baseline and the
  non-trivial slice are mandatory in every report.
- Positives per query are large (median 26 of about 9,400), but chance hit@10 is only about 5%, so the metrics are not
  saturated by chance.
- Not covered: vscode (its PR files are not scraped; its retrieval target is duplicates).

## Alternatives

- Keep the same-PR task as a report-only sanity check: kept as a descriptive line in the ceiling report only.
- Rank PRs by overlap with the issue's predicted files only: not a retrieval task on text; rejected.
- Include cherry-picks: no effect on the answerable counts (355) but they duplicate master PRs; excluded.
