# ADR-0067 — Kubernetes component taxonomy for the quality push: `sig/*` (frozen before any model is scored)

Status: Accepted (frozen 2026-10-20; changing it requires a new ADR written before any model is scored on the new target)
Date: 2026-10-20
Decider: Gaurav Gandhi approved proposing `sig/*` on the condition that the decision rests on label coverage and routing
semantics only, is taken before any model is scored on it, and is frozen here with the coverage evidence
(decision of 2026-10-20, D79 item 4).

## Context

The old kubernetes component classifier (47 classes) was trained on `area/*` labels from the 2014-2018-era
snapshot. On the fresh 2023-2026 scrape (7,707 issues, issues only; spec-quality-push.md section 4.1) `area/*` is
a rarely used label family, while `sig/*` is the family the project's own triage process uses to route issues.
The evidence comes from `scripts/qp0_taxonomy_coverage.py` (design pre-registered in its docstring, no model or
classifier imported), report `reports/qp0_taxonomy_coverage_kubernetes__kubernetes_2026-10.json`; the headline
coverage numbers were independently recomputed from the raw scraped timelines by a separate verifier pass.

| Family | Coverage at 72 h | Coverage final | Exactly 1 label | 2 or more | Churn (72 h set != final) | Classes (total / >=30 / >=100 issues) | Issues in classes >=30 |
|---|---|---|---|---|---|---|---|
| `area/*` | 6.2% | 6.9% | 6.1% | 0.8% | 16.7% | 43 / 7 / 0 | 67% of labelled |
| `sig/*` | 83.7% | 87.6% | 72.7% | 14.9% | 9.4% | 25 / 21 / 16 | 99.9% of labelled |
| either | 84.1% | 87.9% | 68.0% | 19.9% | 10.4% | 68 / 28 / 16 | |

Per created year, `sig/*` final coverage is 88.2% (2023), 90.3% (2024), 84.1% (2025), 87.4% (2026); `area/*` is 5.4-9.2%
every year. 20 of the 25 `sig/*` classes have at least 5 issues in every year; 3 never reach 5 in any year
(`sig/multicluster`, `sig/ui`, `sig/usability`). 95.3% of `area/*`-labelled issues also carry a `sig/*` label, and
only 7.5% of `sig/*`-labelled issues carry an `area/*` label: `area/*` is a finer, optional annotation nested under
the SIG, not a competing partition.

Routing semantics: in kubernetes a new issue is `needs-triage` / `needs-sig` until a SIG label is applied (by Prow,
on behalf of a human `/sig x` command, or by a maintainer); the owning SIG then triages it. Predicting `sig/*`
predicts who the issue is routed to, which is the question a triage assistant answers. The actor recorded on the
label event is the Prow bot account (`k8s-ci-robot`, and `kubernetes-prow` from 2026-06-20), not the person who
issued the command, so the actor field says nothing about whether a human decided the label.

## Decision

1. **The kubernetes component target is the set of `sig/*` labels on the issue.** `area/*` is not a target.
2. **Modelled classes** are the `sig/*` labels with at least 30 labelled issues in the training period
   (<= 2025-06-30); the remainder are merged into one `other` class. The threshold uses counts only and is fixed here.
3. **Multi-label handling.** 14.9% of `sig/*`-labelled issues carry two or more `sig/*` labels. The task is defined
   with a primary label = the `sig/*` label with the earliest `labeled` event (ties: alphabetical), reported as top-1
   and top-3 accuracy against the primary label, plus an any-valid-label accuracy against the full set. Both are
   reported; neither replaces the other.
4. **Evaluation population:** issues with at least one `sig/*` label at their final state (87.6% of issues). Issues
   never labelled are not scored and their share is reported next to every result.
5. **This is a new task.** Results are reported as the new `sig/*` task and are never compared with the old
   47-class `area/*` accuracy (85.99% top-3), in the README, reports, PRs or ADRs.
6. No model is scored on this taxonomy before this ADR is merged.

## Consequences

- The k8s classifier number in the README will change meaning in Phase 6; the old row is retired, not updated.
- A SIG-ownership target is coarser than `area/*` but has about 13x the label coverage and a stable class set.
- Coverage dipped to 84.1% in 2025 and the 2026 test split falls after the Prow account rename; both are reported with
  results rather than corrected for.
- Labels are human routing decisions executed by Prow, so the target contains maintainer judgement and the ceiling
  analysis (Phase 0.4) still measures label churn and multi-label disagreement before any target is set.

## Alternatives

- `area/*`: rejected on coverage (6.9% of issues; 7 classes with at least 30 issues).
- Union of `sig/*` and `area/*`: rejected, it adds 43 sparse classes to a 25-class family and mixes two label
  semantics (ownership and topic).
- Keep the old 47 `area/*` classes for continuity: rejected, the label family is no longer used on most issues.
- vscode is not covered here; its taxonomy is decided in a separate ADR once its data and coverage report exist.
