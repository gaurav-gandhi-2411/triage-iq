# RUN_STATUS — unattended run until 2026-10-27 (spec-autonomous-run.md)

Single source of truth. Re-read first after any restart. Times in IST unless marked UTC.

## Now (updated 2026-10-23 ~10:00 IST)
- Stop new long work from 2026-10-27 12:00 IST; return brief by 2026-10-27 18:00 IST.
- Current subagent task: **Q4 prep — Kaggle embedding pipeline** (branch feat/qk-kaggle-embeddings; k8s issues + PRs first).
- Q1 status: k8s DONE (direct timelines, re-derive, gold check); vscode waiting on the direct-timeline scrape (background).

## Pause-state check at start (2026-10-23, VERIFIED)
- git clean apart from pre-existing untracked files; handoff PR #248 merged; open PRs then #184, #138, #94 (GG's).
- No orphaned TriageIQ processes. main CI green on 31cf1b8. Production `/health?deps=1` ok. No open alert issues.
- GitHub rate limit core 4,870 / GraphQL 4,917. D:\triageiq-data 369 MB; C: 228 GB free, D: 1.4 TB free.

## Queue (runbook section 5)
| # | Task | Depends on | Status |
|---|---|---|---|
| Q1 | Direct timelines; quantify loss; derive from direct; re-derive; re-check k8s gold | - | k8s DONE (PR #250 scraper, #252 derive + gold re-score); vscode WAITING on scrape |
| Q2 | vscode taxonomy (train-counts-only family; recent window / open-set / parent areas / retraining design); ADR-0070 | Q1 vscode | queued |
| Q3 | vscode splits, ceilings, 150-issue LLM eval set | Q2 | queued |
| Q4 | Phase 1 on Kaggle: vscode candidates; embedding candidates both repos; conformal on validation only | Kaggle pipeline (in progress); vscode after Q3 | IN PROGRESS (pipeline) |
| Q5a | vscode duplicate retrieval (Recall@5 >= 80%) | Q3 + embeddings | queued |
| Q5b | k8s prior-PR retrieval (ADR-0068): refresh counts on derived_v2, baselines, models | Q1 k8s (done) | queued (next after Q4 prep) |
| Q6 | Phase 4 resolution (survival) | Q1 | queued |
| Q7 | Phase 3b redesign (default-off flags) | Q4, Q5, Q6 | queued |
| Q8 | Phase 5 metrics harness; single re-record after Q7 acceptance | Q7 | queued |
| Q9 | Background: post-cutoff issues (ADR-0069); dependabot pip job; disk hygiene | - | ongoing |

## Done this run
- Runbook committed, RUN_STATUS started (#249).
- CI unblocked: pillow 12.3.0 -> 13.0.0 (35 CVEs) and certifi -> 2026.10.19 via scoped lock regen (#251). VERIFIED (CI green after).
- Q1 k8s (VERIFIED by an independent verifier): direct timeline scraper (#250); k8s direct timelines 7,707 issues, 77,491 events, 0 missing; embedded data was missing >= 1 LabeledEvent on 122 k8s issues (1.58%); closer-PR fix links 1,956 -> 1,969; duplicate pairs unchanged (98). k8s Phase 1 gold IDENTICAL under corrected timelines (0 primary/set changes in all splits) -> k8s Phase 1 numbers stand: test top-3 95.4% [94.3, 96.5] (point meets the target, lower bound does not), top-1 77.1%. Derive now prefers direct timelines (#252).

## Blocked (with unblock condition)
- (none)

## Background jobs
- vscode direct-timeline scrape: runner `D:\triageiq-data\bin\run_timelines.ps1` (log `D:\triageiq-data\logs\scrape_timelines_microsoft__vscode.log`, done marker in `timelines_done.txt`); ~10,900 batches, ~4-5 h from 03:00 IST.
- Post-cutoff issue collection (ADR-0069 confirmation set): `run_confirm_scrape.ps1`, first pass done 02:51 IST (k8s + vscode since 2026-10-19); rerun daily.
- PR merge monitors: #250 then #252.

## Decisions taken alone this run
- Scoped lock regen for pillow 13.0.0 / certifi (runbook 7: CI broken by advisory). No other pin moved; pillow is transitive.
- Treat labels with no labeled/unlabeled event as present from creation (correct once timelines are complete; on k8s 0 sig/* labels are affected).
