# RUN_STATUS — unattended run until 2026-10-27 (spec-autonomous-run.md)

Single source of truth. Re-read first after any restart. Times in IST unless marked UTC.

## Now
- Started 2026-10-23. Stop new long work from 2026-10-27 12:00 IST; return brief by 2026-10-27 18:00 IST.
- Current task: **Q1 timeline fix** (step 1: build `scripts/qp0_scrape_timelines.py`).

## Pause-state check at start (2026-10-23, VERIFIED)
- git: primary checkout clean apart from pre-existing untracked files (main_pinned_repro.log, scripts/audit_fallback_plans.py,
  scripts/diag_judge_key_v2.py); handoff PR #248 merged; open PRs #184, #138, #94 (GG's).
- No orphaned TriageIQ processes. main CI green on 31cf1b8. Production `/health?deps=1` ok (model store + Groq healthy).
- No open alert issues. GitHub rate limit core 4,870 / GraphQL 4,917. D:\triageiq-data 369 MB; C: 228 GB free, D: 1.4 TB free.

## Queue (runbook section 5, mapped from the handoff resume plan)
| # | Task | Depends on | Status |
|---|---|---|---|
| Q1 | Direct per-issue timeline scraper; run k8s then vscode; quantify loss vs search-embedded timelines; derive reads direct timelines; re-derive both; recompute label-timeline reports; re-check k8s Phase 1 gold (re-run as a new gold version if it changes); then merge or rework the held branches feat/qp0-label-replay-fix and feat/qp0-vscode-audits | - | IN PROGRESS |
| Q2 | vscode taxonomy: train-counts-only class family; evaluate (a) recent training window, (b) open-set abstention as a counted outcome, (c) stable parent areas, (d) scheduled retraining; freeze ADR-0070; amend ADR-0067 (primary rule) and ADR-0068 (counts) | Q1 vscode | queued |
| Q3 | vscode splits, ceilings, 150-issue LLM eval set (75 per repo) | Q2 | queued |
| Q4 | Phase 1 on Kaggle: vscode candidates; embedding candidates both repos (BGE head, kNN vote, fine-tuned encoder ONNX int8, stacking); conformal chosen on validation only | Q3 (vscode); k8s part after Q1 gold check | queued |
| Q5a | vscode duplicate retrieval (7,339 pairs; Recall@5 >= 80% target) | Q3 | queued |
| Q5b | k8s prior-PR retrieval (ADR-0068): PR files already scraped; refresh counts after Q1; score baselines then models | Q1 (fix-link events) | queued |
| Q6 | Phase 4 resolution (survival), Kaggle CPU or light local | Q1 | queued |
| Q7 | Phase 3b redesign, implementation only, default-off flags, finalize prompt | Q4, Q5, Q6 | queued |
| Q8 | Phase 5 metrics harness; single re-record only after Q7 acceptance | Q7 | queued |
| Q9 | Background: post-cutoff k8s issues daily (ADR-0069 confirmation set); dependabot pip job; disk/scratch hygiene | - | ongoing |

## Done this run
- (none yet)

## Blocked (with unblock condition)
- (none yet)

## Background jobs
- (none running)

## Decisions taken alone this run
- (none yet)
