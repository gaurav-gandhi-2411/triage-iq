# RUN_STATUS — unattended run until 2026-10-27 (spec-autonomous-run.md)

Single source of truth. Re-read first after any restart. Times in IST unless marked UTC.

## Now (updated 2026-10-23 ~15:45 IST)
- Stop new long work from 2026-10-27 12:00 IST; return brief by 2026-10-27 18:00 IST.
- Since 13:30: vscode fix-link audit (#280, closer 99.2% pass); audited-links-only fix (#281); k8s strict re-score (#282, = lenient); Phase 3b design + checks (#283); priority P4 (#284) and its default-off flag (#285, deployed dormant, prod healthy); candidate k8s index + evidence, C3 passes (#286, #287); Kaggle text recipe aligned with production (#288); a verified production train/serve skew and its default-off fix flag (#289, top of the GG queue).
- Next: Q7 remaining (flag 3 component-from-conformal-set; prompt v3 with flags 1-3 and check C4 tokens) -> validation-only re-check of the Kaggle-embedding experiments on production-recipe vectors -> Q8 objective-metrics harness (scorers for the 150-issue set, zero tokens) -> vscode work that needs the Kaggle GPU after the weekly reset (align the fine-tuned-encoder branch's text recipe first).
- Memory is tight at times (other workloads); briefs carry a memory guard. Orchestrator git work happens in `triage-iq-wt-readme`, never in the primary checkout while a subagent runs.

## Queue (runbook section 5)
| # | Task | Status |
|---|---|---|
| Q1 | Direct timelines; derive from direct; gold re-check | DONE: k8s (#250, #252); vscode (#266) |
| Q2 | vscode taxonomy; ADR-0070 | DONE (#267, #268) |
| Q3 | vscode splits, ceilings, 150-issue eval set | DONE (#270, #271; eval set #272, #273, #275); vscode fix-linked ceiling waits for the fix-link audit (in progress) |
| Q4 | Phase 1 candidates | k8s DONE (#255; TF-IDF+LR stays); vscode CPU DONE (#276-#278); embedding / fine-tuned candidates BLOCKED (Kaggle weekly GPU quota) |
| Q5a | vscode duplicate retrieval | queued (BGE embeddings need Kaggle GPU) |
| Q5b | k8s prior-PR retrieval | DONE (#256, #257): criterion NOT met; trained retriever/reranker is the next attempt |
| Q6 | resolution (survival) | k8s DONE (#259, #260); serving decision queued for GG; vscode later |
| Q7 | Phase 3b redesign | k8s priority study DONE (#263, #264); kNN resolution DONE (#265, negative); serving flags, evidence artifact, prompt: queued |
| Q8 | Phase 5 | queued (eval set ready; re-freeze with the new retriever before the single re-record) |
| Q9 | background | scraper batch-regrow fix (#262); daily post-cutoff collection |

## Done this run (all VERIFIED by an independent verifier unless marked)
- #249-#278 except GG's PRs; see `docs/DECISION_LOG_2026-10.md` D84 and D85.
- Headlines:
  - k8s classifier top-3 95.4% [94.3, 96.5] (unchanged; lenient `other` convention, see ADR-0070).
  - vscode classifier (new task, ADR-0070): strict top-3 36.4% [35.5, 37.4] vs a 41.8% coverage bound (label drift in 2026), in-family top-3 87.0% [85.9, 88.0], k8s-convention 58.8%; production 4.4% strict. Conformal (exploratory, alpha 0.05): in-family coverage 94.995%, mean set 6.81.
  - k8s priority: serve-time-feasible P3 0.522 vs majority 0.409, +0.113 [+0.032, +0.190] (criterion met; prioritised issues only).
  - k8s resolution: no learned point clears the bar on the full (censored) population; kNN does not beat naive.
  - vscode data: embedded timelines missed >= 1 label event on 15.0% of issues (k8s 1.6%).

## Blocked (with unblock condition)
- k8s resolution serving / production constant (3.04 d from 2014-2016 vs 35.1 d on 2023-2025): needs GG (live behavior).
- vscode component target wording (spec section 1): needs GG (pre-registered target); evidence now includes the measured 36.4% vs 41.8% bound.
- k8s fine-tuned encoder (branch `feat/qp1-k8s-finetuned-encoder`, pushed, not merged; design pre-registered; private dataset ready, test labels never uploaded) and vscode BGE embeddings: Kaggle weekly GPU quota (shared account). Resume with `python scripts/qp1_finetuned_encoder.py push`, then `qk_kaggle.py wait/download`, `validate`, commit, `final-test`.

## Background jobs
- None running.
- Post-cutoff issue collection (ADR-0069): `run_confirm_scrape.ps1`; rerun daily (next due ~03:00 IST 2026-10-24).

## Incidents
- 05:40-06:23: timeline scraper stuck at batch size 1 after HTTP 599s; restarted from checkpoint; fixed in all scrapers (#262).
- ~09:50: Claude Code stopped a background wait command for low system memory; the scrape was unaffected.
- ~12:30: the orchestrator switched the shared primary checkout's branch for about two minutes while an executor worked (its tree was clean; nothing lost or mis-run). Own git work now in the docs worktree.
- ~12:00: a merge loop filtered `[fail]` while the gate prints `[FAIL]`; a DIRTY PR (#274) sat unnoticed for 30 minutes and was replaced by a clean branch (#275).

## Decisions taken alone this run
- Scoped lock regen pillow 13.0.0 / certifi (#251).
- Labels with no events are creation-time on complete timelines.
- Accept Kaggle text-recipe drift (cosine 0.997-0.999 vs production) for experiments.
- k8s classifier stays TF-IDF+LR (stack ties; simpler wins).
- Restart the stuck vscode scrape from its checkpoint rather than wait ~15 h.
- Priority: P3 (creation-time labels) added as a new pre-registered model id after the verifier showed P2 relied on later labels.
- ADR-0070 accepted (parent-area rule, train-only class list and fit, strict headline + k8s-convention variant).
- Eval set redesigned to v2 before any use; Phase 5 priority metric scores label-sourced rows only.
- ADR-0071 amended to the spec's alpha 0.05 after an orchestrator brief error (disclosed; alpha-0.1 test numbers kept as superseded).
