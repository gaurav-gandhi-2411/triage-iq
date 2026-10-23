# RUN_STATUS — unattended run until 2026-10-27 (spec-autonomous-run.md)

Single source of truth. Re-read first after any restart. Times in IST unless marked UTC.

## Now (updated 2026-10-23 ~08:45 IST)
- Stop new long work from 2026-10-27 12:00 IST; return brief by 2026-10-27 18:00 IST.
- Current subagent task: fix the timeline scraper so batch size grows back after transient errors (fix/qp0-timeline-batch-regrow).
- Next: vscode re-derive on direct timelines as soon as the scrape finishes (Q1 vscode) -> Q2 taxonomy -> Q3. Kaggle GPU work waits for the weekly quota reset.

## Queue (runbook section 5)
| # | Task | Status |
|---|---|---|
| Q1 | Direct timelines; derive from direct; gold re-check | k8s DONE (#250, #252); vscode WAITING on scrape (~76% at 08:30) |
| Q2 | vscode taxonomy; ADR-0070 | queued (needs Q1 vscode) |
| Q3 | vscode splits, ceilings, 150-issue eval set | queued (needs Q2) |
| Q4 | Phase 1 Kaggle candidates | k8s embedding candidates DONE (#255; TF-IDF+LR stays); k8s fine-tuned encoder BLOCKED (Kaggle weekly GPU quota); vscode after Q3 |
| Q5a | vscode duplicate retrieval | queued (needs Q3; BGE embeddings need Kaggle GPU) |
| Q5b | k8s prior-PR retrieval | DONE (#256, #257): criterion NOT met; trained retriever/reranker is the next attempt |
| Q6 | resolution (survival) | k8s DONE (#259, #260); serving decision queued for GG; vscode later |
| Q7 | Phase 3b redesign | queued |
| Q8 | Phase 5 | queued |
| Q9 | background | ongoing |

## Done this run (all VERIFIED by an independent verifier unless marked)
- #249 runbook + RUN_STATUS; #251 pillow/certifi lock fix; #250 direct timeline scraper; #252 derive on direct timelines + k8s gold v2 (unchanged); #253 status; #254 Kaggle BGE pipeline; #255 k8s embedding candidates; #256/#257 k8s prior-PR counts v2 and eval; #259/#260 k8s resolution features and models.
- Headlines: k8s classifier top-3 95.4% [94.3, 96.5] (unchanged; no new candidate beats it); k8s prior-PR task hybrid hit@10 0.715 vs TF-IDF 0.673, criterion not met; k8s resolution: no learned point clears the bar on the full (censored) population.

## Blocked (with unblock condition)
- k8s resolution serving / production constant (3.04 d from 2014-2016 vs 35.1 d on 2023-2025): needs GG (live behavior). Queued.
- k8s fine-tuned encoder (branch `feat/qp1-k8s-finetuned-encoder`, pushed, not merged; design pre-registered in the script docstring; private dataset `triageiq-ft-sigcls-k8s` ready, test labels never uploaded): Kaggle reports `Maximum weekly GPU quota of 30.00 hours reached` (shared account). Unblocks at the weekly reset; resume with `python scripts/qp1_finetuned_encoder.py push`, then `qk_kaggle.py wait/download`, `validate`, commit, `final-test`. Queued for GG as FYI.

## Background jobs
- vscode direct-timeline scrape: `D:\triageiq-data\bin\run_timelines.ps1`, log `D:\triageiq-data\logs\scrape_timelines_microsoft__vscode.log`, done marker `timelines_done.txt`; 83,072 of 109,164 issues at 08:29 IST.
- Post-cutoff issue collection (ADR-0069): `run_confirm_scrape.ps1`; first pass done 02:51 IST; rerun daily.

## Incidents
- 2026-10-23 05:40-06:23: after a burst of transient HTTP 599s the timeline scraper dropped to batch size 1 and never grew back (~15 h left instead of ~1.5 h). Stopped the scraper child by PID; the runner restarted it from the atomic checkpoint at batch 10 (no rows lost; derive keys rows by number). Code fix in progress.

## Decisions taken alone this run
- Scoped lock regen pillow 13.0.0 / certifi (#251).
- Labels with no events are creation-time on complete timelines.
- Accept Kaggle text-recipe drift (cosine 0.997-0.999 vs production) for experiments.
- k8s classifier stays TF-IDF+LR (stack ties; simpler wins).
- Restart the stuck vscode scrape from its checkpoint rather than wait ~15 h.
