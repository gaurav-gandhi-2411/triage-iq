# RUN_STATUS — unattended run until 2026-10-27 (spec-autonomous-run.md)

Single source of truth. Re-read first after any restart. Times in IST unless marked UTC.

## Now (updated 2026-10-23 ~11:00 IST)
- Stop new long work from 2026-10-27 12:00 IST; return brief by 2026-10-27 18:00 IST.
- Current subagent task: Q3a vscode temporal splits + label/retrieval ceilings (feat/qp0-vscode-splits).
- Next: Q3b 150-issue LLM eval set (75/repo) -> Q4 vscode classifier candidates (CPU: TF-IDF+LR family; embedding candidates wait for Kaggle GPU) -> Q7 remaining (next-step evidence artifact, component-from-conformal-set flag, priority flag, prompt) -> k8s strict `other` re-score (secondary).
- Memory is tight (~4-9 GB free; other workloads); briefs carry a memory guard.

## Queue (runbook section 5)
| # | Task | Status |
|---|---|---|
| Q1 | Direct timelines; derive from direct; gold re-check | DONE: k8s (#250, #252); vscode (#266: 15.0% of issues missed label events; derived_v2 built) |
| Q2 | vscode taxonomy; ADR-0070 | DONE (#267 evidence, #268 ADR): 85 parent classes, validation coverage 88.7%; target amendment queued for GG |
| Q3 | vscode splits, ceilings, 150-issue eval set | IN PROGRESS (splits + ceilings) |
| Q4 | Phase 1 Kaggle candidates | k8s embedding candidates DONE (#255; TF-IDF+LR stays); k8s fine-tuned encoder BLOCKED (Kaggle weekly GPU quota); vscode after Q3 |
| Q5a | vscode duplicate retrieval | queued (needs Q3; BGE embeddings need Kaggle GPU) |
| Q5b | k8s prior-PR retrieval | DONE (#256, #257): criterion NOT met; trained retriever/reranker is the next attempt |
| Q6 | resolution (survival) | k8s DONE (#259, #260); serving decision queued for GG; vscode later |
| Q7 | Phase 3b redesign | k8s priority study DONE (#263, #264); kNN resolution study DONE (#265, negative); serving flags, evidence artifact, prompt: queued |
| Q8 | Phase 5 | queued |
| Q9 | background | ongoing; scraper batch-regrow fix merged (#262) |

## Done this run (all VERIFIED by an independent verifier unless marked)
- #249 runbook + RUN_STATUS; #251 pillow/certifi lock fix; #250 direct timeline scraper; #252 derive on direct timelines + k8s gold v2 (unchanged); #253, #258, #261 status; #254 Kaggle BGE pipeline; #255 k8s embedding candidates; #256/#257 k8s prior-PR counts v2 and eval; #259/#260 k8s resolution features and models; #262 scrapers regrow batch size; #263/#264 k8s priority study; #265 k8s kNN resolution; #266 vscode timeline completeness; #267/#268 vscode taxonomy + ADR-0070.
- Headlines:
  - k8s classifier top-3 95.4% [94.3, 96.5] (unchanged; lenient `other` convention, see ADR-0070).
  - k8s priority (3-level, 247 test issues maintainers prioritised): serve-time-feasible P3 (text + creation-time labels) 0.522 vs majority 0.409, gain +0.113 [+0.032, +0.190] (criterion met). P2 with labels up to 72 h scored 0.526 but is not serve-time valid.
  - k8s resolution: kNN (K1) does not beat the naive 35.1 d median (-0.80 d [-4.66, 5.53]); K1 x M2 combination passes the median-AE rule but is worse than M2 alone; no learned point clears the bar on the full (censored) population.
  - vscode: embedded timelines missed >= 1 label event on 15.0% of issues (k8s 1.6%); share with a label at 72 h falls 72.1% -> 66.9% with complete timelines.
  - vscode taxonomy: 85 parent classes cover 88.7% [87.9, 89.5] of labelled validation issues; top-3 >= 95% over all labelled issues is not reachable as written.

## Blocked (with unblock condition)
- k8s resolution serving / production constant (3.04 d from 2014-2016 vs 35.1 d on 2023-2025): needs GG (live behavior). Queued.
- vscode component target wording (spec section 1): needs GG (pre-registered target). Queued with a recommendation.
- k8s fine-tuned encoder (branch `feat/qp1-k8s-finetuned-encoder`, pushed, not merged; design pre-registered; private dataset `triageiq-ft-sigcls-k8s` ready, test labels never uploaded) and vscode BGE embeddings: Kaggle `Maximum weekly GPU quota of 30.00 hours reached` (shared account). Unblocks at the weekly reset; resume with `python scripts/qp1_finetuned_encoder.py push`, then `qk_kaggle.py wait/download`, `validate`, commit, `final-test`.

## Background jobs
- None running. vscode direct-timeline scrape finished 09:50 IST (55,612 issues in the final run, 4 missing).
- Post-cutoff issue collection (ADR-0069): `run_confirm_scrape.ps1`; first pass done 02:51 IST; rerun daily (next due ~03:00 IST 2026-10-24).

## Incidents
- 2026-10-23 05:40-06:23: after a burst of transient HTTP 599s the timeline scraper dropped to batch size 1 and never grew back. Stopped the scraper child by PID; the runner restarted it from the atomic checkpoint (no rows lost). Fixed in all four scrapers (#262).
- 2026-10-23 ~09:50: Claude Code stopped a background wait command for low system memory (other workloads). The scrape itself was unaffected; not restarted.

## Decisions taken alone this run
- Scoped lock regen pillow 13.0.0 / certifi (#251).
- Labels with no events are creation-time on complete timelines.
- Accept Kaggle text-recipe drift (cosine 0.997-0.999 vs production) for experiments.
- k8s classifier stays TF-IDF+LR (stack ties; simpler wins).
- Restart the stuck vscode scrape from its checkpoint rather than wait ~15 h.
- Priority: added P3 (creation-time labels) as a new pre-registered model id after the verifier showed P2 relied on labels added up to 72 h later; P2's design left untouched, caveats in `post_hoc_notes`.
- ADR-0070 accepted: parent-area rule, train-only class list and final fit (as k8s), abstention scored as a miss in the headline (runbook Q2(b)), k8s-convention variant reported alongside.
