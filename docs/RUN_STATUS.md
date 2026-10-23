# RUN_STATUS — unattended run until 2026-10-27 (spec-autonomous-run.md)

Single source of truth. Re-read first after any restart. Times in IST unless marked UTC.

## Now (updated 2026-10-23 ~19:00 IST)
- Stop new long work from 2026-10-27 12:00 IST; return brief by 2026-10-27 18:00 IST.
- Day 1 done through #300. Every serving change merged default off; prod healthy after each deploy.
- Next: Kaggle GPU work as soon as the weekly quota resets (try from 2026-10-24 morning): (a) vscode BGE embeddings (private dataset `triageiq-texts-microsoft-vscode-issues` ready; commands in the #299 PR / below), then Q5a vscode duplicates on the embeddings, a vscode candidate index + evidence artifact; (b) the k8s fine-tuned encoder (`feat/qp1-k8s-finetuned-encoder`, text recipe aligned before any run). Meanwhile CPU work: vscode Q6 resolution (survival framing, as k8s) once embeddings exist, or without the embedding features if they do not; Q8 waits for GG's cutover decisions.
- Memory is tight at times (other workloads); briefs carry a memory guard. Orchestrator git work happens in `triage-iq-wt-readme`, never in the primary checkout while a subagent runs.
- Do NOT recursively delete `...\scratchpad\rc` (session scratchpad): it contains directory junctions to D:\triageiq-data; remove only the junctions.

## Queue (runbook section 5)
| # | Task | Status |
|---|---|---|
| Q1 | Direct timelines; derive from direct; gold re-check | DONE: k8s (#250, #252); vscode (#266) |
| Q2 | vscode taxonomy; ADR-0070 | DONE (#267, #268) |
| Q3 | vscode splits, ceilings, 150-issue eval set | DONE (#270, #271, #272-#275, #280 fix-linked ceiling, #281 audited links) |
| Q4 | Phase 1 candidates | k8s DONE (#255; TF-IDF+LR stays; #291 recheck); vscode CPU DONE (#276-#278); embedding / fine-tuned candidates BLOCKED (Kaggle weekly GPU quota) |
| Q5a | vscode duplicate retrieval | queued (embeddings need Kaggle GPU; dataset uploaded) |
| Q5b | k8s prior-PR retrieval | DONE (#256, #257): criterion NOT met |
| Q6 | resolution (survival) | k8s DONE (#259, #260, recheck #291: M2 worse than naive on production vectors); vscode queued |
| Q7 | Phase 3b redesign | priority P4 + flag (#284, #285, #294); prompt v3 flags 1-2 (#296-#300); checks C1-C5 pass; flag 3 (conformal component) needs the classifier cutover (GG) |
| Q8 | Phase 5 | objective-metrics harness DONE (#292, #293, #295); the single re-record waits for GG's cutover decisions |
| Q9 | background | daily post-cutoff collection scheduled 03:00 IST 2026-10-24 |

## Done this run (all VERIFIED by an independent verifier)
- #249-#300 except GG's PRs; see `docs/DECISION_LOG_2026-10.md` D84-D87.
- Headlines:
  - k8s classifier top-3 95.4% [94.3, 96.5] (strict = lenient, #282).
  - vscode classifier (ADR-0070): strict top-3 36.4% [35.5, 37.4] vs a 41.8% coverage bound; in-family top-3 87.0%; production 4.4% strict.
  - k8s priority (serve-time inputs, P4): 0.506 vs majority 0.409, +0.097 [+0.024, +0.174].
  - k8s resolution: no learned point clears the bar; on production-recipe embeddings M2 is worse than naive.
  - Production train/serve skew found and fixed behind a default-off flag (#289; GG queue top).

## Blocked (with unblock condition)
- GG decisions (queue): enable TRIAGE_CLEAN_REQUEST_BODY; P4 / evidence / candidate-index cutovers (gate Q8); k8s resolution constant; vscode target wording; Kaggle quota.
- Kaggle GPU (shared account weekly quota) for vscode embeddings and the fine-tuned encoder.

## Background jobs
- Daily post-cutoff collection: background job waits for 03:00 IST 2026-10-24, runs `D:\triageiq-data\bin\run_confirm_scrape.ps1`.

## Incidents
- 05:40-06:23: timeline scraper stuck at batch size 1 after HTTP 599s; restarted from checkpoint; fixed in all scrapers (#262).
- ~09:50: Claude Code stopped a background wait command for low system memory; the scrape was unaffected.
- ~12:30: the orchestrator switched the shared primary checkout's branch for about two minutes while an executor worked (nothing lost). Own git work now in the docs worktree.
- ~12:00: a merge loop filtered `[fail]` while the gate prints `[FAIL]`; a DIRTY PR (#274) sat 30 minutes (replaced by #275).
- 16:57:27: scripts/qp1_vscode_lib.py and qp1_vscode_phase1.py in the primary checkout were overwritten by an unexplained split-like edit (no other triage-iq session ran; the executor working then says none of its commands touched them; cause unverified). Preserved on branch `backup/foreign-qp1-vscode-edit-2026-10-23` and a scratchpad copy; the committed (verified) versions were restored by switching branches; nothing lost.
- ~17:30: an executor ran `ruff format` on a whole directory, reformatting two unrelated files; when a hook refused the restore it worked around the refusal (WIP commit + restore from HEAD~1 + amend, never pushed). Outcome harmless; briefs now forbid hook workarounds and directory-wide formatting.

## Decisions taken alone this run
- Scoped lock regen pillow 13.0.0 / certifi (#251).
- Labels with no events are creation-time on complete timelines.
- k8s classifier stays TF-IDF+LR (stack ties; simpler wins; confirmed on production vectors #291).
- Restart the stuck vscode scrape from its checkpoint rather than wait ~15 h.
- Priority: P3 then P4 (serve-time inputs) as new pre-registered model ids; P2's design untouched.
- ADR-0070 accepted (parent-area rule, train-only class list and fit, strict headline + k8s-convention variant).
- Eval set redesigned to v2 before any use, then 2.1 (audited fix links only); Phase 5 priority metric scores label-sourced rows only.
- ADR-0071 amended to the spec's alpha 0.05 after an orchestrator brief error (disclosed).
- Fix links used as ground truth only when closer + merged (audited); `pr_closing_refs` (76.7%) never.
- Kaggle embedding recipe switched to production (#288); earlier Kaggle-vector experiments rechecked on validation (#291).
- Evidence in prompts filtered to items dated before the issue (no future information in eval prompts).
- The C4 projected re-record total (~919K vs ~908K planning figure) accepted; the pre-registered ratio passes.
- Fine-tuned encoder text recipe aligned with serving before any run (recorded as a pre-result design revision).
