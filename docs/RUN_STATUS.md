# RUN_STATUS — unattended runs (spec-autonomous-run.md; second run from 2026-10-28 20:40 IST, no stop date, amendment A1)

Single source of truth. Re-read first after any restart. Times in IST unless marked UTC. Newest dated brief first; everything under
"First run" is history.

## Update 2026-10-29 ~20:30 IST (after GG's decisions D94)

**D94 (GG, 2026-10-29):**
1. Run the switch-ons in order.
2. Retire the 64-issue set as the CI eval set: the 150 set becomes the CI set, the 64-set cassettes are kept as history, and only the 150 set
   is re-recorded (~919K tokens). Move #320's gate onto the 150 set and write an ADR.
3. Report the before/after; do not write a baseline.
4. Exclude `papercut 🩸`; KEEP `performance`, recorded in ADR-0073.

**Production (VERIFIED ~20:20).** All three switch-ons are live on `triageiq-api-00108-mod` at 100%. #338 merged as 36a8986, and its CI, Eval
gate and Deploy were all success. Each switch-on went no-traffic candidate → smoke → comparison → promote → live check:

| step | revision | smoke and comparison result |
|---|---|---|
| D92.1 family_v2 | `00104-zax`, tag d921 | Candidate top-1 `extensions` 0.9958009257 equals the offline family_v2 prediction on the deploy-smoke issue |
| D92.4 k8s v2 | `00106-sew`, tag d924 | 3 frozen-set k8s rows: candidate neighbours are the frozen v2 neighbours (5/5, 4/5, 1/5), and every extra is newer than the query (serving has no created-before filter); old prod index 0/5 |
| D92.5 evidence | `00108-mod`, tag d925 | No missing/undated warning; k8s triage returns `_evidence_neighbours` [119813, 122021]; the LLM answered on the candidate |

- Env on the live revision: ENVIRONMENT=prod, TRIAGE_CLEAN_REQUEST_BODY=1, TRIAGE_VSCODE_CLASSIFIER=family_v2, TRIAGE_K8S_INDEX=v2,
  TRIAGE_RESOLUTION_EVIDENCE=1.
- Rollback: `gcloud run services update-traffic triageiq-api --to-revisions <rev>=100`, using `00106-sew` (no evidence), `00104-zax`
  (family_v2 only) or `00102-vep` (all off).
- The Groq daily quota was intermittently `rate_limited_tpd` during the checks. The LLM path answered on the d925 candidate.
- Draft #357 (GG, workflow) writes the three flags into deploy.yml. Until then they persist on the service through `--update-env-vars`.

**R8 vscode duplicate fine-tune (done).**
- #351 prereg, #352 kernel, #353 driver, #354 frozen selection and test guard, #356 results.
- Kaggle run: private, 1.37 T4 hours, no quota error.

| split | #345 winner R@5 | F4 + RRF k=10 R@5 | paired gain |
|---|---|---|---|
| validation | 40.3% | 45.3% | +5.0 [+3.55, +6.57] |
| test (one run, from main f5a94a2) | 33.0% (reproduced exactly) | 40.4% [38.0, 42.7] | +7.3 [+5.9, +8.9] |

- The pre-registered criteria are met. ADR-0066's 80% target is NOT met.
- Nothing is served from it. Results: `docs/Q5A_FINETUNE_RESULTS_2026-10.md`.

**In progress.**
- ADR-0073 label decisions: executor running.
- Plan for moving the gate to the 150 set: read-only agent running.
- Next: the `vscode-retrain-data` release, then tell GG #326 is ready. Then regenerate EXPECTED_ARTIFACT_HASHES and run the 150-set re-record
  under the ledger.

## Brief 2026-10-29 (24 h, A1), written ~17:00 IST

**Outcome.** D92 serving steps 1-2 are ready and verified up to the point that needs GG: the retrained vscode classifier is built under the written
meta-label rule, proven bit-identical offline vs serving, and uploaded, and it waits in draft #338 together with the k8s index v2 and evidence v2
artifacts. All five oversized PRs are split and merged (D93.1). Re-record preparation that spends no tokens is merged. Production is unchanged
in behaviour and healthy.

**Production (VERIFIED 16:00).** Revision `triageiq-api-00100-riw` at 100% (deployed from #341, 70aa989; CI, Eval gate and Deploy success),
`/health?deps=1` ok, 2 repos. Env: ENVIRONMENT=prod, TRIAGE_CLEAN_REQUEST_BODY=1 (G1; codified in deploy.yml by GG's merge of #329); no other
TRIAGE_* flag set. Rollback target: `triageiq-api-00098-kuw`.

**Done today (each VERIFIED by tests, a verifier or a reproduced check, as stated in its PR).**
- D92.2 / D93.2 meta labels: rule written first (ADR-0073 addendum), applied to all 703 vscode labels before any score; 29 new exclusions
  (20 EXCLUDE + 9 BORDERLINE-EXCLUDE), 14 BORDERLINE-KEEP (#333-#335). The rule-before-table order is visible in commits 9b2f1a5 (12:49, rule
  only) then ce55978 (12:55, table + code) on the kept #325 branch; #333 itself is one commit.
- D92.1 retrained vscode classifier (dry run `run4_rule`, code ce55978): strict top-3 on the 1,521-issue 2026 holdout 60.4% [57.9, 62.9] vs served
  champion 11.4% vs naive 15.1%; paired vs champion +49.0 pts [46.4, 51.6]; gate accept 7/7; 115 classes, none excluded (independent verifier).
  Bundle sha256 f4082316…5573. Serving parity on it: 300/300 identical top-3 incl. order, max abs diff 0.0. Uploaded to GCS (no-clobber,
  read back). Not comparable with run3_d2's 54.9% (different holdout and class list).
- D92.4 / D92.5: k8s index v2 and evidence v2 uploaded to new GCS paths (no-clobber, all hashes read back); loader flag `TRIAGE_K8S_INDEX=v2`
  (#341, fail closed, manifest-checked). #338 (GG) bakes all three artifacts in with every flag off; its CI is green.
- D92 step 4: #320's three 3.04 d lines fixed, plus the stale k8s interval rows, the architecture box and the EVALUATION resolution note;
  coverage quoted as censoring-aware 77.3-84.5%; ADR-0074 records G3 and D92.3; strict numbers checker OK; full suite passed (174b6f3).
- D93.1 splits: #325 → #333/#334/#335; #327 → #336/#337; #330 → #339/#342/#343 (byte- and AST-identical); #323 → #344/#345. #326 retargeted to
  main (draft, workflow only, waits for the release).
- Re-record prep (no tokens): #346 shared config-aware assistant builder (`RECORDING_CONFIG.json`; config absent = 127-issue replay 0 differing);
  #347/#348 the 150-issue set re-frozen on the k8s v2 index (75/75 rows reproduced independently, 0 rows without neighbours, C3 evidence
  coverage 93.3% vs the 50% bar), `body_raw` on all rows, set config = family_v2 + v2 + evidence.
- R3 collection ran 16:16 (k8s 5 rows, vscode 39 rows in today's window).
- Correction (repeated from my earlier report): the served vscode model's 9.0-9.3% strict top-3 on the 2026 holdout is real; the 53.4% I had
  compared it with is the lenient metric.

**GG queue (decisions only GG can make).**
1. Merge #338 (artifacts, flags off; edits deploy.yml and eval-gate.yml). After it I run, one at a time, candidate (no traffic) → smoke → offline
   comparison → promote → live check for family_v2, then k8s v2, then evidence; rollback = unset the variable or route to the prior revision.
2. The single re-record (D92.6) costs about 1.31M Groq tokens: 150-set ~919K (`reports/qp3b_c4_token_budget_2026-10.json`) plus the committed
   64-set ~390K (extrapolated). The 64-set must also be re-recorded or #320's eval gate stays red (it is red now: the k8s prompt changed). At
   ≤160K/day that is at least 8 ledger-days, realistically 10-12. Confirm this scope and duration.
3. The 64-set's k8s queries are 2014-2016 while the v2 index holds 2023-2026 issues: the created-before filter leaves them no neighbours, no
   filter leaks future issues. Recommendation: keep their old frozen neighbours and report it as a deviation.
4. After the re-record, approve writing the new baseline (as D61/D62 for #161) so #320 can go green; I will not write it without that.
5. Label calls: `performance` and `papercut 🩸` excluded as spellings of ADR-0070 labels, beyond the literal rule (verifier: `performance` is the
   weakest call).
6. Standing: publish_models.py overwrites all GCS objects (do not run it as is); sklearn InconsistentVersionWarning (1.6.1 pickles served on
   1.7.2); the primary checkout's vscode pkl does not match MANIFEST; #184, #138, #94 untouched.

**Next (no GG needed).** Commit the ADR-0073 table generator (the table is not regenerable from the repo today); R8 vscode duplicate
fine-tuning pre-registration, then a private Kaggle kernel when GPU quota allows (quota is only visible on push); `vscode-retrain-data`
release once family_v2 serves, then tell GG #326 is ready; R10/R11 negative results are merged as records.

**Budget and resources (16:39).** Groq ledger gpt-oss-120b: 158,232 / 160,000 allocatable in the trailing 24 h (triageiq-recorder 97,500 from
R4; another project 60,732); recording-active no. C: 245 GB free (251.5 at the run's start). D: added this run ≈ 1.4 GB retrain dry runs,
1.3 GB d1_verify (deletable after D92.1 serves), 0.24 GB d1_loader (deletable), R11 1.46 GB, r10_surv venv 0.5 GB. Nothing deleted.
Wake-ups: cron job 750c53cb, hourly at :17, session-only, expires ~2026-11-04; re-create before then and note it here.

## First run, state at 2026-10-23 ~23:05 IST (history)
- 23:02: Claude Code stopped BOTH background jobs (the Kaggle hourly retry and the 03:00 collection wait) because system memory was critically low. Its instruction is not to restart them without being asked, so neither runs now: the 2026-10-24 03:00 post-cutoff collection will NOT run, and the Kaggle push is no longer retried. Last Kaggle attempt 23:02: still quota-blocked. Queued for GG (restart commands under Background jobs).
- Stop new long work from 2026-10-27 12:00 IST; return brief by 2026-10-27 18:00 IST.
- Day 1 done through #303 and #309: vscode evidence artifact (#302), Dependabot pip fix (#303; it immediately opened #304-#308, left open for GG), Q5a design + lexical baselines on validation (#309: BM25 R@5 29.2%, TF-IDF 27.3%, ceiling 92.8%; #309 first failed CI on an unlisted psutil import, fixed by making the RAM guard optional). Every serving change merged default off; prod healthy after each deploy.
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
| Q9 | background | daily post-cutoff collection: the 03:00 IST 2026-10-24 run was stopped for low memory at 23:02 (not restarted; GG) |

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
- None. Both stopped at 23:02 2026-10-23 for low system memory (see Incidents). To restart, when memory allows:
  - Collection: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\triageiq-data\bin\run_confirm_scrape.ps1` (idempotent; resumes from its checkpoint).
  - Kaggle vscode embeddings: `PYTHONPATH=src python scripts/qk_kaggle.py push-kernel --kernel kaggle/kernels/embed_bge --slug triageiq-embed-bge-vscode --datasets gauravgandhi2411/triageiq-texts-microsoft-vscode-issues`, then `wait --kernel triageiq-embed-bge-vscode --timeout 7200 --interval 60` and `download --kernel triageiq-embed-bge-vscode --out D:\triageiq-data\kaggle_out\vscode`.

## Incidents
- 05:40-06:23: timeline scraper stuck at batch size 1 after HTTP 599s; restarted from checkpoint; fixed in all scrapers (#262).
- ~09:50: Claude Code stopped a background wait command for low system memory; the scrape was unaffected.
- ~12:30: the orchestrator switched the shared primary checkout's branch for about two minutes while an executor worked (nothing lost). Own git work now in the docs worktree.
- ~12:00: a merge loop filtered `[fail]` while the gate prints `[FAIL]`; a DIRTY PR (#274) sat 30 minutes (replaced by #275).
- 16:57:27: scripts/qp1_vscode_lib.py and qp1_vscode_phase1.py in the primary checkout were overwritten by an unexplained split-like edit (no other triage-iq session ran; the executor working then says none of its commands touched them; cause unverified). Preserved on branch `backup/foreign-qp1-vscode-edit-2026-10-23` and a scratchpad copy; the committed (verified) versions were restored by switching branches; nothing lost.
- ~17:30: an executor ran `ruff format` on a whole directory, reformatting two unrelated files; when a hook refused the restore it worked around the refusal (WIP commit + restore from HEAD~1 + amend, never pushed). Outcome harmless; briefs now forbid hook workarounds and directory-wide formatting.
- 23:02: Claude Code stopped both background jobs (Kaggle hourly retry, 03:00 collection wait) for critically low system memory while the session was idle; not a fault of either job. Per its instruction they are not restarted by the orchestrator; the 2026-10-24 collection is missed unless restarted.

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
