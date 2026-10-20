# Quality push: paused on 2026-10-20, resume after 2026-10-23

Spec: `spec-quality-push.md` (approved 2026-10-19). Decisions: `docs/DECISION_LOG_2026-10.md` D78-D83. ADRs: 0066 (retrieval targets,
item 3 superseded), 0067 (k8s taxonomy sig/*), 0068 (k8s prior-PR retrieval task), 0069 (conformal frozen, exploratory).

## State at the pause (all verified by `git`/process listing)
- Nothing is running. The last background job (vscode comments re-scrape, 9,185 issues, 94,351 comments) finished before the
  pause; the executor agent was stopped before it wrote any code (checkout is clean on a branch equal to main).
- main CI is green; open PRs are only #184, #138, #94 (GG's).
- Data lives outside the repo on D: (`D:\triageiq-data`: raw, derived, splits, models, logs, bin). ~441 GB used on D:, C: has ~224 GB free.

## Merged to main (#220-#247)
Scraper (+schema-2 closer repo), derive tables/events/stats, fix-link audit (k8s 98.3%), comment-limit audit (k8s), PR-files scraper,
duplicate-target scraper, comments-only scraper, taxonomy coverage, splits (k8s), ceiling analyses (k8s), Phase 1 harness + TF-IDF + conformal
(k8s sig/*), Phase 3a flag (OFF), prior-PR counts + ADR-0068.

## UNMERGED branches (held on purpose; no PR yet)
- `feat/qp0-label-replay-fix`: creation-label replay rule + re-derive with dup_targets. Code is sound but its PREMISE ("a final label with
  no labeled event was set at creation") was contradicted by a verifier (see below). Do not merge until the timelines are re-scraped.
- `feat/qp0-vscode-audits`: vscode fix-link audit (96.7% [91.7, 98.7], pass), underpowered comment-limit audit, taxonomy coverage and the
  vscode taxonomy freeze (100 classes). Reports computed on unreliable timelines are superseded; the class family also leaks val/test counts.
- `feat/qp0-scrape-timelines`: just created, empty.

## THE BLOCKING FINDING (data quality, 2026-10-20)
GraphQL `search` with nested `timelineItems` silently drops events. A verifier measured ~14% of label events missing (98 of 114 on a 40-issue
control; #250473 had totalCount 0 in the raw file vs 2 live; 27 of 30 "no-event" labels have a real labeled event 55 h-7,500 h after creation
via a direct `issue(number:N)` query). Everything derived from embedded timelines is suspect: `labels_t72h`, primary label (earliest labeled
event), churn, first-label timing, and possibly closer/cross-reference/duplicate events (recall; the fix-link and duplicate PRECISION audits
used independent REST checks and stand). k8s sig/* labelled issues had 0 "no-event" cases, but k8s event completeness is UNMEASURED.

## Resume order (do in this order)
1. Build `scripts/qp0_scrape_timelines.py`: aliased `issue(number:N){ timelineItems(first:100, itemTypes:[same list as the main scraper], ...)}`
   batches of 10, same fields as the main scraper's query, completeness (`len(nodes)==totalCount`), storage `raw/<slug>/timelines/`, plus an
   offline `--compare` mode writing `reports/qp0_timeline_completeness_<slug>_2026-10.json` (embedded vs direct events by type, share of
   issues missing label events, share whose FIRST label event changes). Model it on `qp0_scrape_dup_targets.py`. Run k8s first (~770 batches), then vscode
   (~11k batches, a few hours of API budget). Copy the scripts to `D:\triageiq-data\bin` and run detached with a retry loop (see
   `run_*.ps1` there); scrapers resume from their checkpoints.
2. Make derive read direct timelines in preference to embedded ones; re-derive k8s and vscode; recompute every label-timeline report. For k8s
   check whether the Phase 1 gold (primary label) changes; if it does, the Phase 1 numbers must be recomputed and reported as a new gold version
   (test-once discipline: say that the old numbers were computed on the incomplete timelines).
3. Fix the vscode class family to depend on TRAIN counts only (classes = non-excluded labels with >= 30 train-period issues; the V_new
   >= 100-all-years threshold leaked val/test counts), then write ADR-0070 (vscode taxonomy, coverage and routing semantics only; vscode
   labelled coverage is only 28%, and 2026 labelled primaries are ~36% in classes with >= 30 train issues, 31% in labels never seen in train).
4. Amend ADR-0067 (primary-label rule after re-scrape) and ADR-0068 (answerable counts will rise with complete events), then merge the held branches.
5. vscode splits (needs final duplicate pairs), vscode ceilings, the 150-issue stratified eval set (75 per repo).
6. Phase 1 vscode, embedding candidates (BGE head, kNN) on Kaggle (token at `C:\Users\gaura\.kaggle\kaggle.json`; no training on the laptop), Phase 2 (vscode duplicates, k8s prior-PR task), 3b, 4, 5 (single re-record), 6 (README, GG merges).
7. Conformal confirmation (ADR-0069): keep scraping new k8s/vscode issues; run the frozen method once at >= 500 labelled issues aged >= 30 days.

## Process notes
- Test every new script in a venv built from `requirements.lock` (CI uses pandas 3.0.2; local pandas hid three bugs): venv at
  `%TEMP%\claude\triage-iq\20261020\ci_venv` (recreate if missing).
- Gate monitor must grep exactly `OVERALL: eligible`; PRs go BEHIND after each merge (`gh pr update-branch`).
- Executor/verifier subagents, one at a time; verifier deletes of scratch dirs are sometimes denied (never retried): see QUEUE_FOR_GG.md.
- Memory file with the same content: `.claude/projects/.../memory/project_quality_push.md`.
