# spec-autonomous-run.md — TriageIQ unattended run until 2026-10-27 (approved by GG)

GG is unavailable from the kickoff until **2026-10-27**. Claude Code works continuously on the
quality push (`spec-quality-push.md`) with no human in the loop. This document defines how.

## 1. Precedence (when instructions conflict, the higher one wins)

1. Safety and the NEVER list (section 4)
2. Pre-registered targets and acceptance checks in `spec-quality-push.md`
3. This runbook
4. `docs/HANDOFF_2026-10-20_quality_push_pause.md` and the decision log

## 2. Operating loop

Repeat until the stop time (section 9):

1. **Read state.** Open `docs/RUN_STATUS.md` (create it on the first run). It is the single source of
   truth: current task, queue, done, blocked (with unblock conditions), background jobs, decisions.
2. **Check health** (at least every 6 hours; section 7).
3. **Re-check blocked items.** Any whose unblock condition is now met returns to the queue.
4. **Pick the next unblocked task** from the queue (section 5), in order.
5. **Do it** through one executor, then one verifier. Strictly sequential: one subagent at a time,
   at most one worktree. Background jobs (scrapes, Kaggle kernels) may run while a subagent works.
6. **Record.** Update `RUN_STATUS.md` after every task, decision, block or background-job change.
   Commit it on a docs branch and merge it at least every 12 hours so state survives a crash.
7. **While waiting** on a background job, never idle: take the next unblocked task. Use scheduled
   wake-ups to poll jobs. If nothing is unblocked, poll every 60 minutes.

Context compaction or a restarted session must lose nothing: always re-read `RUN_STATUS.md` first.

## 3. Decision rule — decide alone, or defer and move on

**Decide autonomously** when ALL hold:
- reversible (a revert or re-run undoes it);
- within the approved spec and its pre-registered targets;
- based on measured evidence, with a verifier pass;
- recorded as an ADR or decision-log entry with the alternatives considered and why they lost.

**Defer** (log in `docs/QUEUE_FOR_GG.md`, mark the task BLOCKED, move to the next task) when ANY hold:
- it costs money or needs a purchase, upgrade or new billable service;
- it changes IAM, Secret Manager, or any credential;
- it publishes or merges public-facing text (README, repo description, UI copy, posts);
- it enables a production feature flag or otherwise changes live product behavior;
- it deletes data or artifacts (scratch files and merged worktrees excepted);
- it would change a pre-registered target, metric or acceptance check;
- a permission classifier or hook refused it.

**Ambiguity:** prefer the option that keeps the live product unchanged and the data intact, record
the reasoning, and continue. A decision taken alone is always reported in the return brief.

## 4. NEVER list

- Spend money; buy credits/units; enable billable GCP services (no BigQuery).
- Train or run CPU/GPU-heavy jobs on the laptop. The laptop shut down from heat before. Local jobs
  must be light and under 10 minutes; bulk embedding, training and inference run on Kaggle.
- Use Colab (it needs an interactive browser; not usable unattended).
- Touch the Groq production reserve or bypass the shared ledger.
- Re-record cassettes except the single Phase 5 re-record (section 5, Q8).
- Enable any production feature flag. New behavior merges default-off only.
- Edit `.github/workflows/deploy.yml`, IAM, secrets, or the hook/merge-gate configuration.
- Use skip-permission/confirmation flags, `git stash`, `--force` on destructive git commands.
- Retry a refused action in another form. Shell-form blocks may be redone with Edit/Write tools;
  content or policy blocks may not.
- Peek at a test split before freezing a method. Merge experiments to main to measure them.
- Ask GG to label data.

## 5. Work queue (ordered; each item lists its dependency and its fallback)

**Q1 — Timeline fix (blocks label-based work).** Direct per-issue timeline scraper; run k8s, then
vscode; quantify event loss vs the search-based scrape; re-derive both repos; recompute every
label-timeline report; re-check k8s Phase 1 gold. If gold changes, re-run k8s Phase 1 as a new gold
version and report old vs new side by side.
Fallback if the API is down or rate-limited: move to Q5b, Q6 or Q9 and retry hourly.

**Q2 — vscode taxonomy (needs Q1 vscode).** 31% of 2026 test labels are classes unseen in training.
Evaluate with evidence: (a) a more recent training window and its cost to validation/test size;
(b) open-set handling — detect unknown components and abstain, scored as a miss unless abstention
is a declared, counted outcome; (c) mapping to stable parent areas; (d) a scheduled-retraining design
for production. Freeze the choice in ADR-0070 before any vscode model is scored. If the evidence shows
top-3 ≥ 95% is infeasible for vscode, say so and report the best achievable.

**Q3 — vscode splits, ceilings, 150-issue LLM eval set (needs Q2).**

**Q4 — Phase 1 on Kaggle.** vscode classifier candidates per `spec-quality-push.md` §6, plus
embedding-based candidates for both repos (BGE head, kNN neighbor vote, fine-tuned encoder with
ONNX int8 export, stacked ensemble), conformal sets chosen on validation only.
Fallback if Kaggle GPU quota is exhausted: CPU kernels for TF-IDF/kNN/stacking; otherwise Q5b/Q6.

**Q5 — Phase 2 retrieval.**
- Q5a vscode duplicates (7,339 pairs): refreshed index, BGE fine-tune with hard negatives on Kaggle,
  domain reranker, latency within budget. Target duplicate Recall@5 ≥ 80%.
- Q5b k8s prior-PR task (ADR-0068): scrape PR files-changed (light, resumable), freeze the task, score.

**Q6 — Phase 4 resolution (survival framing).** Feature build and models on Kaggle CPU kernels or
light local work only.

**Q7 — Phase 3b redesign, implementation only.** Component from the conformal set, priority model,
grounded next steps from resolution evidence, kNN resolution; all behind default-off flags; zero-token
offline evaluation where possible. Finalize the prompt.

**Q8 — Phase 5.** Objective metrics harness first. The single cassette re-record may start ONLY if Q7
is accepted against its pre-registered checks; it runs unattended under the ledger and may continue
past the stop time.

**Q9 — Continuous background track (always available):**
- collect post-cutoff k8s issues daily for the conformal confirmation set (ADR-0069); run the frozen
  method once when the set reaches the size defined there;
- fix the failing dependabot pip job if it needs no workflow-file change, else queue it;
- remove merged worktrees and scratch output; keep the disk footprint current.

## 6. Quality controls (unchanged by being unattended)

- Executor then verifier on every result; numbers recomputed independently before merge.
- Pre-register methods and metrics before scoring; never on test before freezing.
- Lock-built venv for anything CI runs (local pandas differs from CI).
- Gate-sized PRs; split oversized ones; merge when required checks are green and the gate allows.
- A merge-hook "not eligible" contradicted by a manual gate run: wait 10 minutes, retry the identical
  command once.
- New data-integrity problems found mid-task: stop dependent work, fix the root cause, re-verify.

## 7. Health checks and incident handling

Every 6 hours: production `/health?deps=1`, latest deploy result, main CI, open alert issues,
GitHub API rate limit, Kaggle quota, disk free on C: and D:.

- **Production down caused by a recent merge:** open a revert PR, merge it if checks pass, record it.
- **Production down for any other reason** (Groq key, billing, GCP): do not touch credentials or
  billing; record evidence in the queue; continue other work. Degraded mode is expected when Groq
  quota runs out and is not an incident.
- **CI broken by a new advisory:** scoped lock regen per the lock-sync process; suppressions only with
  written reachability analysis.
- **Network loss:** retry with backoff; switch to offline tasks.
- **Claude usage limit reached:** state is in `RUN_STATUS.md`; continue automatically after reset.
- **Session ends or crashes:** nothing is lost if `RUN_STATUS.md` is current.

## 8. Reporting

- `docs/RUN_STATUS.md`: live state, updated after every task.
- Daily entry in the decision log: what finished, decisions taken alone, blocks, numbers with CIs,
  disk footprint.
- Labels: VERIFIED or BELIEVED on every claim.

## 9. Stop time and return brief

- From **2026-10-27 12:00 IST**: start no new long job (no new Kaggle run longer than 2 hours, no new
  scrape). Finish the current task if it completes within the window.
- By **2026-10-27 18:00 IST**: write `docs/RETURN_BRIEF_2026-10-27.md` and merge it. Contents, in order:
  1. Headline numbers per target vs pre-registered targets, with CIs.
  2. Decisions taken alone, each with evidence and the ADR link.
  3. Items queued for GG, most important first, each with exact steps.
  4. Blocked items and their unblock conditions.
  5. Running background jobs and how to check them.
  6. Incidents and how they were handled.
  7. Disk footprint and the next recommended task.
- Then stop starting new work; keep only already-running background jobs.
