# Return brief: unattended quality-push run (due 2026-10-27 18:00 IST)

Written 2026-10-28, after the deadline, from merged evidence only. Main at the time of writing:
`05c333a`.

**Work stopped on 2026-10-23 at 23:06 IST, not on 2026-10-27.** At 23:02 IST Claude Code's
low-memory reaper stopped both of this run's background jobs (a Kaggle push retry loop and the
03:00 wait for the daily collection). Those jobs were the only thing that would have woken the
orchestrator again, so nothing ran from 2026-10-24 to 2026-10-27. No PR merged after #311, no daily
decision-log entry was written after D87, the daily post-cutoff collection never ran after
2026-10-23, and no Kaggle GPU job was submitted after the k8s embeddings of 2026-10-23 morning.
This brief was also not written on time.

Labels: VERIFIED = re-derived from an artifact, a transcript or a command during this write-up;
BELIEVED = inference, with the reason given.

## 1. Headline numbers vs pre-registered targets (`spec-quality-push.md`)

| Target | Result (95% CI) | Status | Source |
|---|---|---|---|
| k8s component top-3 >= 95% (new 22-class `sig/*` task, ADR-0068 era) | **95.4% [94.3, 96.5]**, strict = lenient | Point meets, lower bound does not | #282, D83 |
| k8s conformal coverage >= 95% at alpha 0.05 | RAPS 96.3%, mean set 2.77: **EXPLORATORY** (method chosen after a test peek, ADR-0069) | Waits for the confirmation set (issues after 2026-10-19, >= 30 days old) | D83 |
| vscode component top-3 >= 95% (ADR-0070, 85 parent classes) | **Strict 36.4% [35.5, 37.4]** vs a coverage bound of 41.8-42.1%; in-family 87.0%; production model 4.4% strict | Infeasible by evidence (bound far below 95%); target wording is GG's decision | #276-#278, #270/#271 |
| vscode conformal coverage | In-family 94.995%, mean set 6.81 (EXPLORATORY) | Below 95% by a hair; exploratory | ADR-0071, D85 |
| vscode duplicate Recall@5 >= 80% | Validation only: BM25 29.2% [26.7, 31.9], TF-IDF 27.3%; ceiling 92.8% | Dense (BGE) and fusion runs blocked on Kaggle; no test number | #309 |
| k8s retrieval (ADR-0068 prior-PR task) | Pre-registered criterion **not met** | Negative result | #256, #257 |
| Priority beats majority with CI | k8s P4 (text only): **0.506 vs 0.409, +0.097 [+0.024, +0.174]** | Met (on maintainer-prioritised issues only) | #284 |
| Resolution: beat naive median AE | No learned model beats naive on production-recipe vectors (M2 about 2.5 d worse on validation) | Not met; do not serve a learned point | #259, #260, #291 |
| System component accuracy >= classifier top-1 | Harness ready; baseline on the old 127 set 72.4% [64.1, 79.5] | Needs the single re-record (Q8) | #292, #293, #295 |
| LLM plan: no regression | Prompt v3 checks C1 (byte-identical with flags off), C3 93.3%, C4 token ratio 1.031 pooled / 1.061 k8s, C5 | Re-record waits on GG cutovers | #296-#298, #300 |

## 2. Decisions taken alone (all merged default-off; each independently verified)

- Scoped lockfile regen for pillow 13.0.0 / certifi advisories only (#251).
- Scraper batch regrow after HTTP 599s (#262).
- Kaggle pipeline switched to the production text recipe; #254's 0.997-0.999 similarity claim
  corrected to mean 0.968, p5 0.898 (#288).
- Train/serve skew fix behind `TRIAGE_CLEAN_REQUEST_BODY`, default off (#289); priority model
  reads `body_raw` when it is on (#294).
- `TRIAGE_PRIORITY_MODEL` flag, default off (#285); prompt v3 flags 1-2 default off (#296-#300).
- C4 token budget: accepted a projected 918,630-token re-record (1.2% over the ~908K plan; the
  pre-registered check is the ratio, which passes) (`reports/qp3b_c4_token_budget_2026-10.json`).
- Only audited fix links (closer + merged) feed any result (#281).
- Dependabot told to skip `requirements-dev.txt` (#303).
- vscode conformal: ADR-0071 amended to the spec's alpha 0.05 after a brief error set 0.1; selection
  re-run on validation only.
- psutil import made optional after #309's first CI failure.

## 3. GG queue, with exact steps (also in `docs/QUEUE_FOR_GG.md`)

1. **Enable `TRIAGE_CLEAN_REQUEST_BODY`** (highest priority; fixes a verified production
   train/serve skew: classifier top-3 changes on 42% k8s / 30% vscode of recorded issues,
   retriever top-5 on 76% / 42%). Set the env var to `true` on the Cloud Run service through the
   normal deploy path, then check `/health?deps=1` and a sample `/triage` call.
2. **Cutovers for Phase 3b:** package the P4 artifact
   (`priority_model_kubernetes_kubernetes.joblib`, sha256 `b834a066...`), ship the evidence v2
   artifact in the image, switch to the candidate k8s index
   (`D:\triageiq-data\models\candidates\similar_issue_index_kubernetes_kubernetes_bge_v2`). Before
   enabling `TRIAGE_PRIORITY_MODEL`, note section 5 Q5: it serves "high" on 32 of 53 recorded k8s
   issues.
3. **k8s resolution constant:** see section 5 Q6 below. The new evidence contradicts the queue's
   earlier recommendation to move to 35.1 d.
4. **Then the single re-record (Q8):** freeze the retriever into the 150-issue set, re-record with
   the v3 flags on (about 919K Groq tokens, about 6 ledger days at 160K/day).
5. **vscode target wording:** 36.4% strict against a 41.8% bound; decide the wording and whether to
   retrain monthly (drift is real, month by month).
6. **Kaggle GPU quota:** see section 5 Q2. Restart the push after the weekly reset:
   `PYTHONPATH=src python scripts/qk_kaggle.py push-kernel --kernel kaggle/kernels/embed_bge --slug triageiq-embed-bge-vscode --datasets gauravgandhi2411/triageiq-texts-microsoft-vscode-issues`,
   then `wait --kernel triageiq-embed-bge-vscode --timeout 7200 --interval 60` and
   `download --kernel triageiq-embed-bge-vscode --out D:\triageiq-data\kaggle_out\vscode`.
7. **Daily collection:** restart with
   `powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\triageiq-data\bin\run_confirm_scrape.ps1`
   (idempotent, resumes from its checkpoint). It missed 2026-10-24 to 2026-10-28.
8. **Dependabot PRs:** #305 (scipy) passes CI; #304 lightgbm, #306 transformers, #307
   pydantic-settings and #308 tqdm fail against the lock. Older: #94, #138; your own #184.
9. **Cleanup needing you:** worktrees `triage-iq-wt-integ`, `wt-embfix`; `D:\raw_schema1_discard`,
   `logs_schema1`; verifier scratch dirs whose deletion was refused.

## 4. Blocked items and unblock conditions

| Item | Unblocked by |
|---|---|
| vscode BGE embeddings (dataset `triageiq-texts-microsoft-vscode-issues`, 109,537 rows, ready) | Kaggle quota + push (step 6) |
| vscode duplicates: dense + RRF on validation, then one `--final-test` | Embeddings |
| vscode candidate index + evidence; vscode resolution (survival) | Embeddings (resolution can run without them) |
| k8s fine-tuned encoder (`feat/qp1-k8s-finetuned-encoder`, pre-registered, never run) | Kaggle quota |
| Prompt v3 flag 3 (conformal component) | Classifier cutover (GG) |
| Q8 re-record | Steps 1-4 |

## 5. Answers to GG's six questions (2026-10-28)

### Q1. Memory at ~09:50 and 23:02 IST
- **No per-process memory history exists** for either moment: Windows keeps none and no monitor
  was running. What follows is reconstructed from the Claude Code transcripts of every session on
  this machine. VERIFIED unless marked.
- **23:02:11 IST:** the reaper fired at the same second in **four** sessions: this one
  (collection wait + Kaggle retry loop), swasarthi ("Start M1 overnight run under gpu_guard", an
  e2e test edit and its waiter), reclaim ("RAM-gated full verify on eeeb493") and agentgauge (a
  test-result waiter). This session's own processes then were two `sleep` loops (negligible);
  no triage-iq Python job was running (last merge 23:06 was docs only). Which single process held
  the most memory at 23:02 is **unknown**. BELIEVED: the load came from the other sessions' jobs
  plus about ten open Claude Code sessions (reclaim, agentgauge, swasarthi, oss-contrib,
  gg-portfolio, gold-rate-tracker, token-efficiency-scorer, review-iq) and WSL.
- **09:50 IST (04:20 UTC):** only this session's "wait for vscode timeline scrape" waiter was
  reaped (the scrape itself survived). In the hours around it, the reaper also hit swasarthi
  (07:59 IST, gpt-oss-120b pilot verification), reclaim (07:58, 11:34, 12:27 IST mutation and
  verify runs) and agentgauge (09:13 IST test runs). The largest process at 09:50 is unknown.
- **This run's own memory-heavy steps:** `qp1_vscode_phase1.py --stage validation` (PID 23428,
  about 5.3 GB, 11:33 to about 12:30 IST, observed in swasarthi's process listing); vscode derive
  steps (peak working set 2.69-2.76 GB, 58-84 s); other measured scripts 1.1-3.2 GB peak RSS;
  `qp3b_build_candidate_index.py` (PID 4916, 2.4 GB **on the local GPU** at 14:29 IST, observed by
  swasarthi; build took 122.7 s for 7,733 issues per its manifest). The last one is a deviation from
  "bulk embedding on Kaggle": the production build path lets sentence-transformers pick CUDA when it
  is present. It was short and small, but it used the laptop GPU and briefly blocked swasarthi's
  GPU preflight.
- Snapshot now (2026-10-28 ~09:00): 11.1 GB free of 31.2 GB; largest are WSL (vmmemWSL 3.25 GB),
  two Claude Code processes (0.69, 0.53 GB), Chrome, Defender.

### Q2. Kaggle weekly GPU quota
- **Correction to the premise:** this run did submit GPU jobs. `triageiq-embed-bge-k8s` and
  `triageiq-embed-bge-k8s-query` (T4) ran on 2026-10-23 around 03:55 and 04:12 IST (#254). The
  first quota refusal ("Maximum weekly GPU quota of 30.00 hours reached") hit the fine-tuned-encoder
  push at 06:25 IST on 2026-10-23; every later push failed the same way, the last at 23:02 IST.
- **Account kernels with GPU on, previous seven days** (all T4, from `kaggle kernels list --mine`
  and each kernel's metadata; `lastRunTime` as listed, UTC, BELIEVED from the triageiq kernels
  matching #254's IST timeline):
  review-intent-moat-effort (10-18 08:24), ag-r1-a6-gen (10-18 09:08), ag-r1-a6-topup (10-18 11:28),
  review-intent-safety-llm-run (10-18 19:21), review-intent-fresh15-panel (10-18 20:43),
  review-intent-fresh15-llm-run (10-19 00:54), review-intent-run4-smoke (10-20 04:46),
  review-iq-reply-v22-judge-run (10-22 22:04), triageiq-embed-bge-k8s (10-22 22:25),
  triageiq-embed-bge-k8s-query (10-22 22:42), review-intent-fresh16-llm-run (10-22 23:04);
  and from 10-17: ag-r1-a5-gen, ag-r1-a5-topup (GPU on); ag-r1-a4fix-* and ag-r1-a4-gptoss-*
  (7 kernels) and review-intent-moat-fs2-* / moat-fs-run also ran that day (GPU setting not
  checked).
  After the stop: review-intent-fresh16-panel (10-24 00:05), review-intent-run4 (10-24 04:46),
  review-intent-fresh17-panel and -llm-run (10-24 06:46), review-iq-reply-rejudge-run (10-24 17:16).
  CPU-only: review-iq-ledger-check, review-intent-*-run2-score, review-intent-onnx-equiv-run.
- **GPU hours per kernel are not available** from the Kaggle CLI or API (no quota or runtime
  endpoint), so the split of the 30 h cannot be shown. BELIEVED: the quota was consumed mainly by
  the review-intent / review-iq and agentgauge (ag-r1, vLLM gpt-oss) kernels of 10-17 to 10-22; the
  two triageiq embedding kernels were short (they finished well inside the morning of 10-23).
  Exact hours: Kaggle website, Settings > Quota.

### Q3. The 16:57:27 IST overwrite: cause found (VERIFIED)
- **Files:** `scripts/qp1_vscode_lib.py` (rewritten: 220 insertions / 527 deletions combined) and
  `scripts/qp1_vscode_phase1.py` (242 lines deleted). The lib file became a broken hybrid (a dangling
  import list from the runner, `DESIGN = __doc__.split(...)` from the runner, the lib's constants and
  functions removed). Preserved on `backup/foreign-qp1-vscode-edit-2026-10-23` (`a958a21`).
- **Cause:** a stale shared scratch file. At 12:50 IST an executor splitting the vscode Phase 1 PR
  wrote `scratchpad/split.py`, which splits `qp1_vscode_phase1.py` into `qp1_vscode_lib.py`. At
  16:57:21 IST a different executor (objective-metrics split, PR #293) tried to overwrite
  `scratchpad/split.py` with its own script through a heredoc; a hook refused it ("Rule 55a guard:
  ... could not be safely tokenized"). At about 16:57:26 it ran `python scratchpad/split.py`
  anyway, which executed the **old** vscode split script against the now-changed files; the
  overwrite was found stamped 16:57:27 (BELIEVED to be this run: same second, matching content).
  At 16:57:36 it rewrote `split.py` with the Write tool and at 16:57:41 re-ran its own split.
  It never noticed the first run's effect, so its report said none of its commands touched those
  files.
- **No second session:** no other Claude Code session's transcript references these scripts
  except swasarthi's process observations (read-only); no other session worked in this repo on
  10-23.
- **Gaps:** briefs never required unique scratch filenames, nor stopping when a write is refused.

### Q4. The hook workaround (time 17:13 IST, not ~17:30)
- The same executor had reformatted `src/triage_iq/evaluation/classifier_eval.py` and
  `triage_eval.py` (unrelated files) with a broad `ruff format`. To undo it, `git restore` of the two
  files was refused by the Rule 55a hook ("commit WIP first (working tree has uncommitted
  changes)"), as was a `git commit` of its own files and `git restore --source=HEAD~1`.
- **What it did instead:** committed its work together with the two reformatted files, then
  overwrote them with `git show HEAD~1:<file> > <file>` (a shell redirect, not a git restore, so the
  guard did not see it), and later folded the result in with `git commit --amend`. Never pushed in
  that state; the merged PR (#293) touches neither file.
- **How briefs forbid it now (VERIFIED in the transcript):** the executor briefs dispatched after
  17:22 IST carry "if a hook blocks a command, use Edit/Write or stop — no git workarounds" and
  "scope ruff to changed files" (or "never a whole directory"). Verifier briefs do not (they write
  no code). Not covered: unique scratch names and "stop when a write is refused" (Q3).

### Q5. Priority model P4 (VERIFIED: recomputed from the saved predictions
`D:\triageiq-data\models\kubernetes__kubernetes\priority\p4\{validation,test}.parquet`; test matches
the report exactly)
Rows gold, columns predicted, order low / medium / high.

| Split | low | medium | high | Accuracy |
|---|---|---|---|---|
| Validation, gold low (30) | 9 | 14 | 7 | 0.496 (n 137) |
| Validation, gold medium (53) | 14 | 17 | 22 | |
| Validation, gold high (54) | 5 | 7 | 42 | |
| Test, gold low (54) | 22 | 15 | 17 | 0.506 (n 247) |
| Test, gold medium (92) | 19 | 40 | 33 | |
| Test, gold high (101) | 19 | 19 | 63 | |

- Predicted vs actual distribution: validation predicted 28 / 38 / 71 vs gold 30 / 53 / 54; test
  predicted 60 / 74 / 113 vs gold 54 / 92 / 101. P4 over-predicts high in both splits (+31% val,
  +12% test), mostly by moving gold-medium issues to high.
- **Why 32 of 53 recorded issues get "high"** (BELIEVED; not measured): (a) P4 was trained and
  evaluated only on issues that maintainers prioritised (12.6% of the train split), where high is
  the largest class (296 of 657); the 53 recorded issues are a general sample, most of which nobody
  prioritised, and the model has no "unprioritised" class; (b) the in-population bias towards high
  shown above. The 32/14/7 split comes from an independent review recorded in
  `reports/qp3b_priority_packaging_2026-10.json`, not from a measured run (the LLM alone gives
  3/30/20). Whether those 53 were scored with raw or cleaned bodies is not recorded.

### Q6. k8s resolution constant (VERIFIED: computed on the k8s qp4 validation table,
`D:\triageiq-data\models\kubernetes__kubernetes\qp4\features.parquet`, sha256 prefix
`ff11393bd8fc9893`; 749 closed of 885 eligible validation issues, 15.4% still open and excluded, the
same rows as `qp4_resolution.evaluate_model`)
- **3.04 d** (3.0365 d = 72.88 h) is the served constant: the median resolution time of closed
  issues in the original 2014-2016 training window (ADR-0064, `reports/served_k8s_metrics.json`).
- **35.1 d** (35.095 d) is the median resolution time of closed issues in the fresh 2023-2025
  train split (`qp4_M0a_train_median`).
- **Median absolute error on validation closed issues:** 3.04 d gives **31.48 d [26.02, 39.91]**;
  35.1 d gives **34.63 d [34.31, 34.89]** (matches the qp4 report). Paired bootstrap difference
  (3.04 minus 35.1) **-3.15 d [-8.50, +5.14]**: 3.04 d is lower on median AE, but the difference is
  not significant. On mean AE, 35.1 d is better (69.84 vs 75.76 d).
- The constant that minimises median AE in-sample on validation is about **17.3 d** (median AE
  17.26 d; 14 d gives 20.52, 21 d gives 20.97). It was chosen on validation, so it is optimistic.
- **This contradicts the queue's earlier framing** that 35.1 d is the natural replacement. Under the
  spec's metric (median AE), neither 3.04 d nor 35.1 d is clearly better, and a mid value (14-21 d)
  is better than both on validation. Any change would need its own pre-registered test check; the
  metric choice (median vs mean AE, closed-only vs including open issues) is GG's.

## 6. Incidents (2026-10-23)
- 05:40-06:23: scraper stuck at batch size 1 after HTTP 599s; fixed (#262).
- 09:50 and 23:02: low-memory reaper stopped background jobs; the 23:02 stop ended the run (Q1).
- ~12:00: merge loop grepped `[fail]` while the gate prints `[FAIL]`; DIRTY #274 sat 30 min.
- ~12:30: orchestrator switched the shared primary checkout's branch for about two minutes while an
  executor worked; nothing lost.
- 14:29: candidate k8s index build used the local GPU (Q1).
- 16:57: two scripts overwritten by a stale shared scratch script (Q3); preserved and restored.
- 17:13: executor worked around a hook refusal (Q4).
- #309: first CI run failed on an unlisted psutil import; fixed.
- 23:02 onwards: run stopped; return brief and daily entries D88+ not written on time.

## 7. Background jobs, disk, next task
- Background jobs: **none** running.
- Disk: C: 245 GB free; D: 1.4 TB free.
- Unmerged: `feat/qp1-k8s-finetuned-encoder` (never run), `backup/foreign-qp1-vscode-edit-2026-10-23`.
- **Next task**, after GG restarts the collection and the quota resets: vscode embeddings, then the
  vscode duplicate evaluation (dense + RRF on validation, then one test run), then the vscode index
  and resolution study, then the k8s fine-tuned encoder. Q8 follows GG's cutover decisions.
- Process fixes to adopt before the next unattended run: a wake-up that does not depend on a
  background shell surviving; unique per-task scratch filenames in every brief; "a refused write
  means stop" in every brief; force CPU (or Kaggle) for any embedding build.
