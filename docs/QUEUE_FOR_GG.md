# Queue for GG (only items you must do)

Last updated 2026-10-11 (round 8). Everything else is in `docs/DECISION_LOG_2026-10.md`.

## The one core merge

1. **Core #161 (integration branch). DRAFT, waiting on ONE thing: your baseline approval (item 1a).**
   Re-record and judge pass are done and committed (D61). CI on #161: `test` and `Structural invariants` green;
   `Quality regression` red on exactly one test, `test_cassette_hash_matches_baseline`, because the baseline still
   holds the old cassette hash. It carries D7 (train-median point, both repos, clamped and flagged), the conformal
   interval in the prompt, CQR v2 serving, degrade-on-429, manifest-in-image, checkpoint validation, the `/eval`
   data fix, the `deploy.yml` allowlist (D49) and HEAD on `/health` (D50). Merging deploys (it changes `src/**`)
   and the smoke test needs Groq tokens (ledger: 66,454 of 160,000 tokens spent in the last 24 h at 10:09 UTC Oct 11).
   Merge command, once 1a is done and CI is green and the PR is marked ready:
   `gh pr merge 161 --repo gaurav-gandhi-2411/triage-iq --merge`
   Superseded by it (do not merge separately; kept open as the review record): #154, #155, #156, #157, #159.
   Reviewable-vs-generated split (also in the PR body): about 2,478 hand-written non-test lines (source/scripts/
   workflows 1,992 + docs 486), 1,898 test lines, 10,043 recorder output lines (`eval/cassettes/*`), 679
   designated `reports/**` lines. The merge gate fails gates 3 and 4 (size, deploy config) by design: human merge.

   **1a. Baseline for approval (I did NOT write it).** Judged before/after, same judge model and rubric, paired
   by issue (ADR-0064 "Re-record result"; `python eval/run_eval.py`, cassette sha256 e9f8861f85fe...):

   | Scope | Before | After | Delta |
   |---|---|---|---|
   | microsoft/vscode (n=11) | 12.273 | 12.727 | +0.455 (band 0.45, improvement) |
   | kubernetes/kubernetes (n=53) | 11.981 | 11.962 | -0.019 (band 0.22) |
   | Pooled (n=64) | 12.031 | 12.094 | +0.063 |

   Per dimension (pooled): component_match +0.031, similar_issues_relevance +0.031, **resolution_estimate_
   reasonableness 1.672 -> 1.578 (-0.094): it FELL** (k8s -0.113, 5 up / 11 down; vscode unchanged), priority_
   alignment +0.031, next_steps_actionability +0.016, overall_quality +0.047. None is distinguishable from judge
   noise at this n. Confounds: median point, System 3 label and calibrated prompt interval changed together, and
   synthesis was re-sampled. To approve: tell me, or run `python eval/run_eval.py --update-baseline` on the #161
   branch and commit `reports/eval_baseline.json`; then CI goes green. If you would rather not accept the
   reasonableness drop, the alternative is not to ship D7 for k8s (decision 7-9 section).

## Dashboard / account actions (only you can do these)

3. **Set shorter Vercel deployment retention** (re-checked 2026-10-08: the API still rejects the field, 400): projects `gaurav-gandhi` and `samidha-reviews-web`,
   previews 7 d, canceled 3 d, errored 3 d, production 30 d / keep 10.
4. **Check the Vercel Usage page** for Functions Storage (<10 GB?). I cannot read it.
5. ~~UptimeRobot GET~~ Done in code: `/health` answers HEAD (D50, in #161). Nothing for you to do; I remove this
   line once #161 is deployed and the monitor is seen to pass.
6. **README PR #151 (draft)**: review and merge yourself. Full text:
   https://github.com/gaurav-gandhi-2411/triage-iq/blob/docs/readme-verified-numbers/README.md
   Its two marked blocks and the resolution limitation are refreshed by me after #161 deploys (D7 wording).

## Decisions for you (recommendation first; none blocks #161)

7. **vscode point source.** The median is the 2015-16 training median (3.84 d) while current vscode traffic
   resolves in hours (test median 0.049 d). Recommendation: a recency-window median (D7 says "the training-window
   median the published naive baseline used", so I kept that). Say the word and it is a one-line change plus a
   re-record.
8. **Keep the LightGBM models for intervals/bucket** (P5, D43): recommendation yes, do not simplify to model-free
   intervals. No action needed unless you disagree.
9. **Retrieval-conditional k8s resolution redesign** (P5 study, earlier): recommendation not now.

## After closing #161 (queued work, not for #161; D60)

Groq budget-layer workstream, in order, each its own PR. Nothing here blocks #161.

1. **Caching measurement.** (a) Capture `usage.prompt_tokens_details.cached_tokens` per call in
   `_groq_completion` and store it with each cassette entry and ledger record. (b) At the next TPD 429, compare
   Groq's "Used" with the ledger's total and its non-cached total and record which matches. (Rate-limit headers are
   per-minute only; the daily figure exists only in the 429 body.)
2. Make the ledger count only what Groq counts (follows from 1).
3. Burst scheduling to keep the prompt cache warm (expiry is 2 h without use).
4. Re-measure whether few-shot is still needed now that the schema defect is fixed (ADR-0053 evidence was confounded).
5. Scheduled Cloud Run recording job off the laptop (cloud change, needs your go-ahead).
6. Move the gg-portfolio chatbot to a different Groq model so it stops sharing the gpt-oss-120b pool (other repo,
   your decision).

## FYI (no action)

- **Groq budget (200K tokens/day, org-wide, rolling 24 h) is exhausted until about 16:30-19:00 UTC Oct 8**
  (Used 198,924 at 01:12 UTC). Production `/triage` degrades to the signals-only plan with a 200 only after
  #161 deploys; until then a 429 is a 500 (existing behaviour). The recorder is gated to spend at most 160K per
  24 h (80 pct), leaving 40K for production and deploy smoke tests. Deploys of main until ~17:00 UTC Oct 8 may
  fail their smoke test for lack of budget: that is the budget, not a regression (see D28).
- UI #27 (badge + `/eval` table) is merged and live; it is inert until #161 deploys.
