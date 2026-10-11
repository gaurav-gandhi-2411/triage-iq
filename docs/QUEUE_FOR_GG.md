# Queue for GG (only items you must do)

Last updated 2026-10-11 (round 9). Everything else is in `docs/DECISION_LOG_2026-10.md`.

## The one core merge

1. **Core #161 (integration branch). READY for your merge (baseline approved and written, CI green; see D62).**
   It carries D7 (train-median point, both repos, clamped and flagged), the conformal interval in the prompt,
   CQR v2 serving, degrade-on-429, manifest-in-image, checkpoint validation, the `/eval` data fix, the
   `deploy.yml` allowlist (D49) and HEAD on `/health` (D50). Merging deploys (it changes `src/**`) and the
   smoke test needs Groq tokens (ledger: 66,454 of 160,000 tokens in the last 24 h at 10:09 UTC Oct 11).
   **The single merge command:**
   `gh pr merge 161 --repo gaurav-gandhi-2411/triage-iq --merge`
   Superseded by it (do not merge separately; kept open as the review record): #154, #155, #156, #157, #159.
   Reviewable-vs-generated split (also in the PR body): about 2,478 hand-written non-test lines (source/scripts/
   workflows 1,992 + docs 486), 1,898 test lines, 10,043 recorder output lines (`eval/cassettes/*`), 679
   designated `reports/**` lines. The merge gate fails gates 3 and 4 (size, deploy config) by design: human merge.
   Baseline (approved 2026-10-11, ADR-0064): pooled 12.031 -> 12.094; `resolution_estimate_reasonableness` fell
   1.672 -> 1.578, recorded as an unresolved proxy signal (ground truth favours the median: MAE 3.52 d vs 7.28 d).
   After you merge I poll for the deploy and run the post-deploy checks myself.

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
