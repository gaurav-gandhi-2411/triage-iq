# Queue for GG (only items you must do)

Last updated 2026-10-08 (round 5). Everything else is in `docs/DECISION_LOG_2026-10.md`.

## The one core merge

1. **Core #161 (integration branch). NOT READY: draft, waiting on the 64-entry cassette re-record.**
   It carries D7 (train-median point, both repos, clamped into the interval and flagged), CQR v2 serving,
   degrade-on-429, manifest-in-image, checkpoint validation, the `/eval` data fix, the `deploy.yml` allowlist
   (D49) and HEAD on `/health` (D50). Merging it deploys (it changes `src/**`); the smoke test needs Groq tokens,
   so merge it after the recorder has finished, not before. I will mark it ready and put the final numbers
   (baseline before/after, reviewable-vs-generated split) in its body when CI is green. Expected: the recorder
   finishes about 17:00-19:00 UTC Oct 9, then a local judge pass. When it is ready, the command is:
   `gh pr merge 161 --repo gaurav-gandhi-2411/triage-iq --merge`
   Superseded by it (do not merge separately; kept open as the review record): #154, #155, #156, #157, #159.
   Baseline approval: D1 approved the earlier baseline; D7 changes every prompt, so the re-derived baseline will be
   reported before/after in #161 and needs your OK when you merge (I do not self-approve a new baseline).

## Dashboard / account actions (only you can do these)

2. **Recreate the Vercel deploy hook, delete the old one** (old URL was pasted in chat; hook id `Xafi3lf3AT`,
   name `prod-main`). Project `triage-iq` -> Settings -> Git -> Deploy Hooks. The API has the endpoints, but my
   create call was refused by the permission classifier (a new secret-bearing credential), so it is yours.
   Nothing uses the hook (no workflow, no GitHub secret), so delete-and-recreate breaks nothing; do not paste
   the new URL anywhere saved (D52).
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

## FYI (no action)

- **Groq budget (200K tokens/day, org-wide, rolling 24 h) is exhausted until about 16:30-19:00 UTC Oct 8**
  (Used 198,924 at 01:12 UTC). Production `/triage` degrades to the signals-only plan with a 200 only after
  #161 deploys; until then a 429 is a 500 (existing behaviour). The recorder is gated to spend at most 160K per
  24 h (80 pct), leaving 40K for production and deploy smoke tests. Deploys of main until ~17:00 UTC Oct 8 may
  fail their smoke test for lack of budget: that is the budget, not a regression (see D28).
- UI #27 (badge + `/eval` table) is merged and live; it is inert until #161 deploys.
