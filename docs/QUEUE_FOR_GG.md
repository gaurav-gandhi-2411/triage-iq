# Queue for GG (only items you must do)

Last updated 2026-10-12 (round 11). Everything else is in `docs/DECISION_LOG_2026-10.md`.

## Core #161: MERGED and deployed (2026-10-11, merge f71096a, revision triageiq-api-00045-zac)

Verified live (D63): `/health?deps=1` ok, HEAD /health 200, `/eval/summary` serves the new baseline and the
MANIFEST-hash provenance. **One production exposure found, not fixed (needs your pick, item A below):**
the Groq 8,000 tokens/minute limit makes /triage take 50-135 s when the key's per-minute pool is used up.

Round 11 status (D65): core #170 is MERGED (f7b118c) but its deploy failed the smoke test on `rate_limited_tpd` (the shared
120b daily pool is exhausted); production still serves 00045-zac. I re-run the deploy when the pool recovers and merge #171, #172
and UI #28 in that order, each after the previous deploy is confirmed. Nothing for you to merge.

A2 is built as gg-portfolio draft PR #308 (chatbot, canary, curator, framer on gpt-oss-20b; verifier stays on qwen). **Needs you:**
re-record the chatbot cassettes on 20b (the key is in your portfolio env file; the secret-file guard blocked me from loading it):
run `node --env-file=<your portfolio env file> evals/chatbot/run-eval.mjs --live --force` from the PR #308 branch, commit the new
cassettes plus the report, and compare with the 120b baseline in the PR body (recall@5 95.0%, groundedness 100%, refusal
precision 100%, false-refusal 0%). Until then 20b answer quality is UNMEASURED; do not merge #308 before that.

## Dashboard / account actions (only you can do these)

3. **Set shorter Vercel deployment retention** (re-checked 2026-10-08: the API still rejects the field, 400): projects `gaurav-gandhi` and `samidha-reviews-web`,
   previews 7 d, canceled 3 d, errored 3 d, production 30 d / keep 10.
4. **Check the Vercel Usage page** for Functions Storage (<10 GB?). I cannot read it.
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
