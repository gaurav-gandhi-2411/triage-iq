# Queue for GG (only items you must do)

Last updated 2026-10-12 (round 11, D66). Everything else is in `docs/DECISION_LOG_2026-10.md`.

## Core #161: MERGED and deployed (2026-10-11, merge f71096a, revision triageiq-api-00045-zac)

Verified live (D63): `/health?deps=1` ok, HEAD /health 200, `/eval/summary` serves the new baseline and the
MANIFEST-hash provenance. **One production exposure found, not fixed (needs your pick, item A below):**
the Groq 8,000 tokens/minute limit makes /triage take 50-135 s when the key's per-minute pool is used up.

Round 11 status (D66, 2026-10-12). Done and deployed: core #170, #171, #172, #174 and your #175 (production now on a revision with the 15 s LLM budget, the `emb_*` log field and the cached-token field); the shared Groq budget ledger is merged as #178, #179 and #180 (the oversized #177 is closed as superseded); gg-portfolio #308 is closed (the chatbot is not the pool consumer; branch `fix/chatbot-groq-gpt-oss-20b` kept, no cassette re-record needed). Waiting, all mine: the UI #28 production build (Vercel is rate limited until about 07:30 UTC Oct 13, see D66 item 4), the pool-dependent measurements (A3 cached-token study, 5-body replay, 3-in-60 s contention test: need about 50K tokens of 120b headroom, the pool was empty at 08:24 UTC), review-iq PR #510 (benchmark budget gate; merged by me when its guard allows).

**1. README #151 (draft): your review is the one step that must be yours.** Full text: https://github.com/gaurav-gandhi-2411/triage-iq/blob/docs/readme-verified-numbers/README.md. I keep it a draft; its latency section is refreshed from the measurements when the pool allows.

**2. OPTIONAL cleanup, not a blocker: give review-iq's benchmark its own Groq org.** Today `GROQ_API_KEY_BENCHMARK` is on the shared Personal account (the Groq console shows it used about 2,100 calls on Oct 11), so it draws on the same per-model daily pools as production; the ledger (#510) makes it wait its turn, an own-org key removes the sharing. Steps, each in your own window (keep the key out of Claude Code):
   1. Sign in to https://console.groq.com as Samidha's own Groq account (her own org), then API Keys, Create API Key, name it `review-iq-benchmark`, copy it.
   2. Open `C:\Users\gaura\ml-projects\review-iq\benchmark\vernacular_v2\.env.benchmark.local` in an editor and replace the value on the `GROQ_API_KEY_BENCHMARK=` line with the new key. Save.
   3. From `C:\Users\gaura\ml-projects\review-iq` run `.venv\Scripts\python.exe benchmark\vernacular_v2\benchmark_groq_key.py`; it must print "OK" (it shows the key's first 8 and last 4 characters, do not paste them anywhere).
   4. Because the new org has its own pools, tell the budget gate to step aside for that org: set the user environment variable `GROQ_BUDGET_SEPARATE_ORG` to `1` (PowerShell: `setx GROQ_BUDGET_SEPARATE_ORG 1`; new terminals only). Without it the ledger would count the new org's tokens against the shared pool.
   5. Run one small benchmark script and check the Groq console of Samidha's org shows the usage (and the Personal account's `gpt-oss-120b` usage stays flat).
   6. Delete the old key: https://console.groq.com on the Personal account, API Keys, delete the key behind the old `GROQ_API_KEY_BENCHMARK` value.
   Nothing breaks if you skip this; it only removes the shared-pool exposure.

**3. OPTIONAL, Vercel (D66 item 4): stop canceled preview deployments from counting toward the Hobby cap.** All three projects already skip builds with an Ignored Build Step (triage-iq now too, set by me through the API: build only `main` and `preview/*` branches), but a skipped deployment is still CREATED, and Vercel Hobby allows 100 created per day per account. `review-iq`'s `web/vercel.json` could add `"git": {"deploymentEnabled": {"main": true, "preview/*": true, "*": false}}` so other branches never create a deployment (BELIEVED to be the stronger fix, not tested). I did not change review-iq's `vercel.json`; say the word and I will PR it.

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
