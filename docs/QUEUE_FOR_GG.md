# Queue for GG (only items you must do)

Last updated 2026-10-12 (round 11). Everything else is in `docs/DECISION_LOG_2026-10.md`.

## Core #161: MERGED and deployed (2026-10-11, merge f71096a, revision triageiq-api-00045-zac)

Verified live (D63): `/health?deps=1` ok, HEAD /health 200, `/eval/summary` serves the new baseline and the
MANIFEST-hash provenance. **One production exposure found, not fixed (needs your pick, item A below):**
the Groq 8,000 tokens/minute limit makes /triage take 50-135 s when the key's per-minute pool is used up.

A/B were approved ("GO", 2026-10-12) and are built; merges are BLOCKED BY A GUARD HOOK (D64), so they are yours.
Run each command alone, in this order (the gate said `eligible` for #170 on head de6f70c; CI green on #170/#171, #172 and
UI #28 pending when this was written; check `gh pr checks N` first):

1. `gh pr merge 170 --repo gaurav-gandhi-2411/triage-iq --merge`   (A1: 15 s LLM time budget; hook timed out, D64 item 2)
2. `gh pr merge 171 --repo gaurav-gandhi-2411/triage-iq --merge`   (non-zero `emb_*` count in the request log)
3. `gh pr merge 172 --repo gaurav-gandhi-2411/triage-iq --merge`   (B core: served vscode coverage 82.7%, named splits, tail share)
4. `gh pr merge 28 --repo gaurav-gandhi-2411/triage-iq-ui --merge` (B UI: empty header, ADR-0064 wording; Vercel builds it)

Each core merge deploys (smoke test, ~6K Groq tokens). After 1, I open A3 (cached-token capture, branch
`feat/groq-cached-tokens`, already pushed) as its own PR. After the deploys I do the live page check, the contention test and
the README latency numbers.

A2 (gg-portfolio chatbot to gpt-oss-20b) is NOT done: no network to read Groq's current docs for per-model limits
(`console.groq.com/docs/rate-limits` unreachable), so the "limits are per model" premise is unverified, and the pricing row and
the eval re-record (cassettes are gpt-oss-120b recordings; run-eval flags that as MODEL DRIFT) depend on it. One decision for
you: the content pipeline's verifier stage uses `qwen/qwen3.8-27b` (a different family on purpose, for cross-family
verification). It is not on the 120b pool, so I plan to leave it and move only curator/framer, unless you want the verifier on
gpt-oss-20b too.

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
