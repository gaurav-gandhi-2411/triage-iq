# Queue for GG (only items you must do)

Last updated 2026-10-16 (round 13, D72-D75). Everything else is in `docs/DECISION_LOG_2026-10.md`.

## Round 13 (2026-10-16): README #151 is READY for your review (not merged)

README #151 (`docs/readme-verified-numbers`, head `3a42c15`) is marked ready. **Do not merge until you have read it; I did not and will not.** This round added the contention result and the `component_match` finding. Text of the sections added or
changed this round, rendered as they appear in the README (the PR itself is the full document; the other sections were number-checked by me in earlier rounds, not reviewed by you):

### Latency, concurrency and caching (README, "Latency" section)

**The 2026-10-11 stage-4 result is a provider rate-limit effect, not a pipeline regression, and it is a real production exposure.** The first call had token headroom and ran in 2.9 s (stage 4). Calls 2–5 each logged `Rate limit hit — sleeping` (attempts 1–3 of 6, Cloud Run logs) and every attempt itself took about 30 s inside the Groq client, so stage 4 took 51–133 s; all five returned HTTP 200 with `_degraded` false. Groq's free-tier limit for this model is **8,000 tokens per minute** (response header `x-ratelimit-limit-tokens: 8000`) and one triage consumes about 6,000 tokens (stored usage: ~4,800 prompt + ~1,100 completion), so a second call inside the same minute, or any other traffic on the same key, exhausts the minute. A single probe call after the replay finished saw 1,909 tokens remaining with a 45.7 s reset, so something else was drawing on the pool. The Groq console later identified it (owner-reported, 2026-10-12): a review-iq benchmark key on the same Groq account made about 2,100 calls on 2026-10-11 (about 340 requests and 300K tokens on this model), TriageIQ only a few dozen and the portfolio chatbot 3, so it was not the chatbot. Stage 3 rose from 52.6 to 93.9 ms; BELIEVED to be the k8s query-embedding PCA transform added in PR #150 (the two vscode calls ran 54–100 ms), not measured separately. Evidence: [`reports/latency_replay_2026-10-12.txt`](reports/latency_replay_2026-10-12.txt). What changed since: a 15-second total budget for the LLM stage shipped (PR #170, ADR-0063 addendum), the benchmark key now waits on a shared per-model budget ledger (`tools/groq_budget`; review-iq PR #510), and the cached-token field is logged (PR #174).

**After the 15 s budget (2026-10-12, revision `triageiq-api-00055-boy`), same five bodies, 75 s apart:** the two calls that had token headroom took **4,870 ms and 5,324 ms** end to end (stage 4: 2,737 and 3,672 ms; no retries, no `Rate limit hit` line). The Groq daily pool then ran out (`Used 196411` of 200,000, TriageIQ's own calls in that window were about 35K, so the rest, about 160K, is BELIEVED to be other keys on the same account that the TriageIQ side cannot see), and calls 3–5 returned HTTP 200 with `_degraded` true, reason `rate_limited_tpd`, in **1,520–1,820 ms** instead of waiting. A daily-limit 429 is not retried; the response is the signals-only plan (component, similar issues, resolution estimate, no LLM text). Evidence: [`reports/latency_replay_2026-10-12_post_a1.txt`](reports/latency_replay_2026-10-12_post_a1.txt).

**Concurrency, stated plainly.** One triage costs about 6,000 of the key's 8,000 tokens per minute, so the free tier serves roughly **one LLM-backed triage per minute**; a burst of simultaneous requests cannot all get an LLM answer. Since PR #170 a request that cannot get one inside 15 s total (retries included) returns the signals-only plan with `_degraded` true and `_llm_status_reason` instead of waiting up to 135 s as on 2026-10-11. VERIFIED: nine unit tests (rate limit degrades within 15 s, a long `Retry-After` degrades immediately, a normal call is unaffected) and the daily-limit path above in production. VERIFIED live, one run (2026-10-16 17:54 UTC, revision `triageiq-api-00061-guj`, three different issues sent 8 s apart; [`reports/contention_test_2026-10-16.json`](reports/contention_test_2026-10-16.json), Cloud Run lines in [`reports/contention_test_2026-10-16_cloudrun_log.txt`](reports/contention_test_2026-10-16_cloudrun_log.txt)): call 1 got an LLM answer in 6.2 s with no retry; call 2 hit a rate limit whose 25 s backoff exceeded the 14.9 s left and **degraded at once, in 1.2 s** (`llm_time_budget_exceeded`); call 3 hit a rate limit, slept 11 s and got an LLM answer in 16.2 s (the 15 s budget bounds the LLM stage, the other stages add about 1.2 s). So 2 of 3 requests inside 32 s received an LLM answer, 1 degraded, there were 2 retries in total and no 5xx. That is one run, so it shows the mechanism works as designed, not a rate; whether prompt caching helped call 3 was not captured.

**Prompt caching is intermittent, not absent.** Groq documents prompt caching for `openai/gpt-oss-120b` ("cached tokens do not count towards your rate limits"). An earlier version of this paragraph said it never engaged; that was read through the SDK. Reading the RAW response JSON of a repeated identical-prefix request, the warm call reported `usage.prompt_tokens_details.cached_tokens` = 1,536 of 1,611 prompt tokens (the cold call carried no such field). Across the direct tests that is about 1 hit in 5 warm calls, and 0 of 4 production triage calls (4.7–4.9K-token prompts) showed the field; one SDK-parsed pair of the same request showed no hit. So caching can happen but cannot be relied on: it is NOT counted in the budget planning above (the 8,000 tokens-per-minute figure is treated as unrelieved), and whether a cached call is charged less against the per-minute counter is untested.

### Note added under the LLM rows: what the grounding numbers count (unchanged this round, for context)

> **What the grounding numbers count, and what they do not.** `fabrication_rate` and the grounding ratchet count plans where `grounding_status.all_grounded` is false: the predicted component is not in the classifier's own top-3, or a cited similar-issue number was not in the retrieved set. That is a **consistency check against the pipeline's upstream outputs, not a correctness check against gold labels**: 20 of the 64 plans predict a component that is wrong versus gold, and all 20 are scored "grounded" because the wrong guess still falls inside the classifier's top-3 (recomputed on the 2026-10-11 cassette with `scripts/measure_grounding.py` and the gold labels in `eval/eval_set.jsonl`; method: ADR-0058 addendum). `0/53` and `0/11` must therefore not be read as error rates. The report-only metric that tracks the LLM overriding the classifier is `component_departed_from_top1_rate` (5 of 64 plans; all 5 wrong vs gold).

### Note added this round: what the judge's component_match measures

> **What the judge's `component_match` measures.** The judge prompt contains the gold component, so `component_match` is a comparison of the plan's component with a supplied label (2 correct, 1 plausible but not gold, 0 wrong), not an inference from the issue text. Checked against exact string equality on the 64 recorded judge calls (`scripts/component_match_agreement.py`, `reports/component_match_agreement_2026-10-16.json`): the judge scored 2 for all 44 plans whose component equals gold, and for 5 of the 20 plans whose component does not (4 scored 0, 11 scored 1); agreement with "equal iff 2" is 59/64 = 92.2%, and every disagreement is the judge being more lenient (all five are kubernetes labels, e.g. gold `cloudprovider` vs predicted `provider/openstack`). Two consequences: the judge's mean `component_match` (1.70 of 2) is higher than strict equality would give (1.38), and the judge's worst band (`component_match` = 0, one of the two inputs to the floor-fail rate) caught 4 of the 20 wrong plans, so the floor-fail rate understates component misses. The count of 20 wrong plans above is the strict-equality number.

How the numbers were checked this round: contention figures from `reports/contention_test_2026-10-16.json` and the Cloud Run lines beside it; component_match figures from
`reports/component_match_agreement_2026-10-16.json` (script `scripts/component_match_agreement.py`, run on the committed cassette); a spot-check of the headline LLM and resolution
numbers (12.09/15 overall, floor-fail 0/11 and 4/53, CQR coverage 79.71% [77.94, 81.38] and 82.7% [78.5, 86.2]) against `reports/eval_baseline.json` and `reports/eval_summary.json`.
Not done: the README refers to a `readme_numbers_check.py` that does not exist in the repo, so no tool re-read every number; the rest of the numbers rest on the verification of earlier rounds.

**Also needs you (blocked by the auto-mode classifier, not retried):** removing 29 merged worktrees (`triage-iq-wt-*`; each was checked first: branch is an ancestor of origin/main, no tracked
changes, no untracked files). The command `git worktree remove C:/Users/gaura/ml-projects/triage-iq-wt-<name>` for these names was denied as "Irreversible Local Destruction": analysis, budget, cached, ckptval, decay,
degrade, docs5, docs6, docs7, docs8, docs9, embfix, emblog, evalb, guard, integ, ladapt, lcore, lock, log, lrec, manifest, sklearn, smoke, startup, startup2, startup3, vscodenaive, m140, p3ro.
Run them yourself (they hold no work), or tell me to retry. They stay on disk until then.

## Round 12: what needs you (D67)

1. **ON HOLD (owner, 2026-10-12): Cloud Run recording job, core #184 and its ONE IAM binding.** Both wait until README #151 is approved. When released:
   merge #184 (touches `.github/workflows/`, so yours), then the binding and the steps after it exactly as in `scripts/cloud_recording/README.md`
   (steps 1-4); after the first watched run check Billing > Reports for Cloud Run (that jobs use the free allowance is BELIEVED, not verified). Nothing
   has been granted or deployed.
2. **Nothing else to merge by you.** Mine and waiting on the clock: triage-iq-ui #29 (Vercel `main`-only deployments; it is also the trigger for
   the UI #28 production build after the Vercel reset at about 06:50 UTC Oct 13), triage-iq-ui #30 (degraded-mode notice; after #29's build is verified),
   the raw-JSON caching check and the 3-in-60 s contention test (they wait for the corrected ledger to reopen, about 20:20 UTC), the vscode eval
   labelling (running, gated by the shared GPU), README #151 (marked ready for your review once the contention result is in; its review is the one step
   that must be yours). Open PRs of mine awaiting CI: #190 (ledger lock flake) and #191 (tracked-cassette write guard).

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
