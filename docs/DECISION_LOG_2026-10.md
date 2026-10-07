# Decision log, October 2026 (autonomous round, 2026-10-07)

Every decision, the evidence, and alternatives rejected. Newest entries at the bottom.
Labels: VERIFIED (evidence given) or BELIEVED.

## Phase 0-2: unblock Vercel, restore /eval (2026-10-07)

**D1. #143 was already merged by GG** (`0d27739`). No action.

**D2. Functions Storage cannot be measured with the access I have.** VERIFIED: (a) Vercel's REST
API has no usage/size endpoint (`vercel api list` shows none; `/v1/billing/charges` returns 404 on
Hobby; deployment detail has empty `lambdas[].output`; `/v6/deployments/{id}/files` returns 404);
(b) the dashboard Usage page needs a login and the automation browser is not logged in (it shows
the Vercel login page; credentials were not entered). Consequence: the cleanup was sized by
rule, not by bytes, and success is judged by whether a build is accepted.

**D3. Docs.** VERIFIED (https://vercel.com/docs/plans/hobby): "In most cases, if you exceed your
usage limits on the Hobby plan, you will have to wait until 30 days have passed before you can
use the feature again." "Functions Storage" is not in the Hobby included-usage table, so how it
is counted is not documented there. https://vercel.com/docs/limits: Hobby 100 deployments per
86,400 s, owner-scoped. Only 30 deployments were seen in 24 h, so the daily cap does not explain
the rejections.

**D4. Deleted 7 old preview deployments, nothing else.** Rule: READY previews of the two
function-bearing projects (`gaurav-gandhi`=gg-portfolio, `samidha-reviews-web`=review-iq) whose
branch has no OPEN PR, plus two 9-day-old gg-portfolio previews. Never touched: any production
deployment, any current production deployment, `triage-iq` (the UI is a static Vite SPA with no
function bundles, so its previews hold no Functions Storage). Alternatives rejected: deleting
previews of branches with open PRs (other sessions were actively working there: new previews at
12:05-12:07 UTC today); deleting CANCELED/ERROR previews (they hold no bundle, no benefit).

| Project | Deployment | Branch | Result |
|---|---|---|---|
| gaurav-gandhi | dpl_9woVUMkNTKcCQkpAQ7UgyLDEoNff | dependabot/npm_and_yarn/minor-and-patch-... | removed (2nd try; 1st was my own bad path) |
| gaurav-gandhi | dpl_HtKeK6ffLzyoFwjTQ63JSzTZPYAc | fix/a10-llm | removed |
| gaurav-gandhi | dpl_BHvn5k7UrN3ZVkhJyiHbdnH311ZQ | dependabot/github_actions/... | removed |
| gaurav-gandhi | dpl_DV2hf2nUVdPvST2SmLqoSLeuSDiC | chore/content-pipeline | removed |
| samidha-reviews-web | dpl_AotCATzvZdrcG76UMPT19CFxiBpK | fix/s17-vercel-ignore-full-diff | removed |
| samidha-reviews-web | dpl_J5nGx1tGVwL1o11FVPEMzqP1ZWTR | fix/s17-remove-authenticity-surface | removed |
| samidha-reviews-web | dpl_EwYtQ7z5X8fkJ3cEo89UbagyPHJG | fix/s16-heldout-exposure | removed |

Before/after totals in bytes: UNKNOWN (see D2). Deployment counts: 65 -> 58 across the team.

**D5. Root cause of accumulation (BELIEVED, from the inventory).** Retention is 30 days for
previews/production/canceled/errored with 10 deployments always kept (read from the project
settings). review-iq produced 26 deployments in two days (14 of them CANCELED by superseding
pushes) and gg-portfolio 14; every READY Next.js deployment stores its function bundle. Fix: shorter
retention for previews/canceled/errored. VERIFIED that the REST API cannot set it (`PATCH
/v9/projects/{id}` with `deploymentExpiration` returns 400 "should NOT have additional property");
so it is a dashboard action, queued for GG.

**D6. Fired the deploy hook once** (12:11:47 UTC, HTTP 201, job state PENDING). It was accepted
and built: production deployment `dpl_J8a5xQ1KGVkbLPjffM8uRSYDkTSw`, sha `c129645`, READY at
12:12:31 UTC. VERIFIED current: project `targets.production.id` equals that ID, aliases include
`triage-iq-orcin.vercel.app`; live bundle changed from `index-CRaV2Kgm.js` to `index-ooLP4duU.js`
(Last-Modified 12:12:42 GMT). Not retried; no retry was needed.
Whether the 7 deletions caused the build to be accepted is UNKNOWN: other projects' builds were
already being accepted that morning, and a preview was accepted on Oct 5 12:20 UTC while the
production build before and after was rejected. No causal claim is made.

**D7. /eval verified on the live site** (hard reload, cache ignored): renders, zero console
messages, "Current baseline: 11.94/15 mean over n=64 gold-set issues (79.6%)", per-repo 12.27
(n=11) and 11.87 (n=53). Screenshot: scratchpad `eval_live_2026-10-07.png`. The calibration table
still shows the retired classifier's figures (T_opt 0.2981/0.3234) until core #144 deploys (D8).
Main triage page end to end: real keyboard input, one `/triage` call, plan rendered (component
`debug` 93%, 4 next steps, 3 similar issues), Under the Hood shows "Groq openai/gpt-oss-120b",
baseline line "qwen3:8b, n=64: 11.94/15 (79.6%)", no console messages.

**D8. Core #144 is blocked by two NEW advisories, published after Oct 5.** The branch was behind
main (strict protection), so it was updated; `test` then failed in "Security audit":
multidict 6.7.1 CVE-2026-104874 (fix 6.9.1) and fsspec 2026.2.0 CVE-2026-104851 (fix 2026.6.0).
Same failure affects every PR on main. Handled by a separate dependency PR (below), not by
editing #144 and not by adding ignores.

**D9. Dependency fix.** Scoped regen `--upgrade-package=multidict --upgrade-package=fsspec` in
python:3.11-slim moved only `multidict 6.7.1 -> 6.9.1`; fsspec stayed (BELIEVED: capped by
`datasets 4.8.5`). Trying a regen that also upgrades `datasets` (declared but never imported in
this repo) to clear fsspec and the old PYSEC-2026-3716 suppression. Result recorded below.

## Phase 5: keep-warm
**D10. UptimeRobot's first ping: 2026-10-05T12:39:50Z** (VERIFIED from Cloud Run request logs,
UA `Mozilla/5.0 (compatible; UptimeRobot/2.0)`). It sends HEAD first (405 from the API) then GET
(200), so each check is 2 requests; ~270 checks/day.

| Day (UTC) | Cold starts | Billable instance-s | vCPU-s |
|---|---|---|---|
| 09-30 | 11 | 515 | 2,299 |
| 10-01 | 9 | 500 | 2,282 |
| 10-02 | 7 | 385 | 1,687 |
| 10-03 | 8 | 309 | 1,154 |
| 10-04 | 6 | 261 | 1,191 |
| 10-05 (monitor started 12:40; deploys + testing that day) | 15 | 1,590 | 4,753 |
| 10-06 (first full day with keep-warm) | 1 | 108 | 336 |
| 10-07 (to 12:21 UTC) | 0 | 45 | 89 |

Keep-warm works and usage fell, not rose. (Script: `coldstarts_per_day.py`; "cold start" = one
uvicorn "Started server process" line. The 48-hour window closes 12:40 UTC 2026-10-07.)
