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

**D9 (final). Dependency fix merged: #147 -> `10a1784`.** The scoped regen moved exactly four pins
(`multidict 6.7.1->6.9.1`, `fsspec 2026.2.0->2026.6.0`, `datasets 4.8.5->5.0.1`, `anyio 4.13.0->4.14.2`);
nothing else (no sentence-transformers/groq/transformers movement). Incremental diffs: multidict
alone; +datasets moved datasets/fsspec/multidict; +anyio moved only anyio. None of the four is
imported by repo code (`git grep`). `pip-audit` with the nine remaining ignores: "No known
vulnerabilities found, 10 ignored". Three obsolete ignores deleted (PYSEC-2026-3716, CVE-2026-63374,
CVE-2026-64847): this only tightens the gate. CI (PR run 37623521925): "Lock in sync" success,
"Security audit" success, "No known vulnerabilities found, 12 ignored" (CI counts differently).
Alternative rejected: adding ignores for the two new advisories (they were fixable).
Deploy of `10a1784`: run 37624456668 success (candidate smoke test passed, promoted).

**D10. #146 merged: removed the permanently red informational pip-audit job.** It audited a freshly
resolved environment, not the lock, with none of the documented suppressions, so it was red by
construction (alternative: make it audit the lock with the same ignores = duplicate of the blocking
step). It was never required (protection requires `test`, `Structural invariants (no LLM)`,
`Quality regression (cassette-replayed judge)`). Backlog row marked done.

**D11. Core #144 merged -> `3b0ed94`** after `gh pr update-branch` (strict protection) and green
required checks. Gate: the UI that reads the new shape (#23) was already CONFIRMED deployed
(D6), so the order was safe. Deploy run 37626771577 success; revision `triageiq-api-00036-xiy`.
VERIFIED live: `/eval/summary` calibration keys now `classifier, ece_*_definition, n_test, n_eval_set,
previous_classifier, ...`, T_opt 1.1388 / 4.8636, ece_test 0.0307 / 0.0431, ece_eval_set 0.1875 /
0.121, judge n=64, no `triageiq-api-0...` strings. Live `/eval` hard-reloaded: renders, no console
messages, no undefined/NaN, "Current baseline: 11.94/15 ... n=64". Screenshot:
scratchpad `eval_live_after_144_2026-10-07.png`.

**D12. Vercel Ignored Build Step: UI #24 merged -> `1072c70`.** `vercel.json` `ignoreCommand`
builds only for `main`, `preview/*` and ref-less (CLI/hook) builds. Logic tested locally for 5
cases. Skip path VERIFIED live: the PR branch got "Vercel: success - Canceled by Ignored Build
Step" (no build consumed, no required check affected: protection requires only `lint-and-build`).
Build path VERIFIED live: production deployment `dpl_CS7htPUwRJ2aeoqksbwvtHem3Zjd` READY and
the project's current production, sha `1072c70`. Convention for a preview: name the branch `preview/<x>`.

**D13. Stuck automation browser.** The browser tool's own Chrome (profile `chrome-devtools-mcp`)
held its profile and blocked new pages; I stopped only that process tree (root pid 19896, after
listing it: all processes were the automation profile, none were your normal Chrome).

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

## Phase 5 decisions: #140 and UI #21 (2026-10-07)

**D14. UI #21 (pre-warm) closed, not merged.** It was written when cold starts were 6-17/day;
with UptimeRobot keep-warm there was 1 on 10-06 and 0 on 10-07, so a per-page-view `/health`
pre-warm has nothing to warm. The warm-up message/150 s timeout only matter if the monitor stops
(the existing "Waking up service" label and Cloud Run's 300 s request timeout already bound that
case). Branch kept; reopen if cold starts return. Alternative rejected: merge it anyway for the
timeout (a UI deploy and extra requests for a scenario that no longer occurs).

**D15. Core #140 (startup timing logs + tiktoken baked into the image) still adds value; queued
for GG, not merged.** Value: per-step startup timings never reached Cloud Run before (verified in
the platform logs: only uvicorn lines), and the first `/triage` on a fresh instance downloads
tiktoken's encoding at request time. Reviewed the Dockerfile/app.py diff (82 lines): logging +
a build-time cache, no model-path change; agent verified outputs byte-identical (sha256 767bff7d..).
Required checks green. The merge guard fails ONLY gate 1: branch `perf/cold-start-startup` is not
in its prefix list (feat/fix/chore/docs/refactor/investigate/test/wave-N). Not renamed (rule 35:
renaming to fit is rule-gaming). Output of `python C:/Users/gaura/.claude/scripts/merge_gate.py
-PrNumber 140 -Repo gaurav-gandhi-2411/triage-iq`: gate 1 FAIL; gates 2, 2b, 3, 4 PASS.

**D16. UI #26 (Eval contract test) queued, not merged.** Size gate ambiguous (see queue item 5).

