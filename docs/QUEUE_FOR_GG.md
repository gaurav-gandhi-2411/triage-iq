# Queue for GG (only items you must do)

Last updated 2026-10-07. Newest first within each urgency. Everything else is in
`docs/DECISION_LOG_2026-10.md`.

## Do soon

0. **Review the README proposal (draft PR #151, NOT merged by me).** Full proposed text is the README.md on the PR
   branch: https://github.com/gaurav-gandhi-2411/triage-iq/blob/docs/readme-verified-numbers/README.md
   (kept there, not copied into this file, so it cannot go stale; 70 claims checked: 21 match, 5
   partly, 44 were stale or wrong, 43 corrected; mismatch table in the PR body).
   It contains new numbers, so it is yours to approve. Highlights to look at: the evaluation table
   (as-served resolution numbers; vscode point estimate worse than naive, stated plainly), the
   "Grounding is a consistency check, not a correctness check" note (19 of 20 wrong-vs-gold plans pass),
   the Training data table (29,994 = k8s retrieval index size, not training data; classifiers saw
   4,226 + 6,710 labeled issues), Latency (n=5), Monitoring (UptimeRobot + GitHub monitor; Cloud
   Monitoring not adopted). Two blocks (`RESOLUTION-ROWS`, `LLM-BASELINE-BLOCK`) must be refreshed
   after #150 lands and the k8s cassette is re-recorded: merge #151 after that.
   `gh pr merge 151 --repo gaurav-gandhi-2411/triage-iq --merge`

1. **Recreate the Vercel deploy hook and delete the old one.** The old URL was pasted in chat.
   Vercel dashboard -> project `triage-iq` -> Settings -> Git -> Deploy Hooks: delete the hook
   (`...Xafi3lf3AT`), create a new one for branch `main`. Do not paste the new URL anywhere that
   is saved. Why: anyone with the URL can trigger production builds and burn the deploy quota.

2. **Set shorter deployment retention in the Vercel dashboard** (the API cannot). For projects
   `gaurav-gandhi` and `samidha-reviews-web`: Settings -> (Security / Deployment Retention):
   previews 7 days, canceled 3 days, errored 3 days; leave production at 30 days and "keep 10".
   Why: both projects were at 30 days, so previews pile up and each READY Next.js deployment
   stores a function bundle (the likely cause of "Exceeded free resources").

3. **Check the Vercel Usage page** (https://vercel.com/gaurav-gandhi-2411s-projects/~/usage):
   is "Functions Storage" under 10 GB now? I could not read it (not logged in). The production
   deploy of the UI succeeded after I removed 7 old previews, but I cannot show the cleanup caused it.

5. **Merge UI #26** (Eval page contract test: renders the Eval page against core main's JSON in
   fixture and live modes; negative controls show it fails on the shape change that blanked `/eval`
   on 2026-10-05). All checks green. I did not merge it: the size gate is ambiguous under rule 70a
   (about 231 reviewable lines, plus 317 lines of verbatim core data fixtures outside a designated
   path, plus 1,002 generated lock lines), and it edits `.github/workflows/ci.yml`, moving CI from
   Node 20 to 22 because vitest 5 / jsdom 30 need it. Review the Node bump, then:
   `gh pr merge 26 --repo gaurav-gandhi-2411/triage-iq-ui --squash`
   The merge triggers one Vercel production build: confirm the deployment is READY and current.

6. **Merge core #140** (startup timing logs reach Cloud Run; tiktoken encoding baked into the image).
   Required checks green, bit-identical outputs verified, diff reviewed (82 lines). The merge guard
   blocks it only on gate 1: the branch is `perf/cold-start-startup` and `perf/` is not in the
   recognized prefix list. I did not rename the branch (that is rule-gaming the gate). Merge:
   `gh pr merge 140 --repo gaurav-gandhi-2411/triage-iq --merge`
   Optional follow-up for your global config: add `perf/` to the prefixes in
   `C:\Users\gaura\.claude\scripts\merge_gate.py` (also affects #137-style branches).

7. **Decide vscode's resolution point estimate.** Served model is -54% MAE vs naive (6.02 d reported
   as -70.5% with embeddings; 5.45 d as served); no variant robustly beats naive (study in
   DECISION_LOG D17). Options: (a) leave as is (README already discloses it, UI shows "Model below naive
   baseline"); (b) serve the naive median (3.84 d) as the point estimate, like the bucket already does
   (needs an interval design + a cassette re-record of the 11 vscode issues). My recommendation: (b)
   when you next touch vscode; not urgent.

8. **Publish a recalibrated CQR artifact for k8s (optional, low value).** After the embedding fix,
   fresh Q = -1.02 h vs stored +0.2835 h; held-out coverage 83.8% -> 79.7% [77.9, 81.4]. Publishing needs a
   new object name in `gs://triageiq-prod-260812-models/models/` plus a MANIFEST/loader change (I did not
   overwrite the existing artifact: the bucket has no versioning).

9. **Merge core #150: READY** (k8s resolution embeddings at serving; ADR-0062; cassette re-recorded;
   all 3 required checks green, PR CLEAN). Approve the new judge baseline, then
   `gh pr merge 150 --repo gaurav-gandhi-2411/triage-iq --merge`
   Baseline before -> after (details in the PR comment and the ADR-0062 addendum): k8s 11.8679 ->
   11.9811 (+0.113, inside the +/-0.22 band, not claimed as an improvement), vscode 12.2727 unchanged,
   overall 11.9375 -> 12.0312, k8s fabrication 1/53 -> 0/53, floor-fail 3/53 -> 4/53. Judge drift
   control 6/6 identical. The merge guard blocks it only on gate 3 (8,874 "reviewable" lines, ~6,800
   of them generated cassette JSON that cannot be split from the code). The merge triggers a
   deploy (smoke-gated). After it deploys: README PR #151 gets its refresh (two marked blocks) and a
   small follow-up fixes the /eval resolution table.

## Low urgency

4. UptimeRobot: switch the monitor's HTTP method to GET. It sends HEAD first and `/health`
   answers 405, so every check logs two requests. Harmless, just noise.
