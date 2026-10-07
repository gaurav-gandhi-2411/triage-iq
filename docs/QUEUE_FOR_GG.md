# Queue for GG (only items you must do)

Last updated 2026-10-07. Newest first within each urgency. Everything else is in
`docs/DECISION_LOG_2026-10.md`.

## Do soon

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

## Low urgency

4. UptimeRobot: switch the monitor's HTTP method to GET. It sends HEAD first and `/health`
   answers 405, so every check logs two requests. Harmless, just noise.
