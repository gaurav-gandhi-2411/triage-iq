# Queue for GG (only items you must do)

Last updated 2026-10-08. Everything else is in `docs/DECISION_LOG_2026-10.md`.
Done since the last version (removed from this list): #150 merged and deployed (revision
`triageiq-api-00042-ves`, health OK); #153 (was #140) and UI #26 merged; baseline approved (D1).

## Merge commands (guard-blocked or human-only; I did not merge these)

1. **Core #154** — checkpoint validation, coupling guard test, ADR-0062 coupling section, release-checklist
   item 6. All 3 required checks green. Guard gate 3 fails on size (~1,118 reviewable lines).
   `gh pr merge 154 --repo gaurav-gandhi-2411/triage-iq --merge`
2. **Core #156** — provider errors (429, connection, timeout, 5xx) degrade to the signals-only plan instead of
   HTTP 500; the deploy smoke test still requires `_degraded is False`. Touches `deploy.yml` (guard gate 4), so
   human merge. Branch needs `gh pr update-branch 156` first if it shows BEHIND.
   `gh pr merge 156 --repo gaurav-gandhi-2411/triage-iq --merge`
3. **Core #157** — Dockerfile.prod never copied `MANIFEST.sha256`, so the runtime drift check has been a silent
   no-op in production (found in the 00042-ves startup log). One COPY line plus a guard test. Deploy config
   (gate 4), so human merge. After the deploy, startup logs should contain no `ARTIFACT_DRIFT` line.
   `gh pr merge 157 --repo gaurav-gandhi-2411/triage-iq --merge`
4. **Core #155 (draft)** — vscode serves the naive median (D2). NOT ready: its cassette needs the 11 vscode
   entries re-recorded, which needs Groq budget (see FYI). I will push the re-record and queue the final command.
5. **README PR #151 (draft)** — review and merge yourself. Full text: the README.md on
   https://github.com/gaurav-gandhi-2411/triage-iq/blob/docs/readme-verified-numbers/README.md
   Two marked blocks (`RESOLUTION-ROWS`, `LLM-BASELINE-BLOCK`) are refreshed by me after the served-path metrics land.

## Dashboard / account actions (only you can do these)

6. **Recreate the Vercel deploy hook, delete the old one** (old URL was pasted in chat; ends `...Xafi3lf3AT`).
   Project `triage-iq` -> Settings -> Git -> Deploy Hooks. Do not paste the new URL anywhere saved.
7. **Set shorter Vercel deployment retention** (API cannot): projects `gaurav-gandhi` and `samidha-reviews-web`,
   previews 7 d, canceled 3 d, errored 3 d, production 30 d / keep 10.
8. **Check the Vercel Usage page** for Functions Storage (<10 GB?). I cannot read it. No causal claim is made
   about the preview deletion (D6).
9. UptimeRobot: switch the monitor to GET (it sends HEAD, `/health` answers 405, two requests per check). Noise only.

## Decisions for you (recommendation first)

10. **vscode point estimate source.** D2 stands (naive median). Recommendation: also consider a recency-window
    median instead of the stale train median (P5 study finding); not urgent, no action taken.
11. **Second-model fallback on Groq 429** (a separate-budget model such as gpt-oss-20b). Proposed in the ADR for
    #156, not implemented. Recommendation: yes, because a single org-wide free-tier budget is also the serving budget.
12. **Retrieval-conditional k8s resolution redesign** (P5 study). Report only; recommendation: not now.

## FYI (no action)

- **Groq daily budget (200k tokens/day, org-wide) was exhausted by my 53-call k8s re-record** (D28). Rolling
  window frees from ~16:30 UTC Oct 8, fully by ~19:00 UTC. Until then production `/triage` may return 500 on 429
  (fixed by #156 once merged and deployed). I will not spend Groq tokens before then; the vscode re-record
  (~45k tokens) and production latency replay (~25k) are scheduled after the budget probe shows headroom.
