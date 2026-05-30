# triage-iq — GitHub Delete+Recreate Checklist

## Pre-push state (after filter-repo)
- Commits rewritten: 71
- Claude trailers stripped: 65
- Dependabot trailers preserved: 10
- Legacy emails remapped: 0 (gaurav.gandhi2411@gmail.com was already canonical)
- .claude/ in .gitignore: yes (line 11)
- Backup: C:\Users\gaura\backup-triage-iq-20260517-182153

## Step 1 — Delete the GitHub repo
URL: https://github.com/gaurav-gandhi-2411/triage-iq/settings
Scroll to "Danger Zone" → Delete this repository

## Step 2 — Recreate the GitHub repo
URL: https://github.com/new
- Name: triage-iq
- Visibility: Public (match original)
- Do NOT initialize with README, .gitignore, or license

## Step 3 — Tell CC to push
Reply: `continue triage-iq`

## Step 4 — Post-push manual steps
- [ ] Set default branch to `main` (Settings → Branches)
- [ ] Recreate any GitHub Actions secrets/variables (CLOUD_RUN_*, WIF_*, METRICS_TOKEN, CORS_ALLOWED_ORIGINS)
- [ ] Recreate any GitHub Releases / tags if applicable
- [ ] Re-enable Dependabot (Settings → Security → Dependabot) — it will recreate its branches automatically
- [ ] Reconnect Vercel if applicable
