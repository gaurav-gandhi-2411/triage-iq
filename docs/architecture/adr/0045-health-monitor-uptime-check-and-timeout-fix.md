# ADR-0045: Health-monitor cron drift — add a GCP uptime check as primary, fix a self-inflicted false-failure bug in the GitHub Actions secondary

Status: Accepted
Date: 2026-08-10

## Context

`health-monitor.yml` was built after the 2026-08-05 billing-account outage went undetected for
up to 12 days (ADR-0038) — nothing was watching `/health` on a schedule. The reported symptom
investigated here: the workflow fires every 1.5-3h despite a `*/30 * * * *` cron, not every
30 minutes.

**Confirmed against real data, not assumed.** Pulled the last 100 real run timestamps
(`gh run list --workflow=health-monitor.yml`): gaps between consecutive runs range 34-135
minutes, median well above the nominal 30. This matches GitHub's documented behavior —
scheduled workflows on lower-activity repos are best-effort and can be delayed under platform
load, with no SLA. Editing the cron expression cannot fix this; it's a scheduler-level
property of GitHub Actions itself, not a bug in this repo's YAML.

**A second, more urgent problem, found while pulling those same 100 runs: 99 of them failed.**
Every failure was the identical error: `curl: (28) Operation timed out after 20002 milliseconds
with 0 bytes received`. Checked ground truth directly — the service is not down; a live
`/health` call returned `200 {"status":"ok",...}` in 36.1s. `deploy.yml` runs the service at
`--min-instances 0`, so a check after any idle period triggers a full cold start (BGE
embeddings + FAISS indices + two classifiers + two resolution predictors loading fresh) —
36s measured live, comfortably past the workflow's `--max-time 20`. Every run for the
observable history of this workflow has been reporting a false "service down" the moment the
instance wasn't already warm, with 1 lucky exception where the prior request happened to keep
it warm.

This is worse than the gap it was built to close. A monitor that constantly false-fires trains
whoever's watching it to stop reading its alerts — the exact failure mode ("nobody notices an
outage") this workflow exists to prevent, just self-inflicted instead of caused by an actual
GCP-side outage.

## Decision

**Two independent fixes, addressing two independent problems.**

**1. Add a GCP Cloud Monitoring uptime check as the primary outage detector.** Created
`triageiq-health` (`expense-tracker-498014`, 5min period, 60s timeout, `GET /health` over
HTTPS, `$.status == "ok"` JSON-path match) with an email alert policy on
`gaurav.gandhi2411@gmail.com` (2 consecutive failures over a 10min window, matching the
existing `expense-tracker-health` check's pattern in the same project — same product family,
same maintainer, no reason to invent a different threshold). GCP uptime checks run from
Google's own global infrastructure at a fixed interval, not subject to GitHub Actions'
scheduled-workflow throttling — this is the check the "~30min to notice an outage" requirement
now actually rests on.

Chose a managed GCP Uptime Check over hand-rolling a Cloud Scheduler job + Cloud Function:
purpose-built for exactly this (uptime + alerting, no custom code to maintain), zero
marginal cost at this volume (well within Cloud Monitoring's free tier — the project already
runs one for the sibling `expense-tracker` service, gone unnoticed as a billable cost), and
mirrors an already-proven pattern in the same project rather than introducing a new one.

**2. Fix `health-monitor.yml`'s timeout, keep it as a secondary check, don't delete it.**
`--max-time 20` → `90`, plus one retry (worst case ~190s, negligible against the 30min+ run
cadence) — sized against the 36s measured cold start with real margin, while staying well
under Cloud Run's own startup-probe tolerance (`deploy.yml`: up to ~180s across 3 attempts)
so a timeout here still means something is genuinely wrong, not just cold. Kept, not retired:
different infrastructure (GitHub Actions vs. Google's uptime-check network) than the GCP
check, so it's a real independent signal, not just a duplicate, and it already existed —
throwing away a working (once fixed) redundant check for no reason isn't the boring-technology
call.

## Consequences

- **What changes:** GCP uptime check + alert policy created (billing: effectively $0 at this
  check volume). `health-monitor.yml`'s timeout fixed; both workflow-level comments corrected
  to state the real ~34-135min interval instead of the disproven "a few minutes late" claim,
  and to document the GCP check as primary.
- **What becomes harder:** Nothing — this is strictly additive plus a bugfix.
- **What becomes easier:** An actual prod outage now gets caught within ~10-15 minutes by the
  GCP check (5min period, 2-consecutive-failure trigger) instead of a best-effort, occasionally
  hours-late GitHub Actions run. `health-monitor.yml`'s alerts are trustworthy again instead of
  arriving on almost every single run regardless of real service state.
- **Verification done:** live `/health` call confirmed genuinely healthy (200, status=ok,
  36.1s cold). Uptime check and alert policy created and confirmed via `gcloud monitoring
  uptime describe` / `policies describe` — path is `/health` (not mangled), matcher targets
  `$.status == "ok"`, alert wired to the same notification channel as the existing
  `expense-tracker-health` alert. `health-monitor.yml` YAML syntax validated
  (`yaml.safe_load`); a `workflow_dispatch` run confirms the timeout fix live (see PR).
- **Reversible:** the uptime check and alert policy can be deleted with `gcloud monitoring
  uptime delete` / a policy delete, no data or migration involved. The workflow fix is a
  single-commit revert if ever needed.

## Alternatives considered

- **Just widen `health-monitor.yml`'s timeout, skip the GCP check** — rejected: fixes the
  false-failure bug but leaves the original reported problem (up to 135min detection gaps)
  unaddressed; editing the cron expression can't fix GitHub's scheduler-level throttling.
- **Hand-roll a GCP Cloud Scheduler job + Cloud Function for alerting** — rejected: more code
  to write and maintain (the Function, its error handling, its own retry/timeout logic) to
  reimplement a feature Cloud Monitoring Uptime Checks already provide as a managed product,
  for no capability gain.
- **Retire `health-monitor.yml` now that the GCP check exists** — rejected: it's a genuinely
  independent detection path (different provider, different failure domain) at zero
  incremental cost once its bug is fixed; removing a working redundant check isn't justified
  by anything in this investigation.
- **Accept the GitHub Actions drift and just document the wider detection window as the new
  norm** — rejected: the standing requirement is "~30min to notice an outage," not "whatever
  GitHub's scheduler happens to deliver," and a managed alternative that actually meets it
  exists at negligible cost — no reason to lower the bar when meeting it is cheap.
