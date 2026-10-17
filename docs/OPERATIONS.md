# Operations: deployment and monitoring

Moved verbatim from the README front page (which carries a short deployment summary). Numbers are re-read by `scripts/readme_numbers_check.py` (manifest `docs/numbers/operations.json`).

## Hosting history

Moved verbatim from the README front page (Live API section).

> Served from GCP project `triageiq-prod-260812` (region `us-central1`), a dedicated project under
> its own GCP identity, still under an IAM-scoped deploy identity with zero project-level grants.
> This is the second migration: the original project (`triageiq-portfolio-495022`) had its billing
> account closed (2026-08-05); the co-tenant stopgap that replaced it (`expense-tracker-498014`)
> then had ITS billing disabled too (2026-08-12), caught this time via CI failing rather than an
> undetected outage. Full rationale and the new billing-status monitoring added specifically to
> catch a third occurrence:
> [`docs/architecture/adr/0050-second-billing-outage-dedicated-project-migration.md`](architecture/adr/0050-second-billing-outage-dedicated-project-migration.md).

## Deployment

CD pipeline: `.github/workflows/deploy.yml`, triggered on every push to `main`.

### Steps

1. Authenticate to GCP via Workload Identity Federation (no static SA key in GitHub)
2. Download production models from GCS
3. Build Docker image from `docker/Dockerfile.prod` (CPU PyTorch, BGE model pre-baked, installs from `requirements.lock`)
4. Push to Artifact Registry
5. Deploy to Cloud Run as a no-traffic candidate revision (tagged `candidate`), with `GROQ_API_KEY` + `METRICS_TOKEN` injected from Secret Manager
6. Smoke test `/health`, `/metrics`, `/triage` against the candidate's own URL; if it fails, the candidate never receives traffic and the previous revision keeps serving (no rollback is needed)
7. Promote the candidate to live traffic (a `/triage` that degrades only on the Groq daily quota still promotes, with a labelled follow-up issue)

### WIF setup (one-time)

```bash
bash scripts/setup_wif.sh
# Prints GCP_WIF_PROVIDER → add as GitHub repository variable
# Settings → Secrets and variables → Actions → Variables tab
```

### Lock file regeneration

```bash
# Must run on Linux for correct wheel hashes
docker run --rm -v "$PWD":/src -w /src python:3.11-slim \
  pip-compile requirements.txt --output-file=requirements.lock --no-header --no-annotate
```

### Manual redeploy

Trigger via GitHub Actions UI: `Actions → Deploy to Cloud Run → Run workflow → Branch: main`.

---

## Monitoring

### Prometheus metrics

```bash
curl https://triageiq-api-1014562031321.us-central1.run.app/metrics \
  -H "Authorization: Bearer $METRICS_TOKEN"
```

Auth behavior:
- `METRICS_TOKEN` set → requires `Authorization: Bearer <token>` (401 otherwise)
- No token + `ENVIRONMENT=prod` → 503 (fail-closed, prevents silent exposure)
- No token + `ENVIRONMENT=dev` → open (local dev only)

### Uptime, keep-warm and alerting (decision, not an open gap)

| Layer | What it does | Evidence |
|---|---|---|
| GitHub Actions health monitor (`.github/workflows/health-monitor.yml`) | `GET /health?deps=1` (model store + one zero-token Groq call), fails loudly on anything but 200 + `status=ok`. Scheduled every 30 min, but GitHub runs it irregularly: the last 8 runs (2026-10-05 to 2026-10-07) were 3–7 h apart, so treat the detection window as hours | `gh run list --workflow=health-monitor.yml` |
| External UptimeRobot monitor (free plan, 5-minute interval, bare `/health`) | Uptime check **and** keep-warm: first ping 2026-10-05 12:39:50 UTC; cold starts fell from 6–11/day to 1 (2026-10-06) and 0 (2026-10-07) | `docs/DECISION_LOG_2026-10.md` D10 (Phase 5), Cloud Run request logs |
| Cloud Monitoring alert policies / uptime checks | **Not adopted.** Alerting could not be verified to be free, and this project runs entirely on free tiers by design. A read-only `gcloud monitoring policies list` and `uptime list-configs` against `triageiq-prod-260812` on 2026-10-07 returned no policies and no uptime checks | owner decision; `scripts/setup_monitoring.sh` is kept as a reference for a future paid setup, **not** applied |

UptimeRobot currently sends `HEAD` first (the API answers 405) and then `GET`; switching the monitor to `GET` is queued in `docs/QUEUE_FOR_GG.md`.

### Log filtering (Cloud Logging)

```
jsonPayload.log_type="access"                                      # all access logs
jsonPayload.log_type="access" AND jsonPayload.status="error"       # failed requests
jsonPayload.log_type="access" AND jsonPayload.llm_status="parse_failure"  # LLM degraded
```

---

## Manual checks the number checker cannot do

`scripts/readme_numbers_check.py` checks digits against their sources. It does **not** check dates (they are excluded as identifiers), words (for example "one" vs "two", "fell" vs "rose", "served" vs "not served"), claims about behaviour, images, or the GitHub repository description. Before merging a change to `README.md` or any `docs/*.md` that makes such a claim, tick these by hand:

- [ ] Every date in the changed text is checked against the event's own record (`gh run list --workflow=deploy.yml` for deploys, the ADR or decision-log entry for decisions), not against another document.
- [ ] Words beside a number (better/worse, fell/rose, served/not served, before/after) agree with the sign and the table of the sourced number.
- [ ] Statements about behaviour (deploy order, rollback, rate limits, degradation) match the code or workflow they describe (`.github/workflows/deploy.yml`, `src/triage_iq/api/app.py`, `src/triage_iq/models/triage.py`).
- [ ] The API response-field list and the `_llm_status` values match `src/triage_iq/api/app.py` and `src/triage_iq/models/triage.py`.
- [ ] Diagram text (the ASCII box in the README and `docs/screenshots/pipeline-diagram.svg`) describes what is SERVED today.
- [ ] The repository description equals `docs/repo_description.txt` (`gh repo view --json description`); update with `gh repo edit --description "$(cat docs/repo_description.txt)"`.
- [ ] Open the README on GitHub after the merge and follow each relative link.
