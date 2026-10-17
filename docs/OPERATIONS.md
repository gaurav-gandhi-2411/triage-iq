# Operations: deployment and monitoring

Moved verbatim from the README front page (which carries a short deployment summary). Numbers are re-read by `scripts/readme_numbers_check.py` (manifest `docs/numbers/operations.json`).

## Deployment

CD pipeline: `.github/workflows/deploy.yml`, triggered on every push to `main`.

### Steps

1. Authenticate to GCP via Workload Identity Federation (no static SA key in GitHub)
2. Download production models from GCS
3. Build Docker image from `docker/Dockerfile.prod` (CPU PyTorch, BGE model pre-baked, installs from `requirements.lock`)
4. Push to Artifact Registry
5. Deploy to Cloud Run with `GROQ_API_KEY` + `METRICS_TOKEN` injected from Secret Manager
6. Smoke test `/health`, `/metrics`, `/triage` — auto-rollback to previous revision on failure

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
| Cloud Monitoring alert policies / uptime checks | **Not adopted.** Alerting could not be verified to be free, and this project runs on a zero-spend rule. A read-only `gcloud monitoring policies list` and `uptime list-configs` against `triageiq-prod-260812` on 2026-10-07 returned no policies and no uptime checks | owner decision; `scripts/setup_monitoring.sh` is kept as a reference for a future paid setup, **not** applied |

UptimeRobot currently sends `HEAD` first (the API answers 405) and then `GET`; switching the monitor to `GET` is queued in `docs/QUEUE_FOR_GG.md`.

### Log filtering (Cloud Logging)

```
jsonPayload.log_type="access"                                      # all access logs
jsonPayload.log_type="access" AND jsonPayload.status="error"       # failed requests
jsonPayload.log_type="access" AND jsonPayload.llm_status="parse_failure"  # LLM degraded
```
