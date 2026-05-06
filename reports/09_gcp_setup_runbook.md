# Day 9 — GCP Setup Runbook

**Status:** Infrastructure code committed. GCP resources not yet created.  
**Last updated:** 2026-05-02

---

## Resources created by this runbook

| Resource | Name | Notes |
|---|---|---|
| GCP Project | `triageiq-portfolio` | Free tier |
| Artifact Registry repo | `triageiq` (us-central1) | Docker images |
| GCS bucket | `triageiq-portfolio-models` | Model artifacts |
| Cloud Run service | `triageiq-api` | Serverless, 0–5 instances |
| Secret Manager secret | `groq-api-key` | Injected at runtime |
| Service account | `triageiq-deployer@...` | CI/CD principal |
| IAM roles | run.admin, ar.writer, storage.viewer, secretmanager.accessor | On SA |

## Cost estimate

Cloud Run free tier: **2M requests/month + 360K GB-seconds memory/month + 180K vCPU-seconds/month**.  
Portfolio traffic estimate: <1K requests/month → **fully covered by free tier**.  
Model storage in GCS: ~2GB → **<$0.05/month** (free tier covers 5GB).  
Groq API: Free tier, rate-limited. No GCP cost.

---

## Prerequisites (do these once)

### 1. Google Cloud account

Create at [console.cloud.google.com](https://console.cloud.google.com) if you don't have one.

Free tier requires a billing account (credit card) — you will **not be charged** for resources within free-tier limits. Portfolio traffic is well within limits.

### 2. gcloud CLI

Install:
```bash
# macOS
brew install google-cloud-sdk

# Windows (PowerShell)
winget install Google.CloudSDK

# Or download installer: https://cloud.google.com/sdk/docs/install
```

Verify: `gcloud version`

### 3. Authenticate

```bash
gcloud auth login
gcloud config set project triageiq-portfolio
```

---

## Step-by-step setup

### Step 1 — Run GCP project setup script

```bash
bash deploy/scripts/setup_gcp.sh
```

This script:
- Creates project `triageiq-portfolio` (or skips if exists)
- Enables required APIs (Cloud Run, Artifact Registry, Secret Manager, etc.)
- Creates Artifact Registry repo `triageiq` in `us-central1`
- Creates GCS bucket `triageiq-portfolio-models`
- Creates service account `triageiq-deployer` with least-privilege IAM roles
- Writes SA key to `/tmp/triageiq-sa-key.json`
- Stores `GROQ_API_KEY` in Secret Manager

**Manual action required mid-script:** The script will pause and ask you to link a billing account:

```bash
gcloud billing accounts list                    # find your billing account ID
gcloud billing projects link triageiq-portfolio \
  --billing-account=XXXXXX-XXXXXX-XXXXXX
```

### Step 2 — Upload model artifacts to GCS

```bash
bash deploy/scripts/upload_models.sh
```

Uploads:
- `component_classifier_*.pkl` (~2MB each)
- `dup_index_*_bge/` (FAISS + BGE index, ~400MB each)
- `resolution_predictor_*.pkl` (~1MB each)
- `*_temporal_train.parquet` (~5MB each)

Total: ~1.5GB. Takes ~3–5 min on a typical connection.

### Step 3 — Set GitHub Actions secrets

In GitHub repo → Settings → Secrets → Actions:

| Secret name | Value |
|---|---|
| `GCP_SA_KEY` | Contents of `/tmp/triageiq-sa-key.json` |
| `GROQ_API_KEY` | Your Groq API key |

```bash
# Get SA key for clipboard/paste
cat /tmp/triageiq-sa-key.json
```

### Step 4 — Push to trigger first deploy

The CD workflow runs on every push to `main`. The infra commit has already been pushed.

```bash
git push origin main
```

Watch the deploy at: GitHub → Actions → Deploy to Cloud Run

### Step 5 — Verify

```bash
# Get service URL
gcloud run services describe triageiq-api \
  --region us-central1 \
  --format 'value(status.url)'

# Health check
curl https://triageiq-api-xxxxx-uc.a.run.app/health

# Triage a test issue
curl -X POST https://triageiq-api-xxxxx-uc.a.run.app/triage \
  -H "Content-Type: application/json" \
  -d '{"repo":"microsoft/vscode","title":"editor crashes on large file","body":"Opening any file over 10MB causes the editor to hang indefinitely."}'
```

Expected `/health` response:
```json
{"status":"ok","repos":["microsoft/vscode","kubernetes/kubernetes"],"uptime_seconds":12.3}
```

---

## Observability

### Structured logs

All `/triage` requests emit a JSON log line visible in Cloud Logging:

```json
{
  "request_id": "...",
  "endpoint": "/triage",
  "repo": "microsoft/vscode",
  "status": "success",
  "total_latency_ms": 1842.3,
  "system1_latency_ms": 3.1,
  "system2_latency_ms": 28.4,
  "system3_latency_ms": 4.2,
  "system4_latency_ms": 1806.6,
  "groq_tokens_prompt": 987,
  "groq_tokens_completion": 312,
  "estimated_cost_usd": 0.00035,
  "predicted_component": "editor-core"
}
```

Query in Cloud Logging:
```
resource.type="cloud_run_revision"
resource.labels.service_name="triageiq-api"
jsonPayload.endpoint="/triage"
```

### Alert policies

Apply the three pre-written alert policies after Cloud Monitoring is enabled:

```bash
# Apply via gcloud (one-time after setup)
# 1. Error rate > 5% over 5 min
# 2. p95 latency > 10s over 5 min
# 3. Estimated daily Groq cost > $4
# See deploy/monitoring/alerts.yaml for full config
```

Manual import: Cloud Console → Monitoring → Alerting → Create Policy → Import JSON.

---

## Rollback

The CD workflow auto-rolls back to the previous revision if the smoke test fails.

Manual rollback:
```bash
gcloud run services update-traffic triageiq-api \
  --region us-central1 \
  --to-revisions PREVIOUS_REVISION=100
```

---

## Benchmarks to record after first deploy

Fill in `reports/09_deployment.md` after live verification:

| Metric | Target | Actual |
|---|---|---|
| Docker image size (compressed) | <500MB | — |
| Cold start latency | <30s | — |
| Warm /health latency | <200ms | — |
| Warm /triage p50 latency | <3s | — |
| Warm /triage p95 latency | <8s | — |
| Memory usage (idle) | <1GB | — |
