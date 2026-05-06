# TriageIQ — Production-Grade GitHub Issue Intelligence

Automatic classification, duplicate detection, resolution-time prediction, and LLM-powered triage — built on real OSS data with full evaluation, deployment, and observability.

**Targets:** Senior Applied ML / ML Engineer roles at Microsoft and Google. Demonstrates full ML production lifecycle on real-world enterprise patterns.

---

## Live Service

**Production API:** https://triageiq-api-779563952988.us-central1.run.app

```bash
# Health check
curl https://triageiq-api-779563952988.us-central1.run.app/health

# Triage an issue
curl -X POST https://triageiq-api-779563952988.us-central1.run.app/triage \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Editor crashes when opening large JSON files",
    "body": "VS Code becomes completely unresponsive when I try to open files larger than 50MB.",
    "repo": "microsoft/vscode"
  }'
```

**Supported repos:** `microsoft/vscode`, `kubernetes/kubernetes`

**Rate limits:** `10 requests/hour` and `30 requests/day` per IP. `/health` and `/metrics` are not rate-limited.

Infrastructure: FastAPI on Google Cloud Run (0–3 instances, free tier) · Groq llama-3.1-8b-instant · Cloud Logging · Cloud Monitoring · GitHub Actions CI/CD  
Latency: ~23s cold start, ~3.5s warm p50

---

## System Architecture

Four ML systems working in concert:

| System | Technique | Task |
|---|---|---|
| 1 — Component Classifier | TF-IDF + Logistic Regression | Predict issue component (label) |
| 2 — Duplicate Detector | BGE embeddings + FAISS | Retrieve top-5 similar issues |
| 3 — Resolution Predictor | LightGBM quantile regression | Predict days-to-close (p10/p50/p90) |
| 4 — LLM Triage Assistant | Groq llama-3.1-8b-instant | Synthesize structured `TriagePlan` |

---

## Evaluation

Judge evaluation (LLM-as-judge, Claude claude-haiku-4-5-20251001):

| System | Metric | Score |
|---|---|---|
| Component classifier | Top-1 accuracy | LLM 58.3% vs TF-IDF 61.7% vs Majority 11.7% |
| LLM triage plan | Judge score (15-pt rubric) | ~10-11/15 avg (vscode) |

---

## Local Development

```bash
# Install
pip install -e .

# Run API
GROQ_API_KEY=your_key uvicorn triage_iq.api.app:app --reload

# Test
pytest tests/
```

---

## Deployment

Hosted on Google Cloud Run via GitHub Actions:

- Push to `main` triggers `.github/workflows/deploy.yml`
- Builds `docker/Dockerfile.prod` (CPU PyTorch, BGE model baked in)
- Pushes to Artifact Registry `us-central1-docker.pkg.dev/triageiq-portfolio-495022/triageiq/api`
- Deploys to Cloud Run with `GROQ_API_KEY` from Secret Manager
- Smoke tests `/health` and rolls back if failed

Required GitHub secrets: `GCP_SA_KEY`, `GCP_PROJECT_ID`, `GROQ_API_KEY`

---

## Observability

### Prometheus metrics (`/metrics`)

The service exposes a Prometheus-compatible `/metrics` endpoint.

**Auth behavior (fail-closed in prod):**
- `METRICS_TOKEN` set (any environment) → requires `Authorization: Bearer <token>`; returns 401 otherwise
- `METRICS_TOKEN` unset, `ENVIRONMENT=prod` → returns 503 (prevents silent exposure if the secret reference breaks)
- `METRICS_TOKEN` unset, `ENVIRONMENT=dev` → open endpoint (local development only)

The operational state is logged at startup: `metrics endpoint: protected with token | open (dev only) | disabled (prod, no token)`.

```bash
# Scrape metrics (token required on the live service)
curl https://triageiq-api-779563952988.us-central1.run.app/metrics \
  -H "Authorization: Bearer $METRICS_TOKEN"
```

Custom metrics exposed:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `triage_requests_total` | Counter | `repo`, `status` | Total /triage calls; status: `success`, `error`, `fallback` |
| `triage_llm_fallback_total` | Counter | — | Calls where LLM parse failed and fallback plan was used |
| `triage_groq_tokens_total` | Counter | — | Cumulative Groq tokens consumed (prompt + completion) |
| `triage_latency_seconds` | Histogram | — | End-to-end request latency with buckets at 0.5, 1, 2, 5, 10, 30s |

Standard HTTP metrics (request count, latency per route) are automatically collected via `prometheus-fastapi-instrumentator`.

### Cloud Monitoring alerts

Three alert policies (email only, free tier), configured via `scripts/setup_monitoring.sh` (run once):

| Alert | Condition | Window |
|---|---|---|
| High error rate | >5% of requests return 5xx | 10 min |
| High p95 latency | p95 > 5s | 10 min |
| Groq quota warning | Daily token usage >70K (70% of 100K TPD limit) | 24h |

To set up alerts after first deploy:
```bash
GCP_PROJECT=triageiq-portfolio-495022 ALERT_EMAIL=you@example.com bash scripts/setup_monitoring.sh
```
