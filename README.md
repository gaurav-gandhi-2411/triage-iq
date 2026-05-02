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

Infrastructure: FastAPI on Google Cloud Run (0–5 instances, free tier) · Groq llama-3.1-8b-instant · Cloud Logging · GitHub Actions CI/CD  
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
