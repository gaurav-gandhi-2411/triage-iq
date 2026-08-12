#!/usr/bin/env bash
# One-time GCP resource setup for TriageIQ, inside a dedicated billed project
# (2026-08-12 migration to a fresh project under a new GCP identity, after the
# prior co-tenant project's billing account went disabled a second time -- see
# docs/architecture/adr for the migration writeup; the 2026-08-05 migration off
# triageiq-portfolio-495022 to the since-abandoned expense-tracker-498014 is the
# prior entry in this same history). Every grant below is still scoped to the
# specific bucket/repo/service/SA/secret resource, never project-level, even
# though this project is no longer shared -- least-privilege by default, not
# only when forced to by co-tenancy. Auth is WIF-only (scripts/setup_wif.sh) --
# no service account key is ever created.
# Run once from a machine authenticated with gcloud (gcloud auth login),
# as a principal with Owner/Editor on the project.
# Usage: bash deploy/scripts/setup_gcp.sh

set -euo pipefail

PROJECT_ID="triageiq-prod-260812"
REGION="us-central1"
SA_NAME="triageiq-deployer"
RUNTIME_SA_NAME="triageiq-api-runtime"
AR_REPO="triageiq"
GCS_BUCKET="triageiq-prod-260812-models"
SERVICE_NAME="triageiq-api"

gcloud config set project "$PROJECT_ID"

echo "=== Enabling APIs ==="
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  sts.googleapis.com

echo "=== Creating Artifact Registry repository ==="
gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="TriageIQ Docker images" 2>/dev/null || \
  echo "Repository $AR_REPO already exists"

echo "=== Creating GCS bucket for models ==="
gcloud storage buckets create "gs://$GCS_BUCKET" --location="$REGION" --uniform-bucket-level-access 2>/dev/null || \
  echo "Bucket $GCS_BUCKET already exists"

echo "=== Creating service accounts ==="
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="TriageIQ CI deploy (GitHub Actions WIF)" 2>/dev/null || \
  echo "Service account $SA_NAME already exists"
gcloud iam service-accounts create "$RUNTIME_SA_NAME" \
  --display-name="TriageIQ Cloud Run runtime identity" 2>/dev/null || \
  echo "Service account $RUNTIME_SA_NAME already exists"

SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
RUNTIME_SA_EMAIL="$RUNTIME_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

# NOTE: the Cloud Run SERVICE resource ($SERVICE_NAME) must already exist before the
# run.developer binding below can be applied -- run it once manually
# (`gcloud run deploy $SERVICE_NAME ...`, see README.md) before this script's IAM step, or
# comment that block out on a from-scratch bootstrap and re-run after the first manual deploy.

echo "=== Granting scoped IAM roles (resource-level only -- NEVER project-level in a shared project) ==="
gcloud storage buckets add-iam-policy-binding "gs://$GCS_BUCKET" \
  --member="serviceAccount:$SA_EMAIL" --role="roles/storage.objectViewer"
gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" \
  --project="$PROJECT_ID" --location="$REGION" \
  --member="serviceAccount:$SA_EMAIL" --role="roles/artifactregistry.writer"
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
  --project="$PROJECT_ID" --region="$REGION" \
  --member="serviceAccount:$SA_EMAIL" --role="roles/run.developer"
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA_EMAIL" \
  --project="$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" --role="roles/iam.serviceAccountUser"
for SECRET in groq-api-key metrics-token; do
  gcloud secrets add-iam-policy-binding "$SECRET" --project="$PROJECT_ID" \
    --member="serviceAccount:$SA_EMAIL" --role="roles/secretmanager.secretAccessor"
  gcloud secrets add-iam-policy-binding "$SECRET" --project="$PROJECT_ID" \
    --member="serviceAccount:$RUNTIME_SA_EMAIL" --role="roles/secretmanager.secretAccessor"
done

echo "=== Storing GROQ_API_KEY in Secret Manager ==="
if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "GROQ_API_KEY not set — skipping Secret Manager. Run manually:"
  echo "  echo -n '\$GROQ_API_KEY' | gcloud secrets create groq-api-key --data-file=-"
else
  echo -n "$GROQ_API_KEY" | gcloud secrets create groq-api-key --data-file=- 2>/dev/null || \
    echo -n "$GROQ_API_KEY" | gcloud secrets versions add groq-api-key --data-file=-
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload production models to GCS: bash deploy/scripts/upload_models.sh"
echo "  2. Wire WIF for GitHub Actions: bash scripts/setup_wif.sh"
echo "  3. First deploy must be manual (a brand-new Cloud Run service can't be created"
echo "     via --no-traffic candidate deploy) -- see README.md's deploy section."
