#!/usr/bin/env bash
# One-time GCP project setup for TriageIQ.
# Run once from a machine authenticated with gcloud (gcloud auth login).
# Usage: bash deploy/scripts/setup_gcp.sh

set -euo pipefail

PROJECT_ID="triageiq-portfolio-495022"
REGION="us-central1"
SA_NAME="triageiq-deployer"
AR_REPO="triageiq"
GCS_BUCKET="triageiq-portfolio-495022-models"

echo "=== Creating GCP project ==="
gcloud projects create "$PROJECT_ID" --name="TriageIQ Portfolio" 2>/dev/null || \
  echo "Project $PROJECT_ID already exists"
gcloud config set project "$PROJECT_ID"

# Must link billing account for Cloud Run — get your billing account ID from:
# gcloud billing accounts list
echo ""
echo "ACTION REQUIRED: link a billing account:"
echo "  gcloud billing projects link $PROJECT_ID --billing-account=XXXXXX-XXXXXX-XXXXXX"
echo ""

echo "=== Enabling APIs ==="
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com

echo "=== Creating Artifact Registry repository ==="
gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker \
  --location="$REGION" \
  --description="TriageIQ Docker images" 2>/dev/null || \
  echo "Repository $AR_REPO already exists"

echo "=== Creating GCS bucket for models ==="
gsutil mb -l "$REGION" "gs://$GCS_BUCKET" 2>/dev/null || \
  echo "Bucket $GCS_BUCKET already exists"

echo "=== Creating service account ==="
gcloud iam service-accounts create "$SA_NAME" \
  --display-name="TriageIQ CI/CD Deployer" 2>/dev/null || \
  echo "Service account $SA_NAME already exists"

SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

echo "=== Granting IAM roles ==="
for ROLE in \
  roles/run.admin \
  roles/artifactregistry.writer \
  roles/storage.objectViewer \
  roles/secretmanager.secretAccessor \
  roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA_EMAIL" \
    --role="$ROLE" \
    --quiet
done

echo "=== Creating service account key for GitHub Actions ==="
gcloud iam service-accounts keys create /tmp/triageiq-sa-key.json \
  --iam-account="$SA_EMAIL"
echo ""
echo "SA key written to /tmp/triageiq-sa-key.json"
echo "Add this as GitHub secret GCP_SA_KEY:"
echo "  cat /tmp/triageiq-sa-key.json | base64 | pbcopy   # macOS"
echo "  cat /tmp/triageiq-sa-key.json                      # then paste into GitHub"
echo ""

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
echo "Next: upload production models to GCS:"
echo "  bash deploy/scripts/upload_models.sh"
echo ""
echo "GitHub Secrets to set:"
echo "  GCP_SA_KEY   — contents of /tmp/triageiq-sa-key.json"
echo "  GCP_PROJECT_ID — $PROJECT_ID"
echo "  GROQ_API_KEY — your Groq API key"
