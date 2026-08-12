#!/usr/bin/env bash
# One-time setup: Workload Identity Federation for GitHub Actions → GCP.
#
# Run this once from a machine authenticated as a GCP project owner.
# After running:
#   1. Store the printed GCP_WIF_PROVIDER value as a GitHub repository variable
#      (Settings → Secrets and variables → Variables).
#   2. Trigger a test deploy from the branch BEFORE merging to confirm WIF works.
#   3. Only after a successful deploy: merge the PR, then delete GCP_SA_KEY.
#
# Usage:
#   bash scripts/setup_wif.sh

set -euo pipefail

PROJECT_ID="triageiq-prod-260812"
REPO="gaurav-gandhi-2411/triage-iq"   # verified from git remote get-url origin
SA_EMAIL="triageiq-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
POOL_ID="gh-actions-pool"
PROVIDER_ID="gh-actions-provider"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")

# ── 0. Pre-flight: confirm the service account exists ────────────────────────
echo "=== 0. Verifying service account exists ==="
gcloud iam service-accounts describe "$SA_EMAIL" \
  --project="$PROJECT_ID" \
  --format="value(email,displayName,disabled)"
echo "Service account confirmed."

# ── 1. Create Workload Identity Pool ────────────────────────────────────────
echo ""
echo "=== 1. Create Workload Identity Pool ==="
gcloud iam workload-identity-pools create "$POOL_ID" \
  --project="$PROJECT_ID" \
  --location="global" \
  --display-name="GitHub Actions pool" 2>&1 \
  | grep -v "already exists" || true
echo "Pool: $POOL_ID"

# ── 2. Create OIDC Provider ──────────────────────────────────────────────────
echo ""
echo "=== 2. Create OIDC Provider ==="
gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_ID" \
  --project="$PROJECT_ID" \
  --location="global" \
  --workload-identity-pool="$POOL_ID" \
  --display-name="GitHub OIDC provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.actor=assertion.actor" \
  --attribute-condition="attribute.repository == '${REPO}'" 2>&1 \
  | grep -v "already exists" || true
echo "Provider: $PROVIDER_ID"

# ── 3. Bind service account to WIF pool ─────────────────────────────────────
echo ""
echo "=== 3. Bind ${SA_EMAIL} to WIF pool ==="
MEMBER="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/attribute.repository/${REPO}"
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="$MEMBER"
echo "Binding added."

# ── 4. Verification ──────────────────────────────────────────────────────────
echo ""
echo "=== 4. Verification ==="
WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}"

echo "WIF_PROVIDER value (copy this exactly):"
echo "  $WIF_PROVIDER"
echo ""

echo "Current IAM bindings for ${SA_EMAIL} (workloadIdentityUser):"
gcloud iam service-accounts get-iam-policy "$SA_EMAIL" \
  --project="$PROJECT_ID" \
  --format="yaml" \
  | grep -A3 "workloadIdentityUser" || echo "(no workloadIdentityUser bindings found — check for errors above)"

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Add GCP_WIF_PROVIDER as a GitHub repository variable:"
echo "     Value: $WIF_PROVIDER"
echo "     Path:  Settings → Secrets and variables → Actions → Variables tab"
echo ""
echo "  2. Add SMOKE_TEST_METRICS_TOKEN as a GitHub Actions repository secret:"
echo "     Value: \$(gcloud secrets versions access latest --secret=metrics-token)"
echo "     Path:  Settings → Secrets and variables → Actions → Secrets tab"
echo ""
echo "  3. Trigger a test deploy from hardening/pr4-polish BEFORE merging:"
echo "     Actions → Deploy to Cloud Run → Run workflow → Branch: hardening/pr4-polish"
echo ""
echo "  4. Confirm smoke test passes (/health, /metrics, /triage all green)."
echo "     Only then merge the PR."
echo ""
echo "  5. Post-merge: delete GCP_SA_KEY from GitHub Secrets."

# ── Rollback (if needed) ─────────────────────────────────────────────────────
# To undo everything created by this script, run:
#
#   gcloud iam workload-identity-pools providers delete gh-actions-provider \
#     --workload-identity-pool=gh-actions-pool --location=global \
#     --project=triageiq-prod-260812
#
#   gcloud iam workload-identity-pools delete gh-actions-pool \
#     --location=global --project=triageiq-prod-260812
#
# Note: pool deletion is soft-deleted (30-day recovery window).
# The IAM binding on the SA will be removed automatically when the pool is deleted.
