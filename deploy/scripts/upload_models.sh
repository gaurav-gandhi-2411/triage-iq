#!/usr/bin/env bash
# Upload production model artifacts to GCS.
# Run once (or when models are retrained) from the repo root.
# Usage: bash deploy/scripts/upload_models.sh

set -euo pipefail

BUCKET="gs://triageiq-models"

echo "=== Uploading production models to $BUCKET ==="

# TF-IDF classifiers
gsutil -m cp \
  data/models/component_classifier_microsoft_vscode.pkl \
  data/models/component_classifier_kubernetes_kubernetes.pkl \
  "$BUCKET/models/"

# BGE FAISS indices (NOT minilm, NOT distilbert)
gsutil -m cp -r \
  data/models/similar_issue_index_microsoft_vscode_bge \
  data/models/similar_issue_index_kubernetes_kubernetes_bge \
  "$BUCKET/models/"

# Resolution predictors
gsutil -m cp \
  data/models/resolution_predictor_microsoft_vscode.pkl \
  data/models/resolution_predictor_kubernetes_kubernetes.pkl \
  "$BUCKET/models/"

# Training data (needed for feature engineering at inference time)
gsutil -m cp \
  data/processed/microsoft_vscode_temporal_train.parquet \
  data/processed/kubernetes_kubernetes_temporal_train.parquet \
  "$BUCKET/processed/"

echo ""
echo "=== Upload complete ==="
gsutil ls -l "$BUCKET/models/"
gsutil ls -l "$BUCKET/processed/"
