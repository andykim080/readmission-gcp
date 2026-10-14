#!/usr/bin/env bash
# Run in a dedicated demo project, after the user has completed billing setup.
set -euo pipefail
: "${PROJECT_ID:?Set PROJECT_ID to your dedicated GCP project ID.}"
REGION="${READMISSION_REGION:-us-central1}"
BUCKET="${PROJECT_ID}-readmission-demo"
REPOSITORY="readmission-demo"
TRAIN_SA="readmission-train@${PROJECT_ID}.iam.gserviceaccount.com"
SERVE_SA="readmission-serve@${PROJECT_ID}.iam.gserviceaccount.com"
BUILD_SA="readmission-build@${PROJECT_ID}.iam.gserviceaccount.com"

READMISSION_BILLING_ENABLED="$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingEnabled)')"
if [[ "$READMISSION_BILLING_ENABLED" != "True" && "$READMISSION_BILLING_ENABLED" != "true" ]]; then
  printf 'Link billing or activate eligible trial credits for this project first.\n' >&2
  exit 1
fi
gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com storage.googleapis.com \
  artifactregistry.googleapis.com cloudbuild.googleapis.com iam.googleapis.com logging.googleapis.com \
  compute.googleapis.com --project="$PROJECT_ID"

for account in readmission-train readmission-serve readmission-build; do
  if ! gcloud iam service-accounts describe "${account}@${PROJECT_ID}.iam.gserviceaccount.com" --project="$PROJECT_ID" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$account" --display-name="$account" --project="$PROJECT_ID"
  fi
done
if ! gcloud storage buckets describe "gs://${BUCKET}" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" --location="$REGION" --uniform-bucket-level-access \
    --public-access-prevention --project="$PROJECT_ID"
fi
if ! gcloud artifacts repositories describe "$REPOSITORY" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPOSITORY" --repository-format=docker --location="$REGION" --project="$PROJECT_ID"
fi
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member="serviceAccount:${TRAIN_SA}" --role=roles/storage.objectAdmin --project="$PROJECT_ID"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member="serviceAccount:${SERVE_SA}" --role=roles/storage.objectViewer --project="$PROJECT_ID"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --member="serviceAccount:${BUILD_SA}" --role=roles/storage.objectViewer --project="$PROJECT_ID"
gcloud artifacts repositories add-iam-policy-binding "$REPOSITORY" --location="$REGION" \
  --member="serviceAccount:${BUILD_SA}" --role=roles/artifactregistry.writer --project="$PROJECT_ID"
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${BUILD_SA}" --role=roles/logging.logWriter --condition=None
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${BUILD_SA}" --role=roles/serviceusage.serviceUsageConsumer --condition=None
printf 'Demo storage, image repository, and scoped service accounts are prepared in %s.\n' "$PROJECT_ID"
