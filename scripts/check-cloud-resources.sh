#!/usr/bin/env bash
# Read-only inventory of demo resources, not a billing forecast.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ID="${1:?Pass your dedicated demo project ID}"
REGION="us-central1"
gcloud billing projects describe "$PROJECT_ID" --format='yaml(projectId,billingEnabled)'
gcloud ai endpoints list --project="$PROJECT_ID" --region="$REGION" --format=json
gcloud ai custom-jobs list --project="$PROJECT_ID" --region="$REGION" --format=json
gcloud artifacts repositories describe readmission-demo --project="$PROJECT_ID" --location="$REGION" --format=json
gcloud storage ls --long "gs://${PROJECT_ID}-readmission-demo/**"
bq --project_id="$PROJECT_ID" --location="$REGION" ls --format=prettyjson "$PROJECT_ID:readmission_demo"
