#!/usr/bin/env bash
# A dedicated demo project and linked billing/trial credits are required.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ID="${1:-${PROJECT_ID:-}}"
if [[ -z "$PROJECT_ID" ]]; then
  printf 'Usage: bash scripts/run-gcp.sh YOUR_PROJECT_ID\n' >&2
  exit 2
fi
export PROJECT_ID
REGION="us-central1"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/readmission-demo/readmission:v1"
PYTHON_BIN=".venv/bin/python"
RUN_ENDPOINT_CLEANUP=0

on_exit() {
  local status=$?
  trap - EXIT
  if [[ "$RUN_ENDPOINT_CLEANUP" == "1" && -f cloud-state.json ]]; then
    if ! "$PYTHON_BIN" -m readmission.cloud cleanup --project "$PROJECT_ID"; then
      printf 'Automatic undeployment failed. Open Vertex AI Endpoints and undeploy the demo model.\n' >&2
      status=1
    fi
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v gcloud >/dev/null || { printf 'Run this script in Google Cloud Shell.\n' >&2; exit 1; }
python3 -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "Use Python 3.11 or 3.12 for this pinned environment."'
if [[ -f cloud-state.json ]]; then
  printf 'Cloud state already exists. Resume individual stages using GCP_SETUP.md to avoid duplicate jobs.\n' >&2
  exit 1
fi

printf '\n[1/7] Preparing the Python environment and cloud services\n'
python3 -m venv .venv
"$PYTHON_BIN" -m pip install -r requirements.txt
# Confirm SDK credentials before creating resources. Cloud Shell normally provides them.
"$PYTHON_BIN" -c 'import google.auth; google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"]); print("Google Cloud application credentials found.")'
bash scripts/bootstrap-gcp.sh

printf '\n[2/7] Downloading public data and preparing features in BigQuery\n'
"$PYTHON_BIN" -m readmission.prepare --download
"$PYTHON_BIN" -m readmission.cloud ingest --project "$PROJECT_ID"

printf '\n[3/7] Building the training and prediction container\n'
gcloud builds submit . --project="$PROJECT_ID" --region="$REGION" \
  --config=cloudbuild.yaml --substitutions="_IMAGE=${IMAGE}" \
  --gcs-source-staging-dir="gs://${PROJECT_ID}-readmission-demo/build-source" \
  --service-account="projects/${PROJECT_ID}/serviceAccounts/readmission-build@${PROJECT_ID}.iam.gserviceaccount.com"

printf '\n[4/7] Submitting Vertex AI training\n'
"$PYTHON_BIN" -m readmission.cloud train --project "$PROJECT_ID" --image "$IMAGE" \
  --service-account "readmission-train@${PROJECT_ID}.iam.gserviceaccount.com"

printf '\n[5/7] Waiting for successful training and collecting evidence\n'
TRAINING_COMPLETE=0
for attempt in $(seq 1 120); do
  if "$PYTHON_BIN" -m readmission.cloud collect --project "$PROJECT_ID"; then
    TRAINING_COMPLETE=1
    break
  else
    READMISSION_COLLECT_EXIT=$?
    if [[ "$READMISSION_COLLECT_EXIT" != "2" ]]; then
      printf 'Collection failed. Resume after resolving the error; the submitted cloud job is recorded in cloud-state.json.\n' >&2
      exit "$READMISSION_COLLECT_EXIT"
    fi
  fi
  READMISSION_JOB_STATUS="$($PYTHON_BIN -c 'import json; print(json.load(open("cloud-state.json")).get("training_status", "unknown"))')"
  case "$READMISSION_JOB_STATUS" in
    JOB_STATE_PENDING|JOB_STATE_QUEUED|JOB_STATE_RUNNING|submitted) sleep 20 ;;
    *) printf 'Training/collection needs attention: %s. See Vertex AI job logs.\n' "$READMISSION_JOB_STATUS" >&2; exit 1 ;;
  esac
done
if [[ "$TRAINING_COMPLETE" != "1" ]]; then
  printf 'Training is still pending. Resume the collect stage using GCP_SETUP.md.\n' >&2
  exit 1
fi

printf '\n[6/7] Deploying briefly and making a real prediction request\n'
RUN_ENDPOINT_CLEANUP=1
"$PYTHON_BIN" -m readmission.cloud deploy --project "$PROJECT_ID" --image "$IMAGE" \
  --service-account "readmission-serve@${PROJECT_ID}.iam.gserviceaccount.com"
"$PYTHON_BIN" -m readmission.cloud predict --project "$PROJECT_ID"
"$PYTHON_BIN" -m readmission.cloud cleanup --project "$PROJECT_ID"
RUN_ENDPOINT_CLEANUP=0

printf '\n[7/7] Packaging verified job IDs, metrics, and prediction evidence\n'
"$PYTHON_BIN" -m readmission.cloud evidence --project "$PROJECT_ID"
"$PYTHON_BIN" -c 'from pathlib import Path; import zipfile; z=zipfile.ZipFile("gcp-evidence.zip", "w", zipfile.ZIP_DEFLATED); z.write("cloud-state.json"); [z.write(p) for p in Path("artifacts/vertex").rglob("*") if p.is_file()]; z.close()'
printf '\nCompleted. Download gcp-evidence.zip from Cloud Shell. The prediction model is undeployed.\n'
