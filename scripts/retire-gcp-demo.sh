#!/usr/bin/env bash
# Removes stored demo assets only after a backup is downloaded to the owner's computer.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != "--backup-downloaded" ]]; then
  printf 'Download and verify gcp-evidence.zip on your computer first. Then run with --backup-downloaded.\n' >&2
  exit 2
fi
PROJECT_ID="${2:?Pass your dedicated demo project ID after --backup-downloaded}"
export PROJECT_ID
REGION="us-central1"
.venv/bin/python - <<'PY'
import json, zipfile, os
from pathlib import Path
from google.cloud import aiplatform
state = json.loads(Path('cloud-state.json').read_text())
if state.get('project') != os.environ['PROJECT_ID'] or state.get('bucket') != os.environ['PROJECT_ID'] + '-readmission-demo':
    raise SystemExit('State does not match the demo resources. Nothing was deleted.')
with zipfile.ZipFile('gcp-evidence.zip') as archive:
    if archive.testzip():
        raise SystemExit('Backup is corrupt. Nothing was deleted.')
    for required in ('cloud-state.json', 'artifacts/vertex/cloud_evidence.json',
                     'artifacts/vertex/metrics.json', 'artifacts/vertex/model.joblib'):
        if required not in archive.namelist():
            raise SystemExit(f'Backup is missing {required}. Nothing was deleted.')
    archived = json.loads(archive.read('cloud-state.json'))
    if archived.get('training_job') != state.get('training_job'):
        raise SystemExit('Backup belongs to another run. Nothing was deleted.')
aiplatform.init(project=state['project'], location='us-central1')
endpoint = aiplatform.Endpoint(state['endpoint'])
if endpoint.gca_resource.deployed_models:
    raise SystemExit('Endpoint has deployed models. Undeploy before removing stored assets.')
active = [job.resource_name for job in aiplatform.CustomJob.list() if job.state.name in {
    'JOB_STATE_PENDING', 'JOB_STATE_QUEUED', 'JOB_STATE_RUNNING', 'JOB_STATE_CANCELLING'}]
if active:
    raise SystemExit('Active training jobs remain. Nothing was deleted.')
print('Checks passed. Removing only the named demo repository, dataset, and bucket.')
PY
# Registry model metadata and the empty endpoint remain, without the deleted assets.
gcloud artifacts repositories delete readmission-demo --project="$PROJECT_ID" --location="$REGION" --quiet
bq --project_id="$PROJECT_ID" --location="$REGION" rm -r -f "$PROJECT_ID:readmission_demo"
# Disable soft delete for new deletions after the verified local backup.
gcloud storage buckets update "gs://${PROJECT_ID}-readmission-demo" --clear-soft-delete
gcloud storage rm --recursive "gs://${PROJECT_ID}-readmission-demo"
printf 'Demo container repository, BigQuery dataset, and Cloud Storage bucket removed. Backup retained.\n'
