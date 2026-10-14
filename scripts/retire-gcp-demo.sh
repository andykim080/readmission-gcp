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
import json, zipfile, os, subprocess, hashlib
from pathlib import Path
state = json.loads(Path('cloud-state.json').read_text())
if state.get('project') != os.environ['PROJECT_ID'] or state.get('bucket') != os.environ['PROJECT_ID'] + '-readmission-demo':
    raise SystemExit('State does not match the demo resources. Nothing was deleted.')
expected = Path('gcp-evidence.zip.sha256').read_text().split()[0]
if hashlib.sha256(Path('gcp-evidence.zip').read_bytes()).hexdigest() != expected:
    raise SystemExit('Backup checksum does not match. Nothing was deleted.')
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
def cloud_json(arguments):
    # Cloud Shell's gcloud session and Python ADC use different credential paths.
    command = ['gcloud', 'ai', *arguments, '--project=' + state['project'],
               '--region=us-central1', '--format=json']
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                text=True, timeout=60)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
        raise SystemExit('Cloud status check failed. Nothing was deleted. Reconnect/authorize Cloud Shell and try again.')
endpoint = cloud_json(['endpoints', 'describe', state['endpoint']])
if not isinstance(endpoint, dict) or endpoint.get('name') != state['endpoint']:
    raise SystemExit('Unexpected endpoint response. Nothing was deleted.')
if endpoint.get('deployedModels'):
    raise SystemExit('Endpoint has deployed models. Undeploy before removing stored assets.')
jobs = cloud_json(['custom-jobs', 'list'])
if not isinstance(jobs, list):
    raise SystemExit('Unexpected job-list response. Nothing was deleted.')
terminal = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
if any(job.get('state') not in terminal for job in jobs):
    raise SystemExit('Active or unrecognized training jobs remain. Nothing was deleted.')
print('Checks passed. Removing only the named demo repository, dataset, and bucket.')
PY
# Registry model metadata and the empty endpoint remain, without the deleted assets.
gcloud artifacts repositories delete readmission-demo --project="$PROJECT_ID" --location="$REGION" --quiet
bq --project_id="$PROJECT_ID" --location="$REGION" rm -r -f "$PROJECT_ID:readmission_demo"
# Disable soft delete for new deletions after the verified local backup.
gcloud storage buckets update "gs://${PROJECT_ID}-readmission-demo" --clear-soft-delete
gcloud storage rm --recursive "gs://${PROJECT_ID}-readmission-demo"
printf 'Demo container repository, BigQuery dataset, and Cloud Storage bucket removed. Backup retained.\n'
