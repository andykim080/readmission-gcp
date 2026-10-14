#!/usr/bin/env bash
# Run in Cloud Shell before deleting cloud assets.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import hashlib, json, zipfile
from pathlib import Path
state = Path('cloud-state.json')
evidence = Path('artifacts/vertex/cloud_evidence.json')
if not state.exists() or not evidence.exists():
    raise SystemExit('The original Cloud Shell state and completion evidence are required.')
run = json.loads(evidence.read_text())
if run.get('training_status') != 'JOB_STATE_SUCCEEDED' or run.get('endpoint_status') != 'undeployed':
    raise SystemExit('Finish prediction and undeployment before archiving completion evidence.')
files = [state] + sorted(p for p in Path('artifacts/vertex').rglob('*') if p.is_file())
with zipfile.ZipFile('gcp-evidence.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, str(path))
with zipfile.ZipFile('gcp-evidence.zip') as archive:
    if archive.testzip():
        raise SystemExit('Archive verification failed.')
checksum = hashlib.sha256(Path('gcp-evidence.zip').read_bytes()).hexdigest()
Path('gcp-evidence.zip.sha256').write_text(f'{checksum}  gcp-evidence.zip\n')
print(f'Archived {len(files)} files: model, metrics, report, and original cloud evidence.')
print('Download gcp-evidence.zip and gcp-evidence.zip.sha256 before retiring cloud resources.')
PY
