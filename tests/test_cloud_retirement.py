"""Exercise retirement safety gates with fake CLIs; never contact or delete cloud resources."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


class CloudRetirementTests(unittest.TestCase):
    def run_retirement(self, case='ready', corrupt_checksum=False):
        script = Path(__file__).resolve().parents[1] / 'scripts/retire-gcp-demo.sh'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'scripts').mkdir()
            shutil.copy2(script, root / 'scripts/retire-gcp-demo.sh')
            (root / '.venv/bin').mkdir(parents=True)
            (root / '.venv/bin/python').symlink_to(sys.executable)
            endpoint = 'projects/123/locations/us-central1/endpoints/456'
            state = {'project': 'demo-project', 'bucket': 'demo-project-readmission-demo',
                     'endpoint': endpoint, 'training_job': 'fixture-training-job'}
            (root / 'cloud-state.json').write_text(json.dumps(state))
            with zipfile.ZipFile(root / 'gcp-evidence.zip', 'w') as archive:
                archive.writestr('cloud-state.json', json.dumps(state))
                for name in ['cloud_evidence.json', 'metrics.json', 'model.joblib']:
                    archive.writestr('artifacts/vertex/' + name, 'test fixture')
            digest = hashlib.sha256((root / 'gcp-evidence.zip').read_bytes()).hexdigest()
            (root / 'gcp-evidence.zip.sha256').write_text(('0' * 64 if corrupt_checksum else digest) + '  gcp-evidence.zip\n')
            stub = '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['CLI_LOG'], 'a') as log:
    log.write(json.dumps([Path(sys.argv[0]).name, *args]) + '\\n')
case = os.environ['RETIREMENT_CASE']
if args[:3] == ['ai', 'endpoints', 'describe']:
    if case == 'auth_failed':
        print('Fixture authentication failed', file=sys.stderr)
        sys.exit(1)
    response = {'name': os.environ['FIXTURE_ENDPOINT']}
    if case == 'deployed': response['deployedModels'] = [{'id': 'fixture-model'}]
    print(json.dumps(response))
elif args[:3] == ['ai', 'custom-jobs', 'list']:
    print(json.dumps([{'state': 'JOB_STATE_RUNNING' if case == 'active_job' else 'JOB_STATE_SUCCEEDED'}]))
'''
            for name in ['gcloud', 'bq']:
                path = root / '.venv/bin' / name
                path.write_text('#!' + sys.executable + '\n' + stub)
                path.chmod(0o755)
            log = root / 'cli-log.jsonl'
            environment = {**os.environ, 'PATH': str(root / '.venv/bin') + ':' + os.environ['PATH'],
                           'CLI_LOG': str(log), 'RETIREMENT_CASE': case, 'FIXTURE_ENDPOINT': endpoint}
            result = subprocess.run(['bash', str(root / 'scripts/retire-gcp-demo.sh'),
                                     '--backup-downloaded', 'demo-project'],
                                    env=environment, text=True, capture_output=True, timeout=15)
            calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            mutations = [call for call in calls if any(word in call for word in ['delete', 'update', 'rm'])]
            return result, mutations

    def test_auth_failure_never_deletes(self):
        result, mutations = self.run_retirement('auth_failed')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Nothing was deleted', result.stdout + result.stderr)
        self.assertEqual(mutations, [])

    def test_deployed_model_and_active_training_never_delete(self):
        for case in ['deployed', 'active_job']:
            with self.subTest(case=case):
                result, mutations = self.run_retirement(case)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(mutations, [])

    def test_bad_backup_checksum_never_deletes(self):
        result, mutations = self.run_retirement(corrupt_checksum=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('checksum', result.stderr)
        self.assertEqual(mutations, [])

    def test_verified_inactive_run_targets_only_demo_assets(self):
        result, mutations = self.run_retirement()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(mutations), 4)
        self.assertIn('readmission-demo', mutations[0])
        self.assertIn('demo-project:readmission_demo', mutations[1])
        self.assertIn('--clear-soft-delete', mutations[2])
        self.assertIn('gs://demo-project-readmission-demo', mutations[3])
        self.assertNotIn('projects', [word for call in mutations for word in call])


if __name__ == '__main__':
    unittest.main()
