import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from readmission import cloud


class CloudEvidenceTests(unittest.TestCase):
    def test_collection_retrieves_existing_job_and_reports_pending_without_retrying_errors(self):
        resource = "projects/123/locations/us-central1/customJobs/456"
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "cloud-state.json"
            state.write_text(json.dumps({"training_job": resource}))
            job = SimpleNamespace(state=SimpleNamespace(name="JOB_STATE_RUNNING"))
            with patch.object(cloud, "STATE", state), \
                    patch("google.cloud.aiplatform.init"), \
                    patch("google.cloud.aiplatform.CustomJob.get", return_value=job) as get_job:
                with self.assertRaises(SystemExit) as result:
                    cloud.collect(SimpleNamespace(project="demo-project", region="us-central1"))
                self.assertEqual(result.exception.code, 2)
                get_job.assert_called_once_with(resource)
                self.assertEqual(json.loads(state.read_text())["training_status"], "JOB_STATE_RUNNING")

    def test_incomplete_or_local_execution_cannot_be_reported_as_cloud_completion(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            os.chdir(temporary)
            try:
                state = Path("cloud-state.json")
                state.write_text(json.dumps({"project": "demo-project", "training_status": "submitted"}))
                with patch.object(cloud, "STATE", state):
                    with self.assertRaisesRegex(SystemExit, "successful training"):
                        cloud.evidence(SimpleNamespace(project="demo-project"))
                    directory = Path("artifacts/vertex")
                    directory.mkdir(parents=True)
                    state.write_text(json.dumps({"project": "demo-project", "bigquery_local_parity": "passed",
                                                "training_status": "JOB_STATE_SUCCEEDED", "endpoint_status": "undeployed",
                                                "endpoint": "test-fixture-endpoint"}))
                    (directory / "endpoint_prediction.json").write_text(json.dumps({"endpoint": "test-fixture-endpoint", "predictions": [0.1]}))
                    (directory / "metrics.json").write_text(json.dumps({"execution": "local"}))
                    with self.assertRaisesRegex(SystemExit, "not produced by Vertex"):
                        cloud.evidence(SimpleNamespace(project="demo-project"))
                    self.assertFalse((directory / "cloud_evidence.json").exists())
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
