import copy
import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from readmission.evaluate import metrics, subgroup_audit, top_mask
from readmission.features import FEATURES, NUMERIC, patient_split, validate_instances
from readmission.train import predict_bundle, preprocessor


class WorkflowTests(unittest.TestCase):
    def test_capacity_allocation_and_zero_positive_group(self):
        selected = top_mask([0.9, 0.9, 0.8, 0.1, 0.0], 0.4)
        self.assertEqual(selected.tolist(), [True, True, False, False, False])
        result = metrics([1, 0, 1, 0, 0], [0.9, 0.9, 0.8, 0.1, 0.0], 0.4)
        self.assertAlmostEqual(result["recall_at_capacity"], 0.5)
        self.assertAlmostEqual(result["precision_at_capacity"], 0.5)
        frame = pd.DataFrame({"age": ["a"]*5, "race": ["r"]*5, "gender": ["g"]*5, "label": [0]*5})
        audit = subgroup_audit(frame, np.array([0.9, 0.9, 0.8, 0.1, 0.0]))
        self.assertIsNone(audit[0]["recall_at_global_capacity"])

    def test_transformer_uses_training_values_and_handles_new_categories(self):
        records = [{name: (i + 1 if name in NUMERIC else f"cat-{i%2}") for name in FEATURES} for i in range(30)]
        frame = pd.DataFrame(records)
        transformer = preprocessor().fit(frame)
        scaler = transformer.named_transformers_["numeric"].named_steps["scale"]
        self.assertAlmostEqual(scaler.mean_[0], 15.5)
        unseen = frame.iloc[[0]].copy()
        unseen["diag_1"] = "previously-unseen-diagnosis"
        unseen["time_in_hospital"] = 1000
        transformer.transform(unseen)
        self.assertAlmostEqual(scaler.mean_[0], 15.5)

    @unittest.skipUnless(Path("data/processed/features.csv").exists(), "Prepare real data first")
    def test_real_data_patient_groups_are_disjoint(self):
        frame = pd.read_csv("data/processed/features.csv", dtype={"patient_nbr": str})
        groups = {name: set(part.patient_nbr) for name, part in frame.groupby("split")}
        for left, right in [("train", "test"), ("train", "validation"), ("validation", "test")]:
            self.assertFalse(groups[left] & groups[right])
        self.assertTrue((frame.patient_nbr.map(patient_split) == frame.split).all())
        self.assertNotIn("patient_nbr", FEATURES)
        self.assertNotIn("readmitted", FEATURES)

    @unittest.skipUnless(Path("artifacts/model.joblib").exists(), "Train real model first")
    def test_serving_predictions_match_saved_model_and_invalid_inputs_fail(self):
        from readmission.serve import app
        request = json.loads(Path("artifacts/sample_request.json").read_text())
        bundle = joblib.load("artifacts/model.joblib")
        expected = predict_bundle(bundle, validate_instances(request["instances"]))[0]
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            response = client.post("/predict", json=request)
            self.assertEqual(response.status_code, 200)
            self.assertAlmostEqual(response.json()["predictions"][0]["readmission_risk"], expected)
            contaminated = copy.deepcopy(request)
            contaminated["instances"][0]["readmitted"] = "<30"
            self.assertEqual(client.post("/predict", json=contaminated).status_code, 422)
            invalid = copy.deepcopy(request)
            invalid["instances"][0]["number_inpatient"] = -1
            self.assertEqual(client.post("/predict", json=invalid).status_code, 422)
            outside_cohort = copy.deepcopy(request)
            outside_cohort["instances"][0]["discharge_disposition_id"] = "11"
            self.assertEqual(client.post("/predict", json=outside_cohort).status_code, 422)


if __name__ == "__main__":
    unittest.main()
