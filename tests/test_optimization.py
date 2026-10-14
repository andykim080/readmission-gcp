import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from readmission.features import FEATURES, NUMERIC
from readmission.optimize import derive_features, diagnosis_group, fit_candidate, load_training, positive_class_weight, run_sweep


def synthetic_frame():
    rows = []
    for person in range(80):
        for visit in range(2):
            row = {name: (1 if name in NUMERIC else "No") for name in FEATURES}
            row.update(patient_nbr=str(person), encounter_id=str(person*2+visit),
                       label=int(person % 4 == 0), split="train", age="[60-70)",
                       discharge_disposition_id="1", diag_1="250.13", diag_2="428", diag_3="V45")
            rows.append(row)
    return pd.DataFrame(rows)


class OptimizationTests(unittest.TestCase):
    def test_feature_derivation_and_portable_serialization(self):
        frame = synthetic_frame()
        transformed = derive_features(frame[FEATURES])
        self.assertEqual(transformed.prior_total_visits.iloc[0], 3)
        self.assertEqual(transformed.diag_1_group.iloc[0], "diabetes")
        self.assertEqual(transformed.coded_diabetes_250.iloc[0], 1)
        self.assertEqual(transformed.coded_heart_failure_428.iloc[0], 1)
        self.assertEqual(transformed.coded_ckd_585.iloc[0], 0)
        self.assertEqual(diagnosis_group("459.9"), "circulatory")
        self.assertEqual(diagnosis_group("460"), "respiratory")
        self.assertEqual(diagnosis_group("__MISSING__"), "missing_or_unmapped")
        self.assertNotIn("label", transformed)
        spec = {"family": "lr", "C": 0.1, "early_stopping": False, "engineered_features": True}
        model, _ = fit_candidate(spec, frame)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            self.assertEqual(restored.predict_proba(frame[FEATURES]).shape, (160, 2))

    def test_class_weight_comes_from_fit_patients(self):
        frame = synthetic_frame()
        spec = {"family": "xgb", "scale_pos_weight": "ratio", "n_estimators": 10,
                "learning_rate": 0.1, "max_depth": 2, "min_child_weight": 1,
                "reg_lambda": 1, "early_stopping": False}
        self.assertEqual(positive_class_weight(spec, frame), 3)
        model, metadata = fit_candidate(spec, frame)
        self.assertEqual(model.named_steps["model"].scale_pos_weight, 3)
        self.assertEqual(metadata["positive_class_weight"], 3)
        lr_spec = {"family": "lr", "C": 0.01, "class_weight": "balanced", "early_stopping": False}
        lr, _ = fit_candidate(lr_spec, frame)
        self.assertEqual(lr.named_steps["model"].class_weight, "balanced")

    def test_early_stopping_refit_and_training_partition_guard(self):
        frame = synthetic_frame()
        spec = {"family": "xgb", "n_estimators": 20, "learning_rate": 0.1,
                "max_depth": 2, "min_child_weight": 1, "reg_lambda": 1, "early_stopping": True}
        model, metadata = fit_candidate(spec, frame)
        self.assertEqual(metadata["inner_patient_overlap"], 0)
        self.assertEqual(metadata["inner_fit_patients"] + metadata["inner_stop_patients"], 80)
        self.assertTrue(1 <= metadata["chosen_trees"] <= 20)
        self.assertEqual(model.named_steps["model"].n_estimators, metadata["chosen_trees"])
        self.assertIsNone(model.named_steps["model"].early_stopping_rounds)
        contaminated = frame.copy()
        contaminated.loc[0, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "only the training partition"):
            run_sweep(contaminated, "unused", specs=[])

    def test_loader_excludes_all_nontraining_patients(self):
        # Use the real source to verify the filter rather than spoofing patient hashes.
        if not Path("data/processed/features.csv").exists():
            self.skipTest("Prepare source data first")
        frame = load_training("data/processed/features.csv")
        self.assertEqual(set(frame.split), {"train"})
        self.assertEqual(len(frame), 51325)


if __name__ == "__main__":
    unittest.main()
