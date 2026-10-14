"""Same training entry point runs locally or in a Vertex AI custom job."""
import argparse
import json
import os
import platform
import shutil
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import BaggingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .evaluate import metrics, subgroup_audit, top_mask
from .features import AUDIT, CATEGORICAL, FEATURES, MISSING, NUMERIC, SEED, patient_split


def preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=25), CATEGORICAL),
    ], remainder="drop", sparse_threshold=1.0)


def model_candidates():
    return {
        "logistic_regression": LogisticRegression(C=0.5, max_iter=2000, random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.85, colsample_bytree=0.85, reg_lambda=5,
                                  min_child_weight=20, tree_method="hist", eval_metric="logloss",
                                  random_state=SEED, n_jobs=2),
        "bagged_trees": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=8, min_samples_leaf=30, random_state=SEED),
            n_estimators=100, max_samples=0.8, max_features=0.8,
            bootstrap=True, random_state=SEED, n_jobs=2,
        ),
    }


def cross_validate_models(train_frame, n_splits=5):
    """Fit preprocessing within each fold; only the training partition participates."""
    from sklearn.base import clone
    folds = list(GroupKFold(n_splits=n_splits).split(
        train_frame[FEATURES], train_frame.label, groups=train_frame.patient_nbr))
    results = []
    for name, estimator in model_candidates().items():
        for number, (fit_index, holdout_index) in enumerate(folds, start=1):
            fit = train_frame.iloc[fit_index]
            holdout = train_frame.iloc[holdout_index]
            overlap = set(fit.patient_nbr) & set(holdout.patient_nbr)
            if overlap:
                raise ValueError("Patient leakage in cross-validation.")
            pipeline = Pipeline([("features", preprocessor()), ("model", clone(estimator))])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(fit[FEATURES], fit.label)
            scores = pipeline.predict_proba(holdout[FEATURES])[:, 1]
            values = metrics(holdout.label, scores)
            results.append({"model": name, "fold": number, "train_patients": int(fit.patient_nbr.nunique()),
                            "holdout_patients": int(holdout.patient_nbr.nunique()), "patient_overlap": 0,
                            **{key: value for key, value in values.items() if key != "calibration"}})
            print(f"CV {name} fold {number}/{n_splits}: AP={values['average_precision']:.4f}", flush=True)
    rows = pd.DataFrame(results)
    keys = ["average_precision", "roc_auc", "brier_score", "precision_at_capacity", "recall_at_capacity"]
    summary = {name: {key: {"mean": float(group[key].mean()), "std": float(group[key].std(ddof=1))}
                      for key in keys} for name, group in rows.groupby("model")}
    return rows, summary


def log_odds(scores):
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def predict_bundle(bundle, frame):
    # Threads avoid spawning extra worker processes inside API/notebook kernels.
    with joblib.parallel_backend("threading", n_jobs=2):
        scores = bundle["pipeline"].predict_proba(frame[FEATURES])[:, 1]
    return bundle["calibrator"].predict_proba(log_odds(scores))[:, 1]


def train(data_path, output_dir, execution="local"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path, dtype={c: str for c in AUDIT + CATEGORICAL}, keep_default_na=False)
    if frame.encounter_id.duplicated().any():
        raise ValueError("Encounter IDs must be unique.")
    expected = frame.patient_nbr.map(patient_split)
    if not (expected == frame.split).all():
        raise ValueError("Split assignments do not match the patient-group contract.")
    # Numeric ordering makes equal-score capacity tie-breaking reproducible after BQ export.
    frame = frame.assign(_order=pd.to_numeric(frame.encounter_id)).sort_values("_order").drop(columns="_order").reset_index(drop=True)
    train_frame = frame[frame.split == "train"]
    validation = frame[frame.split == "validation"]
    test = frame[frame.split == "test"]
    if any(part.label.nunique() != 2 for part in [train_frame, validation, test]):
        raise ValueError("All partitions must contain both classes.")
    cv_rows, cv_summary = cross_validate_models(train_frame)
    cv_rows.to_csv(output_dir / "cross_validation.csv", index=False)
    (output_dir / "cross_validation_summary.json").write_text(json.dumps(cv_summary, indent=2))
    candidates = model_candidates()
    fitted, validation_metrics = {}, {}
    for name, estimator in candidates.items():
        print(f"Training {name} on {len(train_frame):,} encounters...", flush=True)
        pipeline = Pipeline([("features", preprocessor()), ("model", estimator)])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            pipeline.fit(train_frame[FEATURES], train_frame.label)
        fitted[name] = pipeline
        scores = pipeline.predict_proba(validation[FEATURES])[:, 1]
        validation_metrics[name] = metrics(validation.label, scores)
        print(f"Validation average precision: {validation_metrics[name]['average_precision']:.4f}", flush=True)
    original_names = ["logistic_regression", "xgboost"]
    workflow_choices = {
        "no_cv_no_bagging": max(original_names, key=lambda name: validation_metrics[name]["average_precision"]),
        "cv_without_bagging": max(original_names, key=lambda name: cv_summary[name]["average_precision"]["mean"]),
        "cv_with_bagging": max(cv_summary, key=lambda name: cv_summary[name]["average_precision"]["mean"]),
    }
    selected = workflow_choices["cv_with_bagging"]
    calibrated_bundles = {}
    for name, pipeline in fitted.items():
        calibration_scores = pipeline.predict_proba(validation[FEATURES])[:, 1]
        calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=SEED)
        calibrator.fit(log_odds(calibration_scores), validation.label)
        if calibrator.coef_[0, 0] <= 0:
            raise ValueError("Calibration reversed risk ranking; inspect model before exporting.")
        calibrated_bundles[name] = {"pipeline": pipeline, "calibrator": calibrator,
                                   "model_name": name, "features": FEATURES, "schema_version": 1}
    bundle = calibrated_bundles[selected]
    calibrator = bundle["calibrator"]
    joblib.dump(bundle, output_dir / "model.joblib")
    # Test data is first scored after model choice and calibration are frozen.
    test_metrics = {name: metrics(test.label, model.predict_proba(test[FEATURES])[:, 1]) for name, model in fitted.items()}
    calibrated = predict_bundle(bundle, test)
    test_metrics["selected_calibrated"] = metrics(test.label, calibrated)
    workflow_comparison = {
        workflow: {"selected_model": name,
                   "selection": "fixed validation average precision" if workflow == "no_cv_no_bagging" else "training-only five-fold patient-group CV average precision",
                   "test_metrics": metrics(test.label, predict_bundle(calibrated_bundles[name], test))}
        for workflow, name in workflow_choices.items()
    }
    (output_dir / "workflow_comparison.json").write_text(json.dumps(workflow_comparison, indent=2))
    prevalence = float(train_frame.label.mean())
    test_metrics["constant_training_prevalence"] = metrics(test.label, np.full(len(test), prevalence))
    scored = test[AUDIT + ["age", "label"]].copy()
    scored["predicted_risk"] = calibrated
    scored["selected_for_follow_up"] = top_mask(calibrated)
    scored.to_csv(output_dir / "test_predictions.csv", index=False)
    audit = subgroup_audit(test, calibrated)
    (output_dir / "subgroup_audit.json").write_text(json.dumps(audit, indent=2))
    sample = test.iloc[[0]][FEATURES].to_dict(orient="records")
    (output_dir / "sample_request.json").write_text(json.dumps({"instances": sample}, indent=2))
    (output_dir / "sample_prediction.json").write_text(json.dumps({"predictions": [{"readmission_risk": float(calibrated[0]), "model": selected}]}, indent=2))
    quality_file = Path(data_path).parent / "data_quality.json"
    if quality_file.exists():
        shutil.copy2(quality_file, output_dir / "data_quality.json")
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "execution": execution,
        "cloud_status": "not_applicable_local_run" if execution == "local" else "vertex_training_run",
        "selected_model": selected, "selection_metric": "five-fold patient-group CV mean average precision on training data",
        "cross_validation": cv_summary,
        "workflow_comparison": workflow_comparison,
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "features": FEATURES, "training_encounters": len(train_frame),
        "validation_encounters": len(validation), "test_encounters": len(test),
        "patient_overlap_between_splits": 0,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "calibration": {"method": "sigmoid on validation log odds", "slope": float(calibrator.coef_[0, 0]),
                        "intercept": float(calibrator.intercept_[0])},
        "limitations": [
            "Historical 1999–2008 data; results do not establish current clinical usefulness.",
            "Patient-group holdout, not temporal or hospital holdout; encounter dates and hospital IDs are unavailable.",
            "Only diabetes-associated encounters discharged home or with home services are included.",
            "The target is recorded early readmission, not necessarily avoidable or captured across all hospitals.",
            "Outreach prioritization is evaluated retrospectively; reduced readmissions are not demonstrated.",
            "Model selection used five-fold training-group CV; validation data was reserved for calibration.",
            "Group differences are descriptive and do not certify fairness; small samples are flagged.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    from .report import render
    render(output_dir)
    print(json.dumps({"selected_model": selected, "test": test_metrics["selected_calibrated"]}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/features.csv")
    parser.add_argument("--output", default=os.environ.get("AIP_MODEL_DIR", "artifacts"))
    parser.add_argument("--execution", default="local", choices=["local", "vertex"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        data_path = args.data
        if data_path.startswith("gs://"):
            from google.cloud import storage
            bucket, key = data_path[5:].split("/", 1)
            data_path = str(Path(temporary) / "features.csv")
            source_bucket = storage.Client().bucket(bucket)
            source_bucket.blob(key).download_to_filename(data_path)
            quality_key = f"{key.rsplit('/', 1)[0]}/data_quality.json" if '/' in key else "data_quality.json"
            quality_blob = source_bucket.blob(quality_key)
            if quality_blob.exists():
                quality_blob.download_to_filename(str(Path(temporary) / "data_quality.json"))
        output_dir = Path(temporary) / "output" if args.output.startswith("gs://") else Path(args.output)
        train(data_path, output_dir, args.execution)
        if args.output.startswith("gs://"):
            from google.cloud import storage
            bucket, prefix = args.output[5:].split("/", 1)
            client = storage.Client()
            for path in output_dir.iterdir():
                client.bucket(bucket).blob(f"{prefix.rstrip('/')}/{path.name}").upload_from_filename(str(path))
            print(f"Uploaded artifacts to {args.output}", flush=True)


if __name__ == "__main__":
    main()
