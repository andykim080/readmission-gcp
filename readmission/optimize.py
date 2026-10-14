"""Training-only model development with grouped CV and inner early stopping.

This module deliberately never scores the previously inspected test partition.
Development CV estimates after parameter search are not unbiased final estimates.
"""
import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .evaluate import metrics
from .features import AUDIT, CATEGORICAL, FEATURES, NUMERIC, SEED, patient_split
from .train import preprocessor

DERIVED_NUMERIC = ["prior_total_visits", "prior_acute_visits", "any_prior_inpatient",
                   "active_diabetes_medications", "changed_diabetes_medications", "medications_per_day",
                   "log_prior_inpatient", "log_prior_emergency", "log_prior_outpatient"]
DERIVED_CATEGORICAL = ["diag_1_group", "diag_2_group", "diag_3_group"]
CLINICAL_FLAGS = ["coded_diabetes_250", "coded_heart_failure_428", "coded_ckd_585"]


def diagnosis_group(value):
    value = str(value)
    if value.startswith("V"):
        return "supplementary"
    if value.startswith("E"):
        return "external_causes"
    try:
        number = float(value)
    except ValueError:
        return "missing_or_unmapped"
    if 250 <= number < 251:
        return "diabetes"
    boundaries = [(1, 140, "infectious"), (140, 240, "neoplasms"), (240, 280, "endocrine"),
                  (280, 290, "blood"), (290, 320, "mental"), (320, 390, "nervous_sensory"),
                  (390, 460, "circulatory"), (460, 520, "respiratory"), (520, 580, "digestive"),
                  (580, 630, "genitourinary"), (630, 680, "pregnancy"), (680, 710, "skin"),
                  (710, 740, "musculoskeletal"), (740, 760, "congenital"),
                  (760, 780, "perinatal"), (780, 800, "symptoms"), (800, 1000, "injury")]
    return next((label for low, high, label in boundaries if low <= number < high), "other")


def derive_features(frame):
    """Deterministic discharge-time features; no outcomes, IDs, or future information."""
    from .features import MEDICATIONS
    result = frame.copy()
    result["prior_total_visits"] = result.number_inpatient + result.number_emergency + result.number_outpatient
    result["prior_acute_visits"] = result.number_inpatient + result.number_emergency
    result["any_prior_inpatient"] = (result.number_inpatient > 0).astype(int)
    result["active_diabetes_medications"] = result[MEDICATIONS].isin(["Steady", "Up", "Down"]).sum(axis=1)
    result["changed_diabetes_medications"] = result[MEDICATIONS].isin(["Up", "Down"]).sum(axis=1)
    result["medications_per_day"] = result.num_medications / result.time_in_hospital.clip(lower=1)
    for name in ["inpatient", "emergency", "outpatient"]:
        result[f"log_prior_{name}"] = np.log1p(result[f"number_{name}"])
    for name in ["diag_1", "diag_2", "diag_3"]:
        result[f"{name}_group"] = result[name].map(diagnosis_group)
    diagnosis_numbers = result[["diag_1", "diag_2", "diag_3"]].apply(pd.to_numeric, errors="coerce")
    for name, code in zip(CLINICAL_FLAGS, [250, 428, 585]):
        result[name] = ((diagnosis_numbers >= code) & (diagnosis_numbers < code + 1)).any(axis=1).astype(int)
    return result


def feature_transform(spec):
    engineered = spec.get("engineered_features", False)
    flags = spec.get("comorbidity_flags", False)
    if not (engineered or flags):
        return preprocessor()
    numeric = NUMERIC + (DERIVED_NUMERIC if engineered else []) + (CLINICAL_FLAGS if flags else [])
    categorical = CATEGORICAL + (DERIVED_CATEGORICAL if engineered else [])
    transform = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=25), categorical),
    ], remainder="drop", sparse_threshold=1.0)
    return Pipeline([("derive", FunctionTransformer(derive_features, validate=False)), ("encode", transform)])


def followup_specs():
    return [
        {"name": "lr_C_0.01", "family": "lr", "C": 0.01, "early_stopping": False},
        {"name": "lr_C_0.1", "family": "lr", "C": 0.1, "early_stopping": False},
        {"name": "lr_engineered", "family": "lr", "C": 0.5, "early_stopping": False, "engineered_features": True},
        {"name": "xgb_engineered", "family": "xgb", "learning_rate": 0.05,
         "n_estimators": 300, "max_depth": 4, "min_child_weight": 20, "reg_lambda": 5,
         "early_stopping": False, "engineered_features": True},
    ]


def imbalance_specs():
    xgb_base = {"family": "xgb", "learning_rate": 0.05, "n_estimators": 300,
                "max_depth": 4, "min_child_weight": 20, "reg_lambda": 5, "early_stopping": False}
    return [
        {"name": "lr_balanced_C_0.01", "family": "lr", "C": 0.01, "early_stopping": False, "class_weight": "balanced"},
        {"name": "lr_balanced_C_0.1", "family": "lr", "C": 0.1, "early_stopping": False, "class_weight": "balanced"},
        {**xgb_base, "name": "xgb_positive_weight_2", "scale_pos_weight": 2.0},
        {**xgb_base, "name": "xgb_positive_weight_ratio", "scale_pos_weight": "ratio"},
        {**xgb_base, "name": "xgb_comorbidity_flags", "comorbidity_flags": True},
        {**xgb_base, "name": "xgb_weighted_comorbidities", "comorbidity_flags": True, "scale_pos_weight": "ratio"},
        {"name": "lr_comorbidity_flags", "family": "lr", "C": 0.01, "early_stopping": False, "comorbidity_flags": True},
    ]


def positive_class_weight(spec, fit):
    value = spec.get("scale_pos_weight", 1.0)
    if value == "ratio":
        positive = int(fit.label.sum())
        if not positive:
            raise ValueError("Cannot calculate class ratio without positive examples.")
        return (len(fit) - positive) / positive
    return float(value)


def candidate_specs():
    """A small, declared sweep rather than unbounded tuning."""
    specs = [
        {"name": f"lr_C_{c:g}", "family": "lr", "C": c, "early_stopping": False}
        for c in [0.5, 1.0, 10.0, 100.0]
    ]
    specs += [
        {"name": "xgb_original", "family": "xgb", "learning_rate": 0.05,
         "n_estimators": 300, "max_depth": 4, "min_child_weight": 20, "reg_lambda": 5,
         "early_stopping": False},
        {"name": "xgb_more_trees", "family": "xgb", "learning_rate": 0.03,
         "n_estimators": 1500, "max_depth": 4, "min_child_weight": 20, "reg_lambda": 5,
         "early_stopping": True},
        {"name": "xgb_flexible", "family": "xgb", "learning_rate": 0.03,
         "n_estimators": 1500, "max_depth": 6, "min_child_weight": 5, "reg_lambda": 1,
         "early_stopping": True},
        {"name": "xgb_deeper", "family": "xgb", "learning_rate": 0.03,
         "n_estimators": 1500, "max_depth": 8, "min_child_weight": 1, "reg_lambda": 1,
         "early_stopping": True},
    ]
    return specs


def estimator(spec, n_estimators=None, stopping=False, positive_weight=1.0):
    if spec["family"] == "lr":
        return LogisticRegression(C=spec["C"], class_weight=spec.get("class_weight"), max_iter=2000, random_state=SEED)
    options = dict(
        n_estimators=n_estimators or spec["n_estimators"],
        max_depth=spec["max_depth"], learning_rate=spec["learning_rate"],
        min_child_weight=spec["min_child_weight"], reg_lambda=spec["reg_lambda"],
        subsample=0.85, colsample_bytree=0.85, tree_method="hist",
        random_state=SEED, n_jobs=2, eval_metric="aucpr" if stopping else "logloss",
        scale_pos_weight=positive_weight,
    )
    if stopping:
        options["early_stopping_rounds"] = 50
    return XGBClassifier(**options)


def fit_candidate(spec, fit, seed=SEED):
    """Select the tree count using inner patients; refit using all fold-fit patients."""
    metadata = {"inner_patient_overlap": 0, "chosen_trees": None,
                "positive_class_weight": positive_class_weight(spec, fit),
                "inner_fit_patients": None, "inner_stop_patients": None}
    count = None
    if spec["early_stopping"]:
        inner_fit_index, stop_index = next(GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=seed
        ).split(fit, fit.label, groups=fit.patient_nbr))
        inner_fit, stop = fit.iloc[inner_fit_index], fit.iloc[stop_index]
        overlap = set(inner_fit.patient_nbr) & set(stop.patient_nbr)
        if overlap:
            raise ValueError("Patient overlap in early-stopping groups.")
        transform = feature_transform(spec)
        fit_x = transform.fit_transform(inner_fit[FEATURES])
        stop_x = transform.transform(stop[FEATURES])
        model = estimator(spec, stopping=True, positive_weight=positive_class_weight(spec, inner_fit))
        model.fit(fit_x, inner_fit.label, eval_set=[(stop_x, stop.label)], verbose=False)
        count = int(model.best_iteration + 1)
        metadata.update(chosen_trees=count, inner_fit_patients=int(inner_fit.patient_nbr.nunique()),
                        inner_stop_patients=int(stop.patient_nbr.nunique()))
    elif spec["family"] == "xgb":
        metadata["chosen_trees"] = spec["n_estimators"]
    pipeline = Pipeline([("features", feature_transform(spec)),
                         ("model", estimator(spec, n_estimators=count, positive_weight=positive_class_weight(spec, fit)))])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(fit[FEATURES], fit.label)
    return pipeline, metadata


def load_training(data_path):
    frame = pd.read_csv(data_path, dtype={col: str for col in AUDIT + CATEGORICAL}, keep_default_na=False)
    # Discard validation and test rows before any diagnostic or parameter comparison.
    frame = frame.loc[frame.split == "train"].copy()
    if frame.empty or frame.label.nunique() != 2 or frame.encounter_id.duplicated().any():
        raise ValueError("Invalid training partition.")
    if not (frame.patient_nbr.map(patient_split) == "train").all():
        raise ValueError("Training-only partition contract failed.")
    return frame.assign(_order=pd.to_numeric(frame.encounter_id)).sort_values("_order").drop(columns="_order").reset_index(drop=True)


def diagnostic_scores(frame, scores):
    values = metrics(frame.label, scores)
    return {key: values[key] for key in ["average_precision", "roc_auc", "precision_at_capacity", "recall_at_capacity", "brier_score"]}


def run_sweep(train_frame, output_dir, specs=None, n_splits=5):
    if "split" not in train_frame or not (train_frame.split == "train").all():
        raise ValueError("The optimization sweep accepts only the training partition.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = specs or candidate_specs()
    folds = list(GroupKFold(n_splits=n_splits).split(train_frame[FEATURES], train_frame.label,
                                                   groups=train_frame.patient_nbr))
    rows = []
    spec_lookup = {spec["name"]: spec for spec in specs}
    for spec in specs:
        for number, (fit_index, holdout_index) in enumerate(folds, start=1):
            fit, holdout = train_frame.iloc[fit_index], train_frame.iloc[holdout_index]
            if set(fit.patient_nbr) & set(holdout.patient_nbr):
                raise ValueError("Patient overlap in outer CV.")
            pipeline, metadata = fit_candidate(spec, fit, seed=SEED + number)
            train_scores = pipeline.predict_proba(fit[FEATURES])[:, 1]
            holdout_scores = pipeline.predict_proba(holdout[FEATURES])[:, 1]
            row = {"candidate": spec["name"], "family": spec["family"], "fold": number,
                   "outer_patient_overlap": 0, **metadata,
                   **{f"train_{key}": value for key, value in diagnostic_scores(fit, train_scores).items()},
                   **{f"cv_{key}": value for key, value in diagnostic_scores(holdout, holdout_scores).items()}}
            rows.append(row)
            print(f"{spec['name']} fold {number}/{n_splits}: training AP={row['train_average_precision']:.4f}, CV AP={row['cv_average_precision']:.4f}, trees={metadata['chosen_trees']}", flush=True)
            pd.DataFrame(rows).to_csv(output_dir / "optimization_folds.csv", index=False)
        # Persist summaries after each candidate so results survive an interrupted run.
        summary = summarize(pd.DataFrame(rows))
        summary.to_csv(output_dir / "optimization_summary.csv", index=False)
    selected = str(summary.sort_values("cv_average_precision_mean", ascending=False).iloc[0].candidate)
    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "selected_candidate": selected,
        "selected_spec": spec_lookup[selected], "selection_metric": "training-only group CV mean average precision",
        "training_encounters": len(train_frame), "training_patients": int(train_frame.patient_nbr.nunique()),
        "candidate_count": len(specs), "cv_folds": n_splits, "candidate_specs": specs,
        "evaluation_status": "development_cv_only_no_test_scoring",
        "limitations": ["Repeated parameter comparisons can overfit the development folds.",
                        "CV variability is not a confidence interval.",
                        "These comparisons do not establish clinically useful performance.",
                        "The original test results were previously inspected and are not an untouched final evaluation."],
    }
    (output_dir / "optimization_metadata.json").write_text(json.dumps(results, indent=2))
    return pd.DataFrame(rows), summary, results


def summarize(rows):
    columns = [col for col in rows if col.startswith(("train_", "cv_"))]
    summary = rows.groupby(["candidate", "family"])[columns].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary["ap_gap"] = summary.train_average_precision_mean - summary.cv_average_precision_mean
    return summary.sort_values("cv_average_precision_mean", ascending=False).reset_index(drop=True)


def export_candidate(train_frame, selected_spec, output_dir):
    """Save a development candidate without overwriting the original calibrated model."""
    pipeline, metadata = fit_candidate(selected_spec, train_frame)
    bundle = {"pipeline": pipeline, "model_name": selected_spec["name"], "features": FEATURES,
              "schema_version": 1, "calibration_status": "not_calibrated", "development_only": True,
              "fit_metadata": metadata}
    joblib.dump(bundle, Path(output_dir) / "candidate_uncalibrated.joblib")
    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/features.csv")
    parser.add_argument("--output", default="artifacts/optimization")
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--imbalance", action="store_true")
    args = parser.parse_args()
    train_frame = load_training(args.data)
    specs = imbalance_specs() if args.imbalance else followup_specs() if args.followup else None
    _, summary, result = run_sweep(train_frame, args.output, specs=specs)
    export_candidate(train_frame, result["selected_spec"], args.output)
    print(summary[["candidate", "train_average_precision_mean", "cv_average_precision_mean", "ap_gap"]].to_string(index=False))


if __name__ == "__main__":
    main()
