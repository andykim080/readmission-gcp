"""Shared feature contract for local training, BigQuery, and serving."""
import hashlib

NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
CATEGORICAL = [
    "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult", *MEDICATIONS, "change", "diabetesMed",
]
FEATURES = NUMERIC + CATEGORICAL
AUDIT = ["encounter_id", "patient_nbr", "race", "gender"]
HOME_DISPOSITIONS = {"1", "6", "8"}
MISSING = "__MISSING__"
SEED = 42


def patient_split(patient_id):
    """SHA256 bucket is reproducible in Python and GoogleSQL; no IDs as features."""
    bucket = int(hashlib.sha256(str(patient_id).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def validate_instances(instances):
    """Require the full feature contract; reject IDs/outcomes and invalid numbers."""
    import math
    import pandas as pd
    if not instances or len(instances) > 1000:
        raise ValueError("Provide between 1 and 1000 discharge records.")
    rows = []
    for record in instances:
        missing = sorted(set(FEATURES) - set(record))
        extra = sorted(set(record) - set(FEATURES))
        if missing or extra:
            raise ValueError(f"Feature contract mismatch: missing={missing}, extra={extra}")
        row = {}
        for name in NUMERIC:
            value = record[name]
            if value is None or isinstance(value, bool):
                raise ValueError(f"{name} must be a nonnegative finite number.")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric.") from exc
            if not math.isfinite(number) or number < 0 or not number.is_integer():
                raise ValueError(f"{name} must be a nonnegative finite integer.")
            if name == "time_in_hospital" and not 1 <= number <= 14:
                raise ValueError("time_in_hospital must be between 1 and 14.")
            row[name] = number
        for name in CATEGORICAL:
            value = record[name]
            if value is not None and not isinstance(value, (str, int, float)):
                raise ValueError(f"{name} must be a scalar category.")
            row[name] = MISSING if value is None or str(value) in {"", "?"} else str(value)
        if row["discharge_disposition_id"] not in HOME_DISPOSITIONS:
            raise ValueError("This model only covers discharge home or with home services (1, 6, 8).")
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES)

