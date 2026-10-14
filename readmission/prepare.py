"""Download source data, define the cohort, audit quality, and persist splits."""
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from .features import AUDIT, CATEGORICAL, FEATURES, HOME_DISPOSITIONS, MISSING, NUMERIC, patient_split

SOURCE_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"


def download(raw_dir):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in ["diabetic_data.csv", "IDS_mapping.csv"]:
            member = next(n for n in archive.namelist() if Path(n).name == name)
            (raw_dir / name).write_bytes(archive.read(member))
    return raw_dir / "diabetic_data.csv"


def prepare(source, out_dir):
    source, out_dir = Path(source), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    required = set(AUDIT + FEATURES + ["readmitted"])
    if required - set(raw):
        raise ValueError(f"Source is missing columns: {sorted(required - set(raw))}")
    if raw.encounter_id.duplicated().any():
        raise ValueError("Duplicate encounter IDs found; resolve before modeling.")
    if (raw.patient_nbr.isin(["", "?"])).any():
        raise ValueError("Missing patient identifiers prevent reliable patient-group splitting.")
    if not raw.readmitted.isin(["<30", ">30", "NO"]).all():
        raise ValueError("Unexpected readmission labels found.")
    eligible = raw.discharge_disposition_id.isin(HOME_DISPOSITIONS)
    frame = raw.loc[eligible, AUDIT + FEATURES].copy()
    frame["label"] = (raw.loc[eligible, "readmitted"] == "<30").astype(int)
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if frame[col].isna().any() or (frame[col] < 0).any():
            raise ValueError(f"Invalid nonnegative count: {col}")
    if not frame.time_in_hospital.between(1, 14).all():
        raise ValueError("Length of stay outside the documented 1–14 day source cohort.")
    for col in CATEGORICAL + ["race", "gender"]:
        frame[col] = frame[col].replace({"?": MISSING, "": MISSING})
    frame["split"] = frame.patient_nbr.map(patient_split)
    split_patients = {name: set(part.patient_nbr) for name, part in frame.groupby("split")}
    if set(split_patients) != {"train", "validation", "test"}:
        raise ValueError("All three patient groups are required.")
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if split_patients[left] & split_patients[right]:
            raise ValueError("Patient overlap detected between partitions.")
    if any(part.label.nunique() != 2 for _, part in frame.groupby("split")):
        raise ValueError("Every partition must contain both target classes.")
    frame.to_csv(out_dir / "features.csv", index=False)
    quality = {
        "source_url": SOURCE_URL, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": len(raw), "eligible_rows": len(frame), "excluded_rows": int((~eligible).sum()),
        "unique_patients": int(frame.patient_nbr.nunique()),
        "cohort": "Discharge home (1), home with health service (6), or home under care of Home IV provider (8).",
        "target": "readmitted == '<30'; '>30' and 'NO' are the negative class",
        "positive_rate": float(frame.label.mean()),
        "patient_overlap_between_splits": 0,
        "source_missing_percent": {c: round(float(raw[c].isin(["?", ""]).mean()) * 100, 2) for c in raw},
        "splits": {name: {"encounters": len(part), "patients": int(part.patient_nbr.nunique()),
                          "positive_rate": float(part.label.mean())} for name, part in frame.groupby("split")},
        "checks": {"unique_encounter_ids": True, "valid_target_labels": True,
                   "valid_numeric_counts": True, "patient_groups_disjoint": True},
    }
    (out_dir / "data_quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps({k: quality[k] for k in ["source_rows", "eligible_rows", "positive_rate", "splits"]}, indent=2))
    return frame, quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/diabetic_data.csv")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download(Path(args.source).parent)
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()

