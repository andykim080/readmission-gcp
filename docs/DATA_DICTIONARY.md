# Feature table and prediction contract

One row represents an eligible diabetes-associated inpatient encounter at discharge. The original source covers stays of 1–14 days. Eligibility is restricted to discharge disposition 1, 6, or 8, as described in UCI's `IDS_mapping.csv`.

The prediction request must provide all 40 model inputs: eight numeric fields and 32 categorical fields. The API rejects additional keys, including outcomes and identifiers. Missing categorical values (`?`, empty text, or JSON null) are normalized to `__MISSING__`. Unknown categories are handled by the trained encoder. Numeric inputs must be finite nonnegative integers, and length of stay must be 1–14 days.

## Numeric model inputs

| Field | Meaning |
|---|---|
| `time_in_hospital` | Recorded days between admission and discharge |
| `num_lab_procedures` | Recorded number of laboratory procedures during the encounter |
| `num_procedures` | Recorded number of other procedures during the encounter |
| `num_medications` | Recorded number of medications during the encounter |
| `number_outpatient` | Recorded outpatient visits in the preceding year |
| `number_emergency` | Recorded emergency visits in the preceding year |
| `number_inpatient` | Recorded inpatient visits in the preceding year |
| `number_diagnoses` | Recorded number of diagnoses |

No numeric values are missing in the prepared local cohort. The transformation pipeline includes median imputation, fitted on training data only, and standardization.

## Categorical model inputs

| Field | Meaning / handling |
|---|---|
| `age` | Source age band in ten-year intervals |
| `admission_type_id` | Source admission type code; interpret with `IDS_mapping.csv` |
| `admission_source_id` | Source admission source code; interpret with `IDS_mapping.csv` |
| `discharge_disposition_id` | Discharge code; this cohort contains only 1, 6, and 8 |
| `diag_1`, `diag_2`, `diag_3` | First, second, and third recorded diagnosis codes; ICD-9 strings are preserved |
| `max_glu_serum` | Source glucose-result category; literal `None` means no recorded test |
| `A1Cresult` | Source HbA1c-result category; literal `None` means no recorded test |
| `change` | Source flag for a change in diabetes medication |
| `diabetesMed` | Source flag for diabetes medication prescribed |

Each of these 21 medication fields is also a categorical input. Values reflect the source medication-status categories, such as no prescription, steady dose, dose increase, or dose decrease:

`metformin`, `repaglinide`, `nateglinide`, `chlorpropamide`, `glimepiride`, `acetohexamide`, `glipizide`, `glyburide`, `tolbutamide`, `pioglitazone`, `rosiglitazone`, `acarbose`, `miglitol`, `troglitazone`, `tolazamide`, `insulin`, `glyburide-metformin`, `glipizide-metformin`, `glimepiride-pioglitazone`, `metformin-rosiglitazone`, `metformin-pioglitazone`.

The encoder groups categories occurring fewer than 25 times in its training portion. Within CV, it fits this grouping separately in each fold. Treating diagnosis codes as categories avoids implying a numeric ordering. This prototype does not convert them to ICD-10.

## Other prepared columns

| Field | Role |
|---|---|
| `encounter_id` | Audit key; used for stable ordering/tie-breaking, never as a predictor |
| `patient_nbr` | Audit/grouping key; all encounters from one patient remain in one partition; never as a predictor |
| `race` | Subgroup audit field; excluded from prediction inputs |
| `gender` | Source sex/gender category for subgroup audit; excluded from prediction inputs |
| `label` | 1 when source `readmitted` is `<30`; otherwise 0 for `>30` and `NO`; excluded from prediction inputs |
| `split` | Train when patient SHA256 bucket is 0–69; validation 70–84; test 85–99; excluded from prediction inputs |

The source `readmitted` field is never a model input. IDs and outcome values are not accepted by the prediction API.

## Source columns omitted

`weight`, `payer_code`, and `medical_specialty` are omitted from this first experiment because of substantial missingness and to keep scope focused. `examide` and `citoglipton` are omitted because they are constant in the original data. Neither excluding sensitive demographics nor grouping rare categories establishes fairness.

## Timing and outcome limitations

Prediction occurs at discharge, after the recorded diagnoses, tests, medications, and length of stay are available. These same features would not all be appropriate for an admission-time prediction. Encounter dates and hospital identifiers are not supplied, preventing temporal or hospital holdout evaluation. The outcome is recorded early readmission and does not guarantee complete capture of readmissions elsewhere or identify avoidable readmissions.

Source: [UCI dataset documentation](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), [dataset paper](https://doi.org/10.1155/2014/781670), and the downloaded mapping file. License: CC BY 4.0; attribution is included in the README.
