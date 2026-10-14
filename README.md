# Discharge follow-up prioritization on GCP

A retrospective predictive-modeling prototype for the question: **if a team can follow up with 10% of eligible discharges, how many recorded early readmissions can the model identify?**

## Start here

- `01_model_experiments.ipynb`: explained, runnable modeling workflow with saved outputs, five-fold patient-group CV, logistic regression, XGBoost, bagged trees, and the requested workflow comparison.
- `02_gcp_workflow.ipynb`: inspectable cloud stages and BigQuery SQL; does not create cloud resources when run as delivered.
- `03_model_improvements.ipynb`: regularization sweeps, inner-group early stopping, derived-feature experiments, and actual training-versus-CV diagnostics.
- `artifacts/report.html`: self-contained evaluation report.
- `GCP_SETUP.md`: account/project setup and cloud execution steps.
- `readmission/train.py`: the deployment-compatible source used to generate the notebook's training cells.

**Completed:** local model development and the GCP workflow ran successfully on October 14, 2026: BigQuery feature preparation and parity checks, Vertex AI training, model registration, a verified endpoint prediction, and undeployment. This is a retrospective learning prototype, with modest predictive performance.

## Verified cloud result

| Measure | Vertex run |
|---|---:|
| Eligible feature rows | 73,244 |
| Selected model | Bagged decision trees |
| Precision at top 10% | 25.14% |
| Recall at top 10% | 25.07% |
| Average precision | 0.20490 |
| ROC AUC | 0.65890 |
| Sample endpoint risk score | 7.87% |
| Endpoint after the test | Undeployed |

The sample score is one prediction, not an accuracy measure. Cloud training reproduced the original model-comparison workflow; the later tuning experiments remain development-only.

[Cloud results](artifacts/vertex/cloud_results.json) records the successful workflow status, sample prediction, and measured results. Cloud account and resource IDs are omitted from this public repository. [Provenance](artifacts/vertex/evidence_provenance.json): this local evidence was faithfully transcribed from the successful Cloud Shell output supplied by the project owner. The original file has not yet been downloaded or independently rechecked. `artifacts/metrics.json` remains the measured **local** run, so its execution type and metric values are not relabeled as cloud results.

## Files worth reviewing

1. [Model experiments notebook](01_model_experiments.ipynb): patient-group splits, CV, modeling, calibration, and measured results.
2. [Improvement experiments notebook](03_model_improvements.ipynb): 19 configurations, regularization, early stopping, class weights, and feature engineering.
3. [GCP workflow notebook](02_gcp_workflow.ipynb): BigQuery SQL and explicit cloud stages.
4. [Measured local report](artifacts/report.html) and [cloud completion evidence](artifacts/vertex/cloud_results.json).

Raw data, model binaries, virtual environments, credentials, and machine-specific cloud state are excluded from version control. Download the public UCI data and rerun the pipeline to regenerate them.

## Reproduce locally

Use Python 3.12 from this folder:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
bash scripts/local.sh prepare --download
bash scripts/local.sh train
bash scripts/local.sh test
```

Training runs 15 CV fits and three full-training fits. `scripts/local.sh` configures the bundled OpenMP library on macOS. Linux container dependencies are installed in the Dockerfile.

Register the notebook kernel within this environment:

```bash
.venv/bin/python scripts/configure_kernel.py --user
.venv/bin/python scripts/execute_notebooks.py
```

The execution helper launches the notebooks with this kernel and writes real outputs. In the delivered modeling notebook, `RUN_TRAINING=False` inspects saved artifacts; changing it to `True` and running all cells repeats the training workflow. In an interactive notebook editor, select **Readmission GCP (Python 3.12)** from the Jupyter kernel list, restart the kernel, and run all cells. This registered kernel includes the macOS OpenMP setting; selecting an unrelated Python 3.9.6 interpreter does not.

To test the local HTTP service:

```bash
bash scripts/local.sh serve
```

In a second terminal, from this project folder:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  --data-binary @artifacts/sample_request.json
```

## Method

- Dataset: UCI Diabetes 130-US Hospitals (1999–2008), 101,766 source encounters.
- Cohort: discharge disposition 1 (home), 6 (home health service), or 8 (Home IV provider): 73,244 encounters.
- Outcome: recorded readmission `<30` versus `>30` or `NO`.
- Scoring time: discharge; only features recorded by that time are used.
- Partitions: deterministic patient SHA256 buckets; roughly 70/15/15 train/validation/test. No patient crosses partitions.
- CV: five GroupKFold folds inside the training partition, identical across candidates. Preprocessing is fitted inside each fold.
- Baseline: regularized logistic regression.
- Other candidates: XGBoost and 100 bagged decision trees with bootstrap samples.
- Primary model-selection measure: mean CV average precision. The outreach-capacity metric is reported separately.
- Calibration: sigmoid fitted on validation log odds. The original no-CV workflow also selects its model on validation; the CV workflows reserve validation for calibration.
- Test: one common patient holdout for the three workflows, without test-driven parameter tuning. An initial two-model run preceded the requested CV/bagging expansion; its test results were not used to tune settings.
- Audit: age, race, and sex at one global outreach capacity; small groups flagged. Race and sex are excluded from model inputs, which does not by itself establish fairness.

## Measured local comparison

All rows use the same test group and calibrated probabilities. Point estimates do not establish statistically significant superiority.

| Selection workflow | Chosen model | Test average precision | Precision at 10% | Recall at 10% | Brier score |
|---|---|---:|---:|---:|---:|
| No CV, no bagging | Logistic regression | 0.2020 | 23.11% | 23.05% | 0.08666 |
| CV, no bagging | XGBoost | 0.2042 | 25.41% | 25.34% | 0.08638 |
| CV + bagging candidate | Bagged trees | 0.2049 | 25.05% | 24.98% | 0.08636 |

CV selects bagged trees by average precision, but XGBoost captures slightly more readmissions at 10% outreach capacity. This illustrates why the operational decision and model-selection metric both matter. CV is an evaluation and selection method; it does not automatically improve a fixed model. Bagging is a separate modeling method.

## Model-development follow-up

The improvements notebook shows 19 candidate configurations evaluated on the same five training-only patient-group folds. Weakening LR regularization worsened CV; stronger regularization at C=0.01 improved its mean average precision from 0.1984 to 0.2045. Adding coarse diagnosis and history features improved XGBoost from 0.2038 to 0.2050. A more complex depth-8 XGBoost increased training AP to 0.3980 but reduced CV AP to 0.1997. XGBoost class weighting using each fit group's negative/positive ratio reached 0.2057; original bagging remains slightly higher at 0.2058. No new candidate is a clear overall improvement.

Early stopping uses a separate inner patient group, then the selected tree count is refitted on all outer-fit patients. The outer fold never controls early stopping. These scores are development comparisons after repeated parameter checks, not an unbiased final estimate or clinical validation. Candidate bundles are saved separately and are uncalibrated; the original calibrated model remains the service default.

Reproduce the development sweeps:

```bash
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs" .venv/bin/python -m readmission.optimize
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs" .venv/bin/python -m readmission.optimize --followup --output artifacts/optimization_followup
DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs" .venv/bin/python -m readmission.optimize --imbalance --output artifacts/optimization_imbalance
.venv/bin/python scripts/execute_notebooks.py --notebook 03_model_improvements.ipynb
```

Alternatively set `RUN_OPTIMIZATION=True` in the improvements notebook and run all cells using the registered kernel. The shown original-model capacity reference reuses existing test predictions only for explanation; those outcomes do not select new candidate settings.

The imbalance experiments use LR `class_weight='balanced'`, XGBoost weights of 2 or the fold-specific class ratio, and optional ICD-9 250.x/428.x/585.x indicators in the three available diagnosis fields. These are limited code flags, not complete comorbidity phenotypes. Heavy class weighting worsened raw probability calibration despite small ranking gains. Weighted candidate probabilities need separate calibration before risk interpretation. No oversampling or target encoding is performed. Existing diagnosis groups are coarse chapter groups, not AHRQ CCS; formal CCS for ICD-9-CM is an appropriate separate experiment, while ICD-10 CCSR cannot be applied directly to these codes.

## Limits of evaluation

The data is historical and diabetes-specific. Hospital identifiers and encounter dates are unavailable, so temporal and hospital holdouts cannot be validated. The recorded outcome may not capture readmissions outside the contributing system and is not an avoidable-readmission label. No causal claim about outreach reducing readmissions is supported. The prototype has not been validated for clinical use.

## Evidence files

`artifacts/` contains measured metrics, CV fold results, a subgroup audit, data-quality results, sample predictions, an HTML report, and a saved preprocessing/model/calibration bundle. `cloud-state.json` will record actual cloud resource and job IDs when the cloud stages run. Endpoint evidence is only written after a real endpoint prediction succeeds. See `artifacts/vertex/` for the completed run and its source disclosure. Follow [backup and resource retirement](docs/HANDOFF.md) before removing cloud assets.

## Data attribution

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This project filters the discharge cohort, normalizes missing markers, and derives an outcome and patient partitions.

Raw source data, patient-level prediction files, model binaries, and credentials are excluded from Git by default. The public source can be downloaded again. Do not place credentials in notebooks.
