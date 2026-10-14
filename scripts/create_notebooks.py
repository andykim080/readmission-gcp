"""Build readable notebooks directly from the deployment training code."""
import ast
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "readmission/train.py").read_text()
tree = ast.parse(source)
lines = source.splitlines()


def function(name):
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return "\n".join(lines[node.lineno-1:node.end_lineno])


def md(value):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
body = textwrap.dedent("\n".join(lines[node.lineno:node.end_lineno-1]))
markers = ["    cv_rows, cv_summary", "    candidates =", "    original_names =", "    # Test data"]
markers = [marker.strip() for marker in markers]
parts = []
start = 0
for marker in markers:
    end = body.index(marker)
    parts.append(body[start:end].rstrip())
    start = end
parts.append(body[start:].replace("from .report import render", "from readmission.report import render"))

imports = source[:source.index("def preprocessor")].replace("from .evaluate", "from readmission.evaluate").replace("from .features", "from readmission.features")
notebook = nbf.v4.new_notebook()
notebook.metadata = {
    "kernelspec": {"display_name": "Readmission GCP (Python 3.12)", "language": "python", "name": "readmission-gcp"},
    "language_info": {"name": "python", "version": "3.12.14"},
}
notebook.cells = [
    md("""
    # Discharge follow-up prioritization: model experiments

    **Question:** If an outreach team can contact only 10% of eligible discharges, how many recorded early readmissions can a predictive model identify?

    This notebook walks through the actual project code and measured results, including logistic regression, XGBoost, and bagged decision trees. It compares a fixed validation split, patient-group cross-validation, and adding bagging.

    **Reading the saved run:** leave `RUN_TRAINING = False`. The notebook shows saved results from the real training run, without repeating 15 cross-validation fits. Set it to `True` and run every cell in order to train again.

    **Cloud status:** BigQuery preparation, Vertex training, a real endpoint prediction, and undeployment completed on October 14, 2026. This notebook retains the local evaluation outputs; cloud results and their provenance are saved separately in `artifacts/vertex/cloud_results.json` and `evidence_provenance.json`. See `02_gcp_workflow.ipynb` for the cloud stages.

    Data: [UCI Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), Clore, Cios, DeShazo & Strack (2014), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Historical diabetes-associated encounters; this is a retrospective learning prototype, not clinical validation.
    """),
    md("""
    ## 1. Environment and project paths
    Select **Readmission GCP (Python 3.12)** or this project's **`.venv/bin/python`** as the notebook kernel, restart it, and run all cells in order. The system Python 3.9.6 kernel does not match this project's dependencies. `scripts/configure_kernel.py` configures the project kernel, including the Mac OpenMP library. The introduction and other explanatory cells are Markdown cells, not Python cells.
    """),
    code("""
    from pathlib import Path
    import os
    import sys
    if sys.version_info < (3, 11):
        raise RuntimeError(f"Current kernel is Python {sys.version.split()[0]}. Select this project's .venv/bin/python (Python 3.12), restart the kernel, then Run All. Do not use the system Python 3.9.6 kernel.")
    ROOT = Path.cwd()
    if not (ROOT / "readmission").exists() and (ROOT / "readmission-gcp/readmission").exists():
        ROOT = ROOT / "readmission-gcp"
    assert (ROOT / "readmission").exists(), "Open the notebook from the readmission-gcp folder."
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    RUN_TRAINING = False
    data_path = Path("data/processed/features.csv")
    output_dir = Path("artifacts")
    execution = "local"
    print("Mode:", "new training run" if RUN_TRAINING else "inspect saved training results")
    print("Project folder:", ROOT)
    print("Kernel interpreter:", sys.executable)
    print("Python version:", sys.version.split()[0])
    """),
    code(imports + "\nimport matplotlib.pyplot as plt\nfrom IPython.display import display"),
    md("""
    ## 2. Inspect the source and define the cohort
    The outcome is `readmitted == '<30'`; `'>30'` and `'NO'` are negative. We score at discharge, so features recorded during the stay can be used. Only discharge dispositions **1 (home), 6 (home health), and 8 (Home IV provider)** are eligible. The original mapping file confirms these meanings.

    Patient and encounter IDs are used for grouping and audit only. Race and sex are retained for subgroup checks but are not prediction inputs. Age remains an input. Weight, payer code, medical specialty, and two constant medication columns are omitted.
    """),
    code("""
    from readmission.prepare import download, prepare
    raw_path = Path("data/raw/diabetic_data.csv")
    if not raw_path.exists():
        download(raw_path.parent)
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    display(raw[FEATURES + ["readmitted"]].head())
    print("Raw encounters:", len(raw))
    _, quality = prepare(raw_path, "data/processed")
    display(pd.DataFrame(quality["splits"]).T)
    """),
    md("""
    ## 3. Hold out patients, not random visits
    A deterministic SHA256 bucket assigns roughly 70% of patients to training, 15% to validation, and 15% to testing. The training group alone participates in cross-validation. The validation group calibrates probabilities; the original no-CV workflow also selects its model there. The same final test group evaluates all workflows.

    This is a patient-group split. It cannot establish performance on future years or unseen hospitals because encounter dates and hospital identifiers are not provided.
    """),
    code(parts[0] + "\ndisplay(pd.DataFrame({name: {'encounters': len(part), 'patients': part.patient_nbr.nunique(), 'early_readmission_rate': part.label.mean()} for name, part in frame.groupby('split')}).T)\nassert frame.groupby('patient_nbr')['split'].nunique().max() == 1\nprint('Patient overlap between partitions: 0')"),
    md("""
    ## 4. Define preprocessing and the three models
    Numeric counts are imputed and standardized. Categorical values are one-hot encoded; rare categories are grouped. Each cross-validation fold fits its own transformations on its training portion, avoiding preprocessing leakage.

    **Logistic regression:** the simple baseline, with regularization (`C=0.5`).

    **XGBoost:** sequential boosted trees, with fixed parameters for this first comparison.

    **Bagging:** 100 decision trees. Each tree sees a bootstrap sample containing 80% of training rows and a random 80% of encoded features; predicted probabilities are averaged. Tree depth and leaf size limit complexity. Bootstrap sampling happens inside the training fold; patient groups remain separated from fold holdout data.
    """),
    code(function("preprocessor")),
    code(function("model_candidates") + "\ndisplay(pd.DataFrame({'model': list(model_candidates()), 'estimator': [str(m) for m in model_candidates().values()]}))"),
    md("""
    ## 5. Five-fold patient-group cross-validation
    Each training patient appears in the holdout portion of exactly one fold. All three models use the same folds. We select using mean **average precision**, a summary of the precision–recall curve appropriate for the minority readmission outcome. Fold standard deviation describes variability; it is not a confidence interval.
    """),
    code(function("cross_validate_models")),
    code("if RUN_TRAINING:\n" + textwrap.indent(parts[1], "    ") + "\nelse:\n    cv_rows = pd.read_csv(output_dir / 'cross_validation.csv')\n    cv_summary = json.loads((output_dir / 'cross_validation_summary.json').read_text())\ndisplay(cv_rows[['model', 'fold', 'average_precision', 'roc_auc', 'recall_at_capacity', 'patient_overlap']])"),
    code("""
    cv_table = pd.DataFrame([
        {"model": name, "mean_average_precision": values["average_precision"]["mean"],
         "fold_sd": values["average_precision"]["std"],
         "mean_recall_at_10pct": values["recall_at_capacity"]["mean"]}
        for name, values in cv_summary.items()
    ])
    display(cv_table.round(4))
    """),
    md("""
    ## 6. Fit each candidate on the full training group
    These are the actual training commands used by the cloud training script. The saved-run branch shows the measured validation results. No class weighting or oversampling is applied in this first experiment, so probability interpretation remains straightforward.
    """),
    code("if RUN_TRAINING:\n" + textwrap.indent(parts[2], "    ") + "\nelse:\n    result = json.loads((output_dir / 'metrics.json').read_text())\n    validation_metrics = result['validation_metrics']\ndisplay(pd.DataFrame(validation_metrics).T[['average_precision', 'roc_auc', 'brier_score']])"),
    md("""
    ## 7. Select models and calibrate probabilities
    Compare three workflows on the same inputs and partitions:

    1. **No CV, no bagging:** choose logistic regression or XGBoost using fixed validation average precision.
    2. **CV, no bagging:** choose between those same models using training-only group CV.
    3. **CV + bagging candidate:** allow all three models and choose using that CV score.

    Calibration fits a sigmoid to each model's validation log odds. It adjusts probability estimates while preserving ranking. If adding bagging does not improve CV, the selection rule is allowed to keep another model. Test scores do not determine these choices.
    """),
    code(function("log_odds") + "\n\n" + function("predict_bundle")),
    code("if RUN_TRAINING:\n" + textwrap.indent(parts[3], "    ") + "\nelse:\n    workflow_choices = {key: value['selected_model'] for key, value in result['workflow_comparison'].items()}\n    bundle = joblib.load(output_dir / 'model.joblib')\n    selected = bundle['model_name']\nprint('Workflow choices:', workflow_choices)\nprint('Final saved model:', selected)"),
    md("""
    ## 8. Evaluate the reserved test group and save evidence
    **Recall at 10%:** fraction of recorded readmissions captured by selecting the highest-risk 10% of discharges.

    **Precision at 10%:** fraction of selected discharges that actually had a recorded early readmission.

    **Lift:** selected-group readmission rate divided by the overall test rate. Random outreach captures roughly 10% of readmissions in expectation.

    **Brier score:** probability prediction error; lower is better. ROC AUC and average precision summarize discrimination. Model comparisons are point estimates and do not demonstrate statistically significant superiority.

    The test outcome is historical; identifying risk does not establish that outreach would prevent a readmission. An initial two-model run preceded the user-requested CV/bagging expansion. No test results were used to tune settings.
    """),
    code("if RUN_TRAINING:\n" + textwrap.indent(parts[4], "    ") + "\nelse:\n    result = json.loads((output_dir / 'metrics.json').read_text())\n    workflow_comparison = result['workflow_comparison']\n    test_metrics = result['test_metrics']\nprint('Run timestamp:', result['created_at_utc'])\nprint('Execution environment:', result['execution'])"),
    code("""
    comparison = pd.DataFrame([
        {"workflow": workflow, "chosen_model": item["selected_model"],
         **{metric: item["test_metrics"][metric] for metric in
            ["average_precision", "roc_auc", "precision_at_capacity", "recall_at_capacity", "brier_score", "lift_at_capacity"]}}
        for workflow, item in workflow_comparison.items()
    ])
    display(comparison.round(4))
    print("Same test patients, features, capacity, and calibration method for every workflow.")
    display(pd.DataFrame(test_metrics).T[["average_precision", "roc_auc", "precision_at_capacity", "recall_at_capacity", "brier_score"]].round(4))
    """),
    md("""
    ## 9. Visual comparison and calibration
    The first chart shows variation across the five training folds. The second compares the three workflows on the same test group. A cross-validation choice can change the selected model, but CV does not change the prediction formula of a model that is otherwise identical.
    """),
    code("""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, part in cv_rows.groupby("model"):
        axes[0].plot(part.fold, part.average_precision, marker="o", label=name.replace("_", " "))
    axes[0].set(title="Patient-group CV: average precision", xlabel="Fold", ylabel="Average precision")
    axes[0].legend(fontsize=8)
    labels = ["No CV\\nno bagging", "CV\\nno bagging", "CV + bagging\\ncandidate"]
    axes[1].bar(labels, comparison.recall_at_capacity * 100, color=["#78959d", "#168b82", "#143e49"])
    axes[1].axhline(10, linestyle="--", color="#ad5e35", label="Random outreach (expected)")
    axes[1].set(title="Test readmissions captured at 10% capacity", ylabel="Recall (%)", ylim=(0, 40))
    axes[1].legend(fontsize=8)
    calibrated_rows = test_metrics["selected_calibrated"]["calibration"]
    axes[2].plot([r["predicted"] for r in calibrated_rows], [r["observed"] for r in calibrated_rows], marker="o", color="#168b82")
    axes[2].plot([0, .4], [0, .4], linestyle="--", color="gray")
    axes[2].set(title="Selected model: test calibration", xlabel="Mean predicted risk", ylabel="Observed readmission rate", xlim=(0,.4), ylim=(0,.4))
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
    """),
    md("""
    ## 10. Inspect subgroup results
    This uses one overall top-10% selection rule, rather than a separate threshold per group. Small groups are flagged. Race and sex were not inputs, but other features can still act as proxies; excluding those inputs does not establish fairness.
    """),
    code("""
    audit = pd.DataFrame(json.loads((output_dir / "subgroup_audit.json").read_text()))
    display(audit[["attribute", "group", "n", "positives", "selection_rate", "recall_at_global_capacity", "small_group"]].round(4))
    """),
    md("""
    ## 11. Check a real prediction request
    The API uses the saved model and the same preprocessing as training. It requires the full feature contract and rejects patient IDs, outcomes, negative counts, and discharges outside the cohort. This in-process check verifies local serving behavior; it is not a cloud endpoint call.
    """),
    code("""
    from fastapi.testclient import TestClient
    from readmission.serve import app
    request = json.loads((output_dir / "sample_request.json").read_text())
    display(pd.DataFrame(request["instances"]))
    os.environ["MODEL_DIR"] = str(output_dir)
    with TestClient(app) as client:
        response = client.post("/predict", json=request)
        assert response.status_code == 200, response.text
        print(json.dumps(response.json(), indent=2))
    """),
    md("""
    ## 12. Next: execute the Google Cloud workflow
    Open `02_gcp_workflow.ipynb` and `GCP_SETUP.md`. The cloud workflow loads the raw CSV into BigQuery, creates the same feature table using SQL, verifies parity, runs these models as a Vertex AI custom training job, deploys a prediction endpoint, records a successful request, and undeploys the model after the demo.

    Current evidence supports **local predictive modeling and evaluation**. Claim BigQuery or Vertex AI execution only after the corresponding real jobs and endpoint evidence have been recorded.
    """),
]
nbf.write(notebook, ROOT / "01_model_experiments.ipynb")

cloud_notebook = nbf.v4.new_notebook(metadata=notebook.metadata)
cloud_notebook.cells = [
    md("""
    # Google Cloud workflow: BigQuery → Vertex AI → prediction endpoint
    This notebook explains each cloud stage. **No cloud resources are created when it is run as delivered.** Start with `GCP_SETUP.md`, using a dedicated project with billing/trial credits linked. The project completed BigQuery preparation, Vertex training, endpoint prediction, and undeployment on October 14, 2026. Saved completion evidence and its provenance are in `artifacts/vertex/`.
    """),
    code("""
    import os
    from pathlib import Path
    import sys
    if sys.version_info < (3, 11):
        raise RuntimeError("Select the Readmission GCP (Python 3.12) notebook kernel, restart it, and run all cells.")
    from readmission.cloud import feature_sql
    PROJECT_ID = "replace-with-your-project-id"
    REGION = "us-central1"
    DATASET = "readmission_demo"
    BUCKET = f"{PROJECT_ID}-readmission-demo"
    IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/readmission-demo/readmission:v1"
    print("Replace PROJECT_ID before executing cloud stages.")
    """),
    md("""
    ## 1. Review the BigQuery feature SQL
    Raw columns are loaded as strings so missing markers and ICD-9 codes are preserved. Numeric features are explicitly cast. The same SHA256 patient bucket used locally is calculated in GoogleSQL. The ingest stage compares every prepared row and column against local preparation before Vertex training.
    """),
    code("print(feature_sql('readmission-demo-example', DATASET))"),
    md("""
    ## 2. Run the stages in Cloud Shell
    Follow `GCP_SETUP.md` for exact commands. Separate stages make the work inspectable:

    - `ingest`: load raw data, execute feature SQL, verify parity, export features to Cloud Storage.
    - `train`: submit the custom Vertex job that executes `readmission.train`, including five-fold group CV and bagging.
    - `collect`: confirm the job succeeded, then download measured metrics and the saved model.
    - `deploy`: register the cloud-trained model and deploy its custom prediction container.
    - `predict`: make an actual authenticated endpoint request and save the returned result.
    - `cleanup`: undeploy serving compute after the demo.

    The training job has a one-hour timeout and one CPU worker. Deployment uses one replica. Budget alerts notify you; they are not a hard spending cap.
    """),
    code("""
    commands = {
        "ingest": f"python -m readmission.cloud ingest --project {PROJECT_ID}",
        "train": f"python -m readmission.cloud train --project {PROJECT_ID} --image {IMAGE} --service-account readmission-train@{PROJECT_ID}.iam.gserviceaccount.com",
        "collect": f"python -m readmission.cloud collect --project {PROJECT_ID}",
        "deploy": f"python -m readmission.cloud deploy --project {PROJECT_ID} --image {IMAGE} --service-account readmission-serve@{PROJECT_ID}.iam.gserviceaccount.com",
        "predict": f"python -m readmission.cloud predict --project {PROJECT_ID}",
        "cleanup": f"python -m readmission.cloud cleanup --project {PROJECT_ID}",
    }
    for stage, command in commands.items():
        print(f"{stage}:\\n  {command}\\n")
    """),
    md("""
    ## 3. Inspect cloud evidence when it exists
    Successful BigQuery jobs, the Vertex training resource, and endpoint status are written to `cloud-state.json`. A verified prediction is saved in `artifacts/vertex/endpoint_prediction.json`. Credentials are not included in the project or upload package.
    """),
    code("""
    import json
    state_path = Path("cloud-state.json")
    if state_path.exists():
        print(json.dumps(json.loads(state_path.read_text()), indent=2))
    else:
        print("Cloud execution has not started. Complete the dedicated GCP project setup first.")
    """),
]
nbf.write(cloud_notebook, ROOT / "02_gcp_workflow.ipynb")
print("Created 01_model_experiments.ipynb and 02_gcp_workflow.ipynb")
