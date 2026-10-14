"""Create a visible experiment notebook from the optimization implementation."""
import ast
from pathlib import Path
import textwrap

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
source = (root / "readmission/optimize.py").read_text()
tree = ast.parse(source)
lines = source.splitlines()


def function(name):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(lines[node.lineno-1:node.end_lineno]).replace("from .features", "from readmission.features")


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


imports = source[:source.index("DERIVED_NUMERIC")].replace("from .evaluate", "from readmission.evaluate").replace("from .features", "from readmission.features").replace("from .train", "from readmission.train")
constants = source[source.index("DERIVED_NUMERIC"):source.index("def diagnosis_group")]
notebook = nbf.v4.new_notebook(metadata={
    "kernelspec": {"name": "readmission-gcp", "display_name": "Readmission GCP (Python 3.12)", "language": "python"},
    "language_info": {"name": "python", "version": "3.12.14"},
})
notebook.cells = [
    md("""
    # Improving the models: regularization, early stopping, and features

    This is the next experiment after `01_model_experiments.ipynb`. It tests whether the low scores reflect overly restrictive models, and compares actual training scores with scores on patients held out within CV.

    **Reading results:** leave `RUN_OPTIMIZATION=False`. The saved results compare 19 candidate settings × five patient-group folds, including regularization, early stopping, class weighting, and clinical flags. Change it to `True` and run all cells to reproduce those fits plus candidate exports. Early-stopping candidates also perform an internal training fit in each fold.

    **Evaluation:** only the original training partition is used for these experiments. No new predictions on the original test set are used for selection. Reusing these development folds for parameter comparisons means the winner's CV score is not an unbiased final performance estimate. The original bagging model remains the deployed-code default.
    """),
    md("""
    ## 1. Choose the correct kernel
    Select **Readmission GCP (Python 3.12)**, restart, and run all cells. This registered kernel includes the Mac OpenMP setting. Explanatory cells are Markdown, not Python.
    """),
    code("""
    import os, sys
    from pathlib import Path
    if sys.version_info < (3, 11):
        raise RuntimeError("Choose the Readmission GCP (Python 3.12) kernel, restart, then Run All.")
    ROOT = Path.cwd()
    if not (ROOT / "readmission").exists() and (ROOT / "readmission-gcp/readmission").exists():
        ROOT = ROOT / "readmission-gcp"
    assert (ROOT / "readmission").exists(), "Run from the readmission-gcp project folder."
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    RUN_OPTIMIZATION = False
    print("Python:", sys.version.split()[0])
    print("Mode:", "new optimization run" if RUN_OPTIMIZATION else "inspect measured optimization results")
    """),
    code(imports + "\nimport matplotlib.pyplot as plt\nfrom IPython.display import display\nfrom readmission.optimize import load_training, export_candidate"),
    code("""
    train_frame = load_training("data/processed/features.csv")
    print(f"Training only: {len(train_frame):,} encounters, {train_frame.patient_nbr.nunique():,} patients")
    assert set(train_frame.split) == {"train"}
    """),
    md("""
    ## 2. Check the scaling and correlation advice
    Numeric features already use `StandardScaler`, fitted only on the current fold's training patients. We do not add a second scaler. The correlation check below uses training patients only. Correlation alone is not a reason to remove a predictor; coefficient stability and predictive performance are separate questions.
    """),
    code("""
    correlation = train_frame[NUMERIC].corr()
    correlation_pairs = pd.DataFrame([
        {"feature_1": left, "feature_2": right, "correlation": correlation.loc[left, right]}
        for index, left in enumerate(NUMERIC) for right in NUMERIC[index+1:]
    ])
    display(correlation_pairs.assign(abs_r=lambda f: f.correlation.abs()).sort_values("abs_r", ascending=False).head(8))
    """),
    md("""
    ## 3. Declare the parameter comparisons
    For LR, increasing `C` weakens regularization. The first sweep checks C=0.5, 1, 10, 100. The follow-up also checks stronger regularization at C=0.1 and 0.01 because the weaker settings worsened CV.

    XGBoost experiments separate more trees/early stopping from additional flexibility. Early stopping uses up to 1,500 trees, learning rate 0.03, and patience 50, monitoring XGBoost's `aucpr`. Final comparisons use scikit-learn average precision; these are related but distinct measures. Depth 6/8 and smaller minimum child weight test more flexible rules.
    """),
    code(function("candidate_specs") + "\n\n" + function("followup_specs") + "\n\n" + function("imbalance_specs") + "\n\ndisplay(pd.DataFrame(candidate_specs() + followup_specs() + imbalance_specs()).fillna('—'))"),
    md("""
    ## 4. Test a small set of derived features
    The follow-up compares original features with additional visit-history totals, nonlinear logged visit counts, medication-status counts, medications per stay-day, and broad diagnosis groups. This first ablation tests them as a bundle; it does not identify which individual feature helped.

    Groups follow broad ICD-9 chapter ranges, with diabetes (250) separate. These are coarse modeling groups, not an ICD-10 conversion or a validated comorbidity index. [CMS chapter definitions](https://www.cms.gov/research-statistics-data-and-systems/statistics-trends-and-reports/bsapufs/downloads/2008_bsa_outpatient_procedure_puf_datadic_cbpdf), [CDC diagnosis-group reference](https://www.cdc.gov/nchs/data/ad/ad389.pdf).

    All derived information is available by discharge. Outcomes, identifiers, and future records are excluded.
    """),
    code(constants + "\n" + function("diagnosis_group") + "\n\n" + function("derive_features")),
    code("""
    # Use the canonical package function in serialized models so they load outside this notebook.
    from readmission.optimize import derive_features
    example = derive_features(train_frame[FEATURES].head())
    display(example[DERIVED_NUMERIC + DERIVED_CATEGORICAL + CLINICAL_FLAGS])
    """),
    code(function("feature_transform") + "\n\n" + function("positive_class_weight") + "\n\n" + function("estimator")),
    md("""
    ### Class imbalance and specific diagnosis flags
    LR's `class_weight='balanced'` is calculated from the current fitting data. XGBoost compares positive weight 1, 2, and the current fit group's negative/positive ratio (roughly 9). Early-stopping fits would use their own inner fitting ratio, not outer holdout outcomes. Weighting may improve discrimination, but does not automatically solve imbalance. Weighted raw probabilities need calibration before interpreting them as readmission risks.

    We also test flags indicating ICD-9 250.x (diabetes), 428.x (heart failure), and 585.x (CKD) in the three provided diagnoses. These flags are limited code indicators, not complete validated comorbidity phenotypes; absence of a code does not prove absence of a condition. All encounters are diabetes-associated, so a diabetes-code flag is not a general diabetes diagnosis test.

    No SMOTE or other oversampling is performed. Any future oversampling must be restricted to the fitting patients inside each fold. Target encoding would also need training-only fitting and group-aware cross-fitting. [LR class weights](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html), [XGBoost weights](https://xgboost.readthedocs.io/en/release_2.1.0/parameter.html).

    The current diagnosis grouping is a coarse chapter mapping, not CCS/CCSR. The appropriate external taxonomy for these ICD-9 data is [AHRQ CCS for ICD-9-CM](https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp); CCSR for ICD-10 should not be directly applied. An exact CCS mapping remains a separate experiment.
    """),
    md("""
    ## 5. Early stopping without using the evaluation patients
    For each outer CV fold:

    1. Set aside the outer fold's evaluation patients.
    2. Split the remaining patients into an inner fit group (85%) and a stopping group (15%).
    3. Fit preprocessing on inner fit patients only, then use the stopping group to choose the tree count.
    4. Refit preprocessing and the chosen number of trees on all outer-fit patients.
    5. Score the outer evaluation patients, who influenced neither preprocessing nor early stopping.

    Both group boundaries are checked for patient overlap. Other model settings are selected by comparing outer development CV, so this is not a full nested hyperparameter-search estimate.
    """),
    code(function("fit_candidate")),
    md("""
    ## 6. Run or inspect the five-fold experiment
    Training scores describe fit to already-seen patients. CV scores describe predictions on held-out patients. A large training-minus-CV gap is evidence that improved fit is failing to transfer. In-sample training scores are not evidence of clinical usefulness.
    """),
    code(function("diagnostic_scores") + "\n\n" + function("summarize") + "\n\n" + function("run_sweep")),
    code("""
    directories = [Path("artifacts/optimization"), Path("artifacts/optimization_followup"), Path("artifacts/optimization_imbalance")]
    if RUN_OPTIMIZATION:
        for specs, directory in zip([candidate_specs(), followup_specs(), imbalance_specs()], directories):
            fold_rows, summary, metadata = run_sweep(train_frame, directory, specs=specs)
            export_candidate(train_frame, metadata["selected_spec"], directory)
    fold_rows = pd.concat([pd.read_csv(d / "optimization_folds.csv") for d in directories], ignore_index=True)
    summary = pd.concat([pd.read_csv(d / "optimization_summary.csv") for d in directories], ignore_index=True)
    summary = summary.sort_values("cv_average_precision_mean", ascending=False).reset_index(drop=True)
    assert (fold_rows.outer_patient_overlap == 0).all()
    assert (fold_rows.inner_patient_overlap == 0).all()
    display(summary[["candidate", "train_average_precision_mean", "cv_average_precision_mean", "cv_average_precision_std", "ap_gap"]].round(4))
    """),
    md("""
    ## 7. Compare precision and recall at the same 10% capacity
    Average precision (AP) summarizes the full precision–recall curve; it is not the precision at one cutoff. These columns report the fraction readmitted among the highest-risk 10% (precision) and the fraction of all readmissions found there (recall). We retain the same 10% capacity for model comparisons.
    """),
    code("""
    display(summary[["candidate", "train_precision_at_capacity_mean", "train_recall_at_capacity_mean",
                     "cv_precision_at_capacity_mean", "cv_recall_at_capacity_mean"]].round(4))
    original = json.loads(Path("artifacts/metrics.json").read_text())
    reference = pd.DataFrame([
        {"candidate": name + " (original reference)",
         "cv_average_precision_mean": values["average_precision"]["mean"],
         "cv_precision_at_capacity_mean": values["precision_at_capacity"]["mean"],
         "cv_recall_at_capacity_mean": values["recall_at_capacity"]["mean"]}
        for name, values in original["cross_validation"].items()
    ])
    display(reference.round(4))
    best_new = summary.iloc[0]
    bagging_ap = original["cross_validation"]["bagged_trees"]["average_precision"]["mean"]
    print(f"Best new candidate: {best_new.candidate}, CV AP={best_new.cv_average_precision_mean:.4f}")
    print(f"Original bagging reference: CV AP={bagging_ap:.4f}")
    print("Small development-score differences do not establish a significant improvement.")
    """),
    md("""
    ## 8. Visualize what changed
    Higher training scores are helpful only when predictive performance transfers to held-out patients. The plots show why the more complex depth-8 model is not automatically a better candidate.
    """),
    code("""
    plt.style.use("seaborn-v0_8-whitegrid")
    shown = summary.sort_values("cv_average_precision_mean")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    y = np.arange(len(shown))
    axes[0].barh(y-.18, shown.train_average_precision_mean, height=.35, label="Training", color="#8aa4ae")
    axes[0].barh(y+.18, shown.cv_average_precision_mean, height=.35, label="Patient-group CV", color="#168b82")
    axes[0].set(yticks=y, yticklabels=shown.candidate, xlabel="Average precision", title="Better training fit can still generalize worse")
    axes[0].legend()
    axes[1].errorbar(shown.cv_average_precision_mean, y, xerr=shown.cv_average_precision_std,
                     fmt="o", color="#168b82", capsize=3)
    axes[1].axvline(bagging_ap, color="#ae5f36", linestyle="--", label="Original bagging mean")
    axes[1].set(yticks=y, yticklabels=shown.candidate, xlabel="CV average precision (mean ± fold SD)", title="Small changes relative to fold variation")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("artifacts/optimization/optimization_comparison.png", dpi=180, bbox_inches="tight")
    plt.show()
    """),
    md("""
    ## 9. Understand selection size separately from model quality
    This uses the original model's already-saved test predictions as an explanatory reference. It does not score a new candidate or tune its settings against test outcomes. Choosing more patients is a change in the outreach policy, not an improvement in the model. Keep capacity fixed when comparing models.
    """),
    code("""
    saved_test = pd.read_csv("artifacts/test_predictions.csv")
    capacities = []
    for capacity in [.05, .10, .20, .30, .50]:
        values = metrics(saved_test.label, saved_test.predicted_risk, capacity)
        capacities.append({"selected_fraction": capacity, "precision": values["precision_at_capacity"],
                           "recall": values["recall_at_capacity"], "selected": values["outreach_count"]})
    capacity_table = pd.DataFrame(capacities)
    display(capacity_table.round(4))
    capacity_table.to_csv("artifacts/optimization/original_model_capacity_reference.csv", index=False)
    """),
    md("""
    ## 10. Findings and useful next experiments
    **Initial findings before the imbalance experiment:** weaker LR regularization did not help; stronger regularization improved LR's development AP. More trees with early stopping modestly helped the original XGBoost setup. Depth-8 trees increased training fit but worsened CV. The first engineered-feature bundle produced a small XGBoost gain. Use the complete comparison table above to assess whether class weights or specific diagnosis flags improve these results.

    **Next suggestions:**
    - Test the derived features separately to find which ones help, instead of assuming the whole bundle is useful.
    - Test the omitted medical-specialty and payer fields with explicit missing categories. Missingness alone does not prove a field is useless; check predictive gains and subgroup behavior.
    - Plot learning curves using patient groups to see whether additional examples improve held-out performance.
    - If 10% outreach capacity is the actual decision, define that objective before another search; compare its recall alongside AP.
    - For a credible final result, obtain an independent dataset or use a properly nested evaluation; the inspected original test partition cannot become untouched again.

    Saved candidate files are **uncalibrated development models**. They are not installed into the original prediction service. The original calibrated model, comparison, and GCP scripts remain available.
    """),
]
nbf.write(notebook, root / "03_model_improvements.ipynb")
print("Created 03_model_improvements.ipynb")
