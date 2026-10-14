"""Render a self-contained evidence report from measured results."""
import html
import json
from pathlib import Path


def pct(value):
    return "—" if value is None else f"{value:.1%}"


def render(directory):
    directory = Path(directory)
    result = json.loads((directory / "metrics.json").read_text())
    quality_path = directory / "data_quality.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
    source_count = f"{quality['source_rows']:,}" if "source_rows" in quality else "Unrecorded number of"
    scores = result["test_metrics"]["selected_calibrated"]
    names = {"logistic_regression": "Logistic regression", "xgboost": "XGBoost", "bagged_trees": "Bagged decision trees",
             "selected_calibrated": "Selected model + calibration",
             "constant_training_prevalence": "Constant risk reference"}
    rows = "".join(
        f'<tr><td>{names[key]}</td><td>{value["average_precision"]:.3f}</td>'
        f'<td>{value["roc_auc"]:.3f}</td><td>{pct(value["precision_at_capacity"])}</td>'
        f'<td>{pct(value["recall_at_capacity"])}</td><td>{value["brier_score"]:.4f}</td></tr>'
        for key, value in result["test_metrics"].items()
    )
    limitations = "".join(f"<li>{html.escape(value)}</li>" for value in result["limitations"])
    workflow_names = {"no_cv_no_bagging": "No CV, no bagging", "cv_without_bagging": "CV, no bagging", "cv_with_bagging": "CV + bagging candidate"}
    workflow_rows = "".join(
        f'<tr><td>{workflow_names[key]}</td><td>{names[value["selected_model"]]}</td>'
        f'<td>{value["test_metrics"]["average_precision"]:.3f}</td>'
        f'<td>{pct(value["test_metrics"]["precision_at_capacity"])}</td>'
        f'<td>{pct(value["test_metrics"]["recall_at_capacity"])}</td></tr>'
        for key, value in result["workflow_comparison"].items())
    cv_rows = "".join(
        f'<tr><td>{names[key]}</td><td>{value["average_precision"]["mean"]:.3f} ± {value["average_precision"]["std"]:.3f}</td>'
        f'<td>{pct(value["recall_at_capacity"]["mean"])}</td><td>{value["brier_score"]["mean"]:.4f}</td></tr>'
        for key, value in result["cross_validation"].items())
    split_rows = "".join(f'<tr><td>{name.title()}</td><td>{info["encounters"]:,}</td>'
                         f'<td>{info["patients"]:,}</td><td>{pct(info["positive_rate"])}</td></tr>'
                         for name, info in quality.get("splits", {}).items())
    missing_rows = "".join(f'<tr><td>{html.escape(name)}</td><td>{value:.1f}%</td></tr>'
                           for name, value in sorted(quality.get("source_missing_percent", {}).items(),
                                                     key=lambda item: -item[1])[:5])
    text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Discharge follow-up | Evaluation report</title>
<style>
:root{{--ink:#143037;--muted:#526c73;--line:#d5e2e4;--accent:#087f75;--bg:#f3f7f7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:42px 28px 70px}}h1{{font-size:clamp(30px,5vw,46px);line-height:1.15;letter-spacing:-1.5px;margin:14px 0}}h2{{font-size:22px;margin-top:0}}
.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:2px;font-weight:750;color:var(--accent)}}.subtitle{{font-size:19px;max-width:850px;color:var(--muted)}}
.badge{{display:inline-block;padding:6px 12px;background:#dcefe9;color:#155c50;border-radius:8px;font-size:13px;font-weight:650}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:30px 0}}.card,section{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px}}
.number{{font-size:36px;font-weight:750;letter-spacing:-1px}}.label{{color:var(--muted);font-size:14px}}section{{margin-top:22px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:11px 10px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted);font-size:12px}}.tablewrap{{overflow-x:auto}}
.flow{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:18px 0}}.step{{padding:8px 14px;background:var(--bg);border-radius:8px;font-weight:600;font-size:14px}}a{{color:var(--accent)}}li{{margin-bottom:8px}}footer{{font-size:12px;color:var(--muted);margin-top:25px}}
@media(max-width:720px){{.cards,.grid{{grid-template-columns:1fr}}main{{padding:25px 16px}}.card,section{{padding:18px}}}}
</style></head><body><main>
<div class="eyebrow">Predictive analytics · Healthcare · GCP workflow</div>
<h1>Who should receive discharge follow-up first?</h1>
<p class="subtitle">A retrospective evaluation of 30-day readmission risk when an outreach team can contact 10% of eligible discharges.</p>
<span class="badge">{'Local model evaluation completed · GCP execution pending' if result['execution'] == 'local' else 'Vertex AI training completed · endpoint verification tracked separately'}</span>
<div class="cards">
<div class="card"><div class="label">Readmissions captured in the top 10%</div><div class="number">{pct(scores['recall_at_capacity'])}</div><div class="label">{scores['readmissions_in_top_capacity']:,} of {scores['positive_count']:,} test-set readmissions</div></div>
<div class="card"><div class="label">Recorded readmission among selected discharges</div><div class="number">{pct(scores['precision_at_capacity'])}</div><div class="label">Overall test-set rate: {pct(scores['positive_rate'])}</div></div>
<div class="card"><div class="label">Lift over random outreach, in expectation</div><div class="number">{scores['lift_at_capacity']:.2f}×</div><div class="label">Held-out test encounters: {scores['n']:,}</div></div>
</div>
<section><h2>What was built</h2><div class="flow"><span class="step">Public UCI data</span>→<span class="step">Feature preparation</span>→<span class="step">Patient-group split</span>→<span class="step">Model comparison</span>→<span class="step">Prediction API</span></div>
<p>The BigQuery ingestion and SQL feature workflow, Vertex AI training launcher, and deployment scripts are included in the project. Their presence does not establish that they have run in Google Cloud.</p></section>
<section><h2>Does cross-validation or bagging help?</h2><div class="tablewrap"><table><thead><tr><th>Workflow</th><th>Chosen model</th><th>Test average precision</th><th>Precision at 10%</th><th>Recall at 10%</th></tr></thead><tbody>{workflow_rows}</tbody></table></div>
<p class="label">All workflows share the same features, patient partitions, and calibration method. Cross-validation evaluates stability and selects a model; it does not automatically improve an unchanged model. Bagging adds a distinct model candidate. No test results were used to tune model settings. Point estimates do not establish statistically significant superiority.</p></section>
<section><h2>Five-fold cross-validation, training patients only</h2><table><thead><tr><th>Model</th><th>Average precision, mean ± SD</th><th>Mean recall at 10%</th><th>Mean Brier score</th></tr></thead><tbody>{cv_rows}</tbody></table>
<p class="label">Each fold fits its own preprocessing. Repeat encounters from one patient remain together. Fold standard deviation describes variability and is not a confidence interval.</p></section>
<section><h2>Model comparison on the reserved test group</h2><p>Model selection used five-fold cross-validation within the training patient group. The selected model was <strong>{names[result['selected_model']]}</strong>. Its probability calibration was fitted on validation data before scoring the test group.</p>
<div class="tablewrap"><table><thead><tr><th>Model</th><th>Average precision ↑</th><th>ROC AUC ↑</th><th>Precision at 10% ↑</th><th>Recall at 10% ↑</th><th>Brier score ↓</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="label">Average precision summarizes the precision–recall curve; the positive rate is its constant-score reference. Brier score measures probability error. Randomly selecting 10% would capture about 10% of readmissions in expectation. The constant risk row uses deterministic tie-breaking, so its capacity results can differ from that expectation.</p></section>
<div class="grid"><section><h2>Cohort and data quality</h2><p>{source_count} source encounters; {quality.get('eligible_rows',result['training_encounters']+result['validation_encounters']+result['test_encounters']):,} eligible home discharges. No patient appears in more than one group.</p>
<div class="tablewrap"><table><thead><tr><th>Group</th><th>Encounters</th><th>Patients</th><th>Readmission rate</th></tr></thead><tbody>{split_rows}</tbody></table></div>
<p class="label">Excluded: deaths, hospice, facility transfers, and other discharge dispositions. Race and sex are retained for auditing and excluded from prediction inputs. Age is an input.</p></section>
<section><h2>Missingness in the original data</h2><table><thead><tr><th>Field</th><th>Missing</th></tr></thead><tbody>{missing_rows}</tbody></table>
<p class="label">Weight, payer code, and medical specialty are omitted from this first model. “None” for HbA1c/glucose represents no recorded test and is preserved as a category; “?” is a missing-value marker.</p></section></div>
<section><h2>Calibration of predicted probabilities</h2><table><thead><tr><th>Mean predicted risk</th><th>Observed readmission rate</th></tr></thead><tbody>{''.join(f'<tr><td>{pct(row["predicted"])}</td><td>{pct(row["observed"])}</td></tr>' for row in scores['calibration'])}</tbody></table>
<p class="label">Test-set discharges are grouped into up to ten equal-count risk bins. Closer predicted and observed rates indicate better calibration.</p></section>
<section><h2>Limits of the evidence</h2><ul>{limitations}</ul><p>This prototype supports a learning and evaluation story. It is not validated for clinical deployment.</p></section>
<footer>Data: <a href="https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008">UCI Diabetes 130-US Hospitals (1999–2008)</a>, Clore, Cios, DeShazo &amp; Strack (2014), CC BY 4.0. Report generated: {html.escape(result['created_at_utc'])}. Execution: {html.escape(result['execution'])}.</footer>
</main></body></html>"""
    (directory / "report.html").write_text(text)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    render(parser.parse_args().artifacts)
