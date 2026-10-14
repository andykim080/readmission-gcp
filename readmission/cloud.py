"""Explicit cloud stages; every actual resource/job is recorded as evidence."""
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .features import AUDIT, CATEGORICAL, NUMERIC

STATE = Path("cloud-state.json")


def save_state(**values):
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state.update(values)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    STATE.write_text(json.dumps(state, indent=2))


def feature_sql(project, dataset):
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project):
        raise ValueError("Use a valid GCP project ID.")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", dataset):
        raise ValueError("Invalid BigQuery dataset ID.")
    columns = [f"`{c}`" for c in AUDIT[:2]]
    columns += [f"COALESCE(NULLIF(NULLIF(`{c}`, '?'), ''), '__MISSING__') AS `{c}`" for c in AUDIT[2:]]
    columns += [f"CAST(`{c}` AS INT64) AS `{c}`" for c in NUMERIC]
    columns += [f"COALESCE(NULLIF(NULLIF(`{c}`, '?'), ''), '__MISSING__') AS `{c}`" for c in CATEGORICAL]
    columns += ["CAST(readmitted = '<30' AS INT64) AS label"]
    bucket = "MOD(CAST(CONCAT('0x', SUBSTR(TO_HEX(SHA256(patient_nbr)), 1, 8)) AS INT64), 100)"
    columns += [f"CASE WHEN {bucket} < 70 THEN 'train' WHEN {bucket} < 85 THEN 'validation' ELSE 'test' END AS split"]
    select = ",\n  ".join(columns)
    return f"""-- Prediction time: discharge. Outcome and identifiers are excluded by the trainer.
-- 1: home; 6: home health service; 8: Home IV provider (UCI IDS_mapping.csv).
CREATE OR REPLACE TABLE `{project}.{dataset}.features`
OPTIONS(description='Diabetes-associated home discharges, patient-group split, recorded 30-day readmission target') AS
SELECT
  {select}
FROM `{project}.{dataset}.raw_encounters`
WHERE discharge_disposition_id IN ('1', '6', '8');
"""


def ingest(args):
    from google.cloud import bigquery, storage
    from .prepare import prepare
    # Local quality checks run before any source rows are uploaded.
    local, quality = prepare(args.source, "data/processed")
    bq = bigquery.Client(project=args.project)
    dataset = bigquery.Dataset(f"{args.project}.{args.dataset}")
    dataset.location = args.region
    bq.create_dataset(dataset, exists_ok=True)
    with open(args.source, newline="") as stream:
        columns = next(csv.reader(stream))
    config = bigquery.LoadJobConfig(
        schema=[bigquery.SchemaField(c, "STRING") for c in columns],
        source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    with open(args.source, "rb") as stream:
        load = bq.load_table_from_file(stream, f"{args.project}.{args.dataset}.raw_encounters", job_config=config, location=args.region)
    load.result()
    sql = feature_sql(args.project, args.dataset)
    query = bq.query(sql, location=args.region)
    query.result()
    # Materialize BQ's feature result and verify full parity against local preparation.
    records = list(bq.query(f"SELECT * FROM `{args.project}.{args.dataset}.features`", location=args.region).result())
    import pandas as pd
    cloud = pd.DataFrame([dict(row) for row in records])
    pd.testing.assert_frame_equal(
        local.sort_values("encounter_id").reset_index(drop=True),
        cloud[local.columns].sort_values("encounter_id").reset_index(drop=True),
        check_dtype=False,
    )
    output = Path("data/processed/bigquery_features.csv")
    cloud[local.columns].to_csv(output, index=False)
    storage.Client(project=args.project).bucket(args.bucket).blob("data/features.csv").upload_from_filename(str(output))
    storage.Client(project=args.project).bucket(args.bucket).blob("data/data_quality.json").upload_from_filename("data/processed/data_quality.json")
    save_state(project=args.project, region=args.region, bucket=args.bucket,
               bigquery_dataset=f"{args.project}.{args.dataset}", ingest_job=load.job_id,
               feature_job=query.job_id, bigquery_local_parity="passed", feature_rows=len(cloud))
    print(f"BigQuery preparation and parity passed: {len(cloud):,} rows.")


def run_training(args):
    from google.cloud import aiplatform
    aiplatform.init(project=args.project, location=args.region, staging_bucket=f"gs://{args.bucket}")
    job = aiplatform.CustomJob(display_name="readmission-model-comparison", worker_pool_specs=[{
        "machine_spec": {"machine_type": "e2-standard-4"}, "replica_count": 1,
        "container_spec": {"image_uri": args.image,
                           "env": [{"name": "GOOGLE_CLOUD_PROJECT", "value": args.project}],
                           "command": ["python", "-m", "readmission.train"],
                           "args": ["--data", f"gs://{args.bucket}/data/features.csv",
                                    "--output", f"gs://{args.bucket}/model", "--execution", "vertex"]},
    }])
    job.submit(service_account=args.service_account, timeout=3600)
    save_state(training_job=job.resource_name, training_status="submitted", image=args.image,
               artifact_uri=f"gs://{args.bucket}/model")
    print(f"Training submitted: {job.resource_name}")


def collect(args):
    from google.cloud import aiplatform, storage
    state = json.loads(STATE.read_text())
    aiplatform.init(project=args.project, location=args.region)
    job = aiplatform.CustomJob.get(state["training_job"])
    status = job.state.name
    save_state(training_status=status)
    if status != "JOB_STATE_SUCCEEDED":
        print(f"Training state: {status}.", flush=True)
        if status in {"JOB_STATE_PENDING", "JOB_STATE_QUEUED", "JOB_STATE_RUNNING"}:
            raise SystemExit(2)
        print(f"Training needs attention: {job.error}", flush=True)
        raise SystemExit(1)
    target = Path("artifacts/vertex")
    target.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=args.project)
    for blob in client.list_blobs(args.bucket, prefix="model/"):
        if not blob.name.endswith("/"):
            blob.download_to_filename(str(target / Path(blob.name).name))
    client.bucket(args.bucket).blob("data/data_quality.json").download_to_filename(str(target / "data_quality.json"))
    from .report import render
    render(target)
    save_state(vertex_artifacts=str(target), training_status=status)
    print(f"Verified successful Vertex job; collected artifacts in {target}.")


def deploy(args):
    from google.cloud import aiplatform
    state = json.loads(STATE.read_text())
    if state.get("training_status") != "JOB_STATE_SUCCEEDED":
        raise SystemExit("Collect a successful Vertex training job before deploying.")
    if state.get("endpoint"):
        raise SystemExit("An endpoint is already recorded. Use predict or cleanup before redeploying.")
    aiplatform.init(project=args.project, location=args.region)
    model = aiplatform.Model.upload(
        display_name="readmission-home-discharge", artifact_uri=f"gs://{args.bucket}/model",
        serving_container_image_uri=args.image,
        serving_container_command=["python", "-m", "uvicorn", "readmission.serve:app", "--host", "0.0.0.0", "--port", "8080"],
        serving_container_predict_route="/predict", serving_container_health_route="/health",
        serving_container_ports=[8080], sync=True,
    )
    endpoint = aiplatform.Endpoint.create(display_name="readmission-demo-endpoint", sync=True)
    # Record the resource before deployment, so failures still leave a cleanup target.
    save_state(model=model.resource_name, endpoint=endpoint.resource_name, endpoint_status="created")
    model.deploy(endpoint=endpoint, machine_type="n1-standard-2", min_replica_count=1,
                 max_replica_count=1, traffic_percentage=100, service_account=args.service_account, sync=True)
    save_state(endpoint_status="deployed")
    print(f"Endpoint deployed: {endpoint.resource_name}. Run predict, then cleanup.")


def predict(args):
    from google.cloud import aiplatform
    state = json.loads(STATE.read_text())
    aiplatform.init(project=args.project, location=args.region)
    request = json.loads(Path("artifacts/vertex/sample_request.json").read_text())
    response = aiplatform.Endpoint(state["endpoint"]).predict(instances=request["instances"])
    evidence = {"endpoint": state["endpoint"], "deployed_model_id": response.deployed_model_id,
                "predictions": response.predictions, "verified_at_utc": datetime.now(timezone.utc).isoformat()}
    Path("artifacts/vertex/endpoint_prediction.json").write_text(json.dumps(evidence, indent=2))
    save_state(endpoint_status="prediction_verified")
    print(json.dumps(evidence, indent=2))


def cleanup(args):
    from google.cloud import aiplatform
    state = json.loads(STATE.read_text())
    aiplatform.init(project=args.project, location=args.region)
    endpoint = aiplatform.Endpoint(state["endpoint"])
    endpoint.undeploy_all(sync=True)
    # Keep registry model, empty endpoint, and evidence; undeployment stops serving compute.
    save_state(endpoint_status="undeployed")
    print("Model undeployed. Saved model and evidence remain available.")


def evidence(args):
    """Only produce completion evidence when every required stage has succeeded."""
    state = json.loads(STATE.read_text())
    if state.get("project") != args.project:
        raise SystemExit("Evidence belongs to a different project.")
    prediction_path = Path("artifacts/vertex/endpoint_prediction.json")
    if not (state.get("bigquery_local_parity") == "passed"
            and state.get("training_status") == "JOB_STATE_SUCCEEDED"
            and state.get("endpoint_status") == "undeployed" and prediction_path.exists()):
        raise SystemExit("BigQuery, successful training, verified prediction, and undeployment are required.")
    prediction = json.loads(prediction_path.read_text())
    if prediction.get("endpoint") != state.get("endpoint") or not prediction.get("predictions"):
        raise SystemExit("Prediction evidence does not match the recorded endpoint.")
    metrics = json.loads(Path("artifacts/vertex/metrics.json").read_text())
    if metrics.get("execution") != "vertex":
        raise SystemExit("Saved model metrics were not produced by Vertex training.")
    selected = metrics["test_metrics"]["selected_calibrated"]
    result = {
        "project_id": args.project, "region": state["region"], "feature_rows": state["feature_rows"],
        "bigquery_ingest_job": state["ingest_job"], "bigquery_feature_job": state["feature_job"],
        "bigquery_feature_parity": state["bigquery_local_parity"],
        "vertex_training_job": state["training_job"], "training_status": state["training_status"],
        "model": state["model"], "endpoint": state["endpoint"],
        "verified_prediction": prediction["predictions"], "endpoint_status": state["endpoint_status"],
        "trained_model": metrics["selected_model"],
        "test_metrics": {key: selected[key] for key in ["average_precision", "roc_auc", "precision_at_capacity", "recall_at_capacity"]},
        "scope": "Retrospective learning prototype; predictive quality and clinical limitations are documented.",
    }
    Path("artifacts/vertex/cloud_evidence.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sql", "ingest", "train", "collect", "deploy", "predict", "cleanup", "evidence"])
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--dataset", default="readmission_demo")
    parser.add_argument("--bucket")
    parser.add_argument("--source", default="data/raw/diabetic_data.csv")
    parser.add_argument("--image")
    parser.add_argument("--service-account")
    args = parser.parse_args()
    args.bucket = args.bucket or f"{args.project}-readmission-demo"
    if args.stage in {"train", "deploy"} and not (args.image and args.service_account):
        parser.error("train/deploy require --image and --service-account")
    if args.stage == "sql":
        print(feature_sql(args.project, args.dataset))
    else:
        {"ingest": ingest, "train": run_training, "collect": collect, "deploy": deploy,
         "predict": predict, "cleanup": cleanup, "evidence": evidence}[args.stage](args)


if __name__ == "__main__":
    main()
