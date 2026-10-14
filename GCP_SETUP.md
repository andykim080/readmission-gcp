# Set up and run the Google Cloud stages

The local models are already evaluated. These steps establish actual BigQuery, Vertex AI training, and endpoint evidence. Cloud execution has not yet run.

## 1. Create a dedicated demo project

1. Open [Google Cloud Console](https://console.cloud.google.com/) yourself and sign in. The assistant's browser security check did not permit opening the console in this session.
2. If you are new to Google Cloud, check the [official free-trial page](https://cloud.google.com/free). Google currently offers eligible new customers $300 in credit. Complete signup and any billing/terms steps yourself; do not share passwords or card information here.
3. In the project selector, choose **New project**. Name it **Readmission Risk Demo**, and note the generated **project ID**. The project ID is different from the display name.
4. Ensure billing or trial credits are linked to this project. You do not need an organization for a personal demo project.
5. In **Billing → Budgets & alerts**, create a project-scoped budget, for example $10, with alert thresholds at 50%, 90%, and 100%. [Budget alerts are notifications, not a hard spending cap](https://docs.cloud.google.com/billing/docs/how-to/budgets).

The project uses one CPU training worker with a one-hour timeout and one serving replica for a short demo. It does not require a GPU. Undeploy the endpoint promptly after recording the prediction; stored data and images can still have small ongoing storage charges. Check the billing page rather than assuming trial credits guarantee every operation is free.

## 2. Use Cloud Shell to avoid installing the Google Cloud CLI on your Mac

For a single-command run after you upload the ZIP, unzip it and enter the project folder:

```bash
unzip readmission-gcp.zip
cd readmission-gcp
bash scripts/run-gcp.sh YOUR_ACTUAL_PROJECT_ID
```

This prepares the environment, loads BigQuery, builds the container, runs Vertex training, records a real endpoint prediction, and undeploys the model. Download `gcp-evidence.zip` afterward. If a stage fails, keep `cloud-state.json` and use the individual stage commands below to resume. The script attempts undeployment on endpoint-stage errors; if that fails or Cloud Shell disconnects, inspect Vertex AI Endpoints and undeploy any active demo model yourself. Code preparation alone is not completed cloud usage.

Open Cloud Shell using the terminal icon at the top of the console. Upload the supplied `readmission-gcp.zip` with its **Upload file** menu. The ZIP includes code and notebooks; public source data is downloaded again in Cloud Shell.

```bash
unzip readmission-gcp.zip
cd readmission-gcp
export PROJECT_ID='YOUR-ACTUAL-PROJECT-ID'
gcloud config set project "$PROJECT_ID"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m readmission.prepare --download
```

Use Python 3.11 or 3.12 for the pinned package versions. Cloud Shell uses your signed-in Google identity. If Python reports missing application-default credentials, run `gcloud auth application-default login` yourself and complete Google's authentication flow. No downloaded service-account keys are needed.

## 3. Review and run resource setup

The bootstrap script enables the required APIs and creates a private bucket, an image repository, and three service accounts in your dedicated project. Training gets object access to the demo bucket; serving gets read access; the build account gets source-read, repository-write, and build logging permissions. It does not make the bucket public or create credential keys.

```bash
bash scripts/bootstrap-gcp.sh
```

Run this using your personal project's owner identity. Organization-managed accounts may block API, billing, or service-account actions; inspect the error rather than granting unrelated broad roles.

## 4. Prepare the data in BigQuery

```bash
python -m readmission.cloud ingest --project "$PROJECT_ID"
```

This creates the `readmission_demo` dataset in `us-central1`, loads the source CSV, builds `features` with GoogleSQL, and checks every prepared value against the local version. Successful jobs and the parity result are recorded in `cloud-state.json`. BigQuery Console should show both `raw_encounters` and `features`.

## 5. Build one image for training and serving

```bash
export READMISSION_IMAGE="us-central1-docker.pkg.dev/${PROJECT_ID}/readmission-demo/readmission:v1"
gcloud builds submit . \
  --project="$PROJECT_ID" \
  --region=us-central1 \
  --config=cloudbuild.yaml \
  --substitutions="_IMAGE=${READMISSION_IMAGE}" \
  --gcs-source-staging-dir="gs://${PROJECT_ID}-readmission-demo/build-source" \
  --service-account="projects/${PROJECT_ID}/serviceAccounts/readmission-build@${PROJECT_ID}.iam.gserviceaccount.com"
```

Cloud Build runs a Linux Docker build, which avoids Mac/ARM image incompatibility. The image excludes datasets, model outputs, notebooks, and credentials. Build logs go to Cloud Logging.

## 6. Train on Vertex AI

```bash
python -m readmission.cloud train \
  --project "$PROJECT_ID" \
  --image "$READMISSION_IMAGE" \
  --service-account "readmission-train@${PROJECT_ID}.iam.gserviceaccount.com"
```

The job trains all three models, runs patient-group CV, compares the selection workflows, calibrates, and saves artifacts in Cloud Storage. Open **Vertex AI → Training → Custom jobs** to inspect the job and logs. The launcher records the job resource immediately; training completion is verified by the collection stage.

After the job finishes:

```bash
python -m readmission.cloud collect --project "$PROJECT_ID"
```

Collection only succeeds after `JOB_STATE_SUCCEEDED`. Downloaded results go to `artifacts/vertex/`; inspect them before deployment.

## 7. Deploy briefly, verify a request, and stop serving compute

```bash
python -m readmission.cloud deploy \
  --project "$PROJECT_ID" \
  --image "$READMISSION_IMAGE" \
  --service-account "readmission-serve@${PROJECT_ID}.iam.gserviceaccount.com"

python -m readmission.cloud predict --project "$PROJECT_ID"
python -m readmission.cloud cleanup --project "$PROJECT_ID"
```

Run cleanup even if the prediction fails. If deployment is interrupted, check Vertex AI's Endpoints page and undeploy any active model manually if needed. `cleanup` preserves the model registry entry, empty endpoint, and evidence while removing serving compute.

Take screenshots or record a short walkthrough of the BigQuery feature table, successful custom training job, and the actual prediction response. The JSON evidence is saved to `artifacts/vertex/endpoint_prediction.json`. Download the artifacts and `cloud-state.json` from Cloud Shell to keep them with this project.

## What you can accurately claim

- **Now:** built and evaluated a local predictive prototype with patient-group CV, bagging, calibrated risk scores, and serving checks.
- **After ingestion succeeds:** prepared model data in BigQuery using SQL and verified data quality/parity.
- **After cloud training succeeds:** trained and evaluated the models through a Vertex AI custom job.
- **After the endpoint request succeeds:** deployed and called a Vertex AI prediction endpoint.

Relevant official references: [custom training jobs](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-job), [custom prediction container requirements](https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements), and [BigQuery conversion functions](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/conversion_functions).
