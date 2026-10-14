# Preserve the run and retire stored cloud assets

The completed endpoint was undeployed. Container images, Cloud Storage objects, and BigQuery tables remained after that operation. Their continued cost depends on current usage, free allowances, trial status, and billing terms. Billing being enabled does not verify available trial credits.

## Evidence provenance

The public `artifacts/vertex/cloud_results.json` contains aggregate metrics and workflow status transcribed from successful Cloud Shell output. The original `cloud_evidence.json` with resource IDs is kept outside this public repository. `evidence_provenance.json` records that the original file has not yet been downloaded. Do not describe this copy as independently retrieved or reverified.

The original cloud metrics, report, saved model, and state are in the Cloud Shell checkout. Cloud Shell can delete its home directory after 120 days without access: https://cloud.google.com/shell/docs/quotas-limits . GCP resources and this repository's saved transcript-derived evidence are separate records, but a local archive is still necessary.

## Download originals before deleting cloud assets

Upload `scripts/archive-cloud-evidence.sh`, `scripts/check-cloud-resources.sh`, and `scripts/retire-gcp-demo.sh` into the existing Cloud Shell checkout's `scripts/` folder if they are not present. From `~/readmission-gcp`:

```bash
bash scripts/archive-cloud-evidence.sh
```

Use the Cloud Shell menu's **Download** action to download `~/readmission-gcp/gcp-evidence.zip` and `~/readmission-gcp/gcp-evidence.zip.sha256`. Keep them outside Cloud Shell. The archive contains original evidence, full Vertex metrics, report, model binary, and cloud state.

On your Mac, verify with:

```bash
shasum -a 256 gcp-evidence.zip
unzip -t gcp-evidence.zip
```

Compare the checksum with the `.sha256` file. Keep model binaries and machine state out of the public repository. After downloading, replace transcript-derived evidence with the original file and update its provenance. Commit cloud report/metrics separately from local results.

## Inventory and retirement

Read-only inventory:

```bash
bash scripts/check-cloud-resources.sh YOUR_PROJECT_ID
```

Inspect Billing in the console for actual costs, credits, and expiration. The inventory cannot verify credits or certify that the entire project has no chargeable resources.

After downloading and verifying the archive locally, remove the demo image repository, BigQuery dataset, and Cloud Storage bucket:

```bash
bash scripts/retire-gcp-demo.sh --backup-downloaded YOUR_PROJECT_ID
```

This permanently deletes those specific resources. The script checks backup contents, confirms no model is deployed, and refuses to proceed if active custom training jobs remain. It keeps registered model metadata and the empty endpoint. Redeployment requires restoring the backed-up model and rebuilding a container.

Successful completion confirms these named resources were removed. It does not erase prior charges or audit every service in the project.

## Unused Site

An unfinished private Site named **Readmission Explorer** was registered but never published. At inspection it had no versions, live URL, or schedules. The available Sites connector has no Site deletion operation; delete it through the product's management UI if available. No dashboard publication was created.
