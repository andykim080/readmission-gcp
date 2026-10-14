# Preserve the run and retire stored cloud assets

**Completed:** the owner supplied successful retirement output on October 14, 2026. The named demo repository, dataset, and bucket were removed. The instructions below are retained for reproducibility; do not rerun deletion against already removed resources.

The completed endpoint was undeployed. Container images, Cloud Storage objects, and BigQuery tables remained after that operation. Their continued cost depends on current usage, free allowances, trial status, and billing terms. Billing being enabled does not verify available trial credits.

## Evidence provenance

The project owner downloaded the original Cloud Shell archive. Its SHA256 matched the supplied checksum, ZIP integrity passed, and the state, completion evidence, endpoint response, and Vertex metrics were consistent. The full archive and cloud identifiers are preserved locally. Public cloud results omit resource identifiers, raw encounter predictions, model binaries, and machine state.

The endpoint was undeployed at the captured run's completion. Download verification does not establish current cloud state. The retirement script checks the endpoint and active training jobs before removing stored assets.

Cloud Shell can delete its home directory after 120 days without access: https://cloud.google.com/shell/docs/quotas-limits . Keep the downloaded archive on your computer.

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

Compare the checksum with the `.sha256` file. Keep model binaries and machine state out of the public repository. The current repository already includes verified original cloud metrics and the report, separately from the local results. Keep the downloaded originals for your records.

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

This permanently deletes those specific resources. The script verifies the backup SHA256 and contents, uses the Cloud Shell gcloud login for live checks, confirms no model is deployed, and refuses to proceed if active custom training jobs remain. It disables soft delete for new deletions on the demo bucket, then removes that bucket. Already soft-deleted objects are unaffected. It keeps registered model metadata and the empty endpoint. Redeployment requires restoring the backed-up model and rebuilding a container.

Successful completion confirms these named resources were removed. It does not erase prior charges or audit every service in the project.

## Unused Site

An unfinished private Site named **Readmission Explorer** was registered but never published. At inspection it had no versions, live URL, or schedules. The available Sites connector has no Site deletion operation; delete it through the product's management UI if available. No dashboard publication was created.
