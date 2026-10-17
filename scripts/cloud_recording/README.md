# Cloud Run recording job (ADR-0065)

Runs `eval/record_cassettes.py --mode synthesis` as a Cloud Run job in `triageiq-prod-260812`, with the shared
Groq budget ledger and the recorder state in GCS. **Built and tested, not live**: nothing below has been run.

## Owner steps (in order; steps 1-2 are the only cloud changes)

1. **The one IAM binding** (the runtime SA gets create + read on `recorder/` of the models bucket only).
   Run in your own PowerShell, as the project owner:

   ```powershell
   gcloud storage buckets add-iam-policy-binding gs://triageiq-prod-260812-models `
     --account=gaurav.gandhi1129@gmail.com `
     --member="serviceAccount:triageiq-api-runtime@triageiq-prod-260812.iam.gserviceaccount.com" `
     --role="roles/storage.objectUser" `
     --condition='expression=resource.name.startsWith("projects/_/buckets/triageiq-prod-260812-models/objects/recorder/"),title=recorder-prefix-only,description=Recording job state under recorder/ only'
   ```

   Check: `gcloud storage buckets get-iam-policy gs://triageiq-prod-260812-models --account=gaurav.gandhi1129@gmail.com`
   must show the binding with the `recorder-prefix-only` condition and nothing else changed.
   Needs uniform bucket-level access, which the bucket has (verified 2026-10-12).
2. **Merge this PR** (it adds `.github/workflows/recording-job.yml`, which is deploy configuration, so it is
   yours to merge), then create the job once:
   `gh workflow run recording-job.yml --repo gaurav-gandhi-2411/triage-iq -f action=deploy-job`.
   It builds `docker/Dockerfile.recorder` on the current API image and runs `gcloud run jobs deploy`. No Groq
   token is spent.
3. **First execution, watched**: `gh workflow run recording-job.yml -f action=run-job`, then read the job log
   (`gcloud run jobs executions list --job triageiq-recorder --region us-central1 --project triageiq-prod-260812 --account=gaurav.gandhi1129@gmail.com`).
   Its last line is one JSON object: `event` (`recorded`, `skipped_budget` or `busy`), `recorder_exit`,
   `tokens_spent_this_run`, `ledger_published`, `cassette_bundle_published`.
4. **Check the bill once**: Billing > Reports, filter Cloud Run, after that first run. The free-tier fit for
   jobs is BELIEVED, not verified (ADR-0065, "Unverified gap"). If Cloud Run shows a charge, delete the
   schedule trigger (remove the `schedule:` block) before the next Monday.
5. **Use the result**: `python scripts/cloud_recording/pull_bundle.py --bucket triageiq-prod-260812-models`
   restores the latest cassette + checkpoint into your checkout if they extend your committed cassette. Review
   the diff and open a normal PR; the job never pushes to GitHub.

## What a run does and does not do

- Spends at most what the shared ledger allows: 160K of the 200K/day pool on `gpt-oss-120b`, never the 40K
  production reserve, and never past other projects' recorded spend. No budget left means `skipped_budget`, exit 0.
- Records about 26 issues per run at 6K tokens each (about 20 minutes); a full re-record spans several runs
  and days. `--max-retries 0` so a failure cannot re-spend tokens by itself.
- Never overwrites or deletes anything in GCS, never writes an eval baseline, never touches production traffic.
- The Groq key comes from Secret Manager (`groq-api-key`, already readable by the runtime SA).

## Files

| File | Purpose |
|---|---|
| `job_entrypoint.py` | the execution described above |
| `pull_bundle.py` | human step 5 |
| `../../tools/groq_budget/gcs_state.py` | append-only snapshot store + ledger merge |
| `../../docker/Dockerfile.recorder`, `../../requirements-recorder.txt` | the image |
| `../../.github/workflows/recording-job.yml` | `deploy-job` (manual) and `run-job` (weekly + manual) |
| `../../tests/test_cloud_recording.py`, `test_cloud_recording_packaging.py` | 17 tests: an in-memory bucket (no GCP) and the cost/safety guards in the workflow and Dockerfile |

## Not built

A laptop-side command that pulls and pushes the GCS ledger for other projects. Until it exists, laptop projects
keep their local `~/.groq_budget` ledger, and the job's spend reaches them only through Groq's own 429 (folded in
by `observe_429`).
