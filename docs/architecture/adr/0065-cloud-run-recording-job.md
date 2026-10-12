# ADR-0065: Run the Groq synthesis recorder as a Cloud Run job, with its state in GCS

Status: Proposed (built, NOT live: no cloud resource created, no Groq token spent). Date: 2026-10-12.

## Context

Re-recording the eval cassettes (synthesis, `eval/record_cassettes.py --mode synthesis`) ran on the laptop:
the unattended launcher, a private budget ledger in `~/.triageiq`, and a GPU-free but laptop-bound process
that has to survive 24-hour Groq quota waits. Three consequences, all seen in D28, D60 and D66: a recording
cannot proceed while the laptop sleeps; the budget it respects was invisible to every other consumer of the
same Groq pool (on 2026-10-11 a benchmark key spent about 300K tokens on the model production uses, and a
deploy smoke test then failed on a 429); and the recorder's state (cassette, checkpoint) lived in one working tree.

Groq facts that shape the design (VERIFIED 2026-10-12, https://console.groq.com/docs/rate-limits): the free
tier gives `openai/gpt-oss-120b` 8,000 tokens per minute and 200,000 tokens per day per organisation, as a
rolling 24-hour window, with a row per model id. One synthesis call costs about 6,000 tokens, so one run
can record at most about 26 issues, and a full 53-issue re-record needs at least two runs on different days.

## Decision

A scheduled Cloud Run **job** in the existing project `triageiq-prod-260812`, using the shared budget ledger
(`tools/groq_budget`, #178-#180) as its source of truth and GCS as its only state.

- **Image** (`docker/Dockerfile.recorder`): `FROM` the production API image plus one thin layer (recorder code,
  `eval/`, `tools/`, the GCS client). Artifact Registry stores shared layers once, so the 0.5 GB free allowance
  is not doubled.
- **Entrypoint** (`scripts/cloud_recording/job_entrypoint.py`), per execution: pull the latest ledger snapshot,
  claim the recording-active flag, restore the cassette bundle, check the shared budget, run the recorder
  (bounded by `RECORD_TIMEOUT_S`), then ALWAYS publish the merged ledger, publish a new cassette bundle if the
  cassette changed, and release the flag.
- **State in GCS, append-only** (`tools/groq_budget/gcs_state.py`): objects `recorder/ledger/NNNNNNNN.bin`
  and `recorder/cassette/NNNNNNNN.bin`. A snapshot is a NEW object written with `if_generation_match=0`
  (create only if absent); two writers racing for a sequence number cannot both win; the latest is the highest
  number. The ledger merge is a de-duplicated union of entries, so a lost race re-reads, merges and retries.
  Nothing is overwritten or deleted by the code.
- **Concurrency**: the recording-active flag lives in the ledger snapshot (owner, expiry). Claiming it is a
  create-only publish, so exactly one execution proceeds; a crashed one releases itself by expiry
  (`timeout + 10 min`). The job is also `--tasks 1 --parallelism 1` and the workflow has a concurrency group.
- **Cassette lineage**: a bundle records the sha256 of the cassette baked into the image it started from. It is
  restored only into an image with the same baked cassette, so a cassette committed to `main` after the last
  run is never clobbered by an older bundle (the job then starts from the committed one).
- **Output reaches the repo through a person**: `scripts/cloud_recording/pull_bundle.py` fetches the latest
  bundle into a checkout (same lineage rule); the diff is reviewed and committed in a normal PR. The job never
  pushes to GitHub and never writes a baseline.
- **Trigger**: `.github/workflows/recording-job.yml`, weekly (Mondays 04:17 UTC) plus manual dispatch, via the
  existing Workload Identity Federation identity `triageiq-deployer`. That identity already holds
  `roles/run.developer` and `iam.serviceAccountUser` on the runtime SA (read from the live IAM policies,
  2026-10-12), so Cloud Scheduler is not needed and its API stays off.
- **Identity**: the job runs as the existing `triageiq-api-runtime` SA, which already has `secretAccessor` on
  `groq-api-key`. The key reaches the job as a Secret Manager env var; it is never in the image, the repo
  or a log.

## Free-tier fit (zero-spend rule)

| Resource | Use per weekly run (BELIEVED, from the recorder's pacing) | Free allowance | Source |
|---|---|---|---|
| Cloud Run vCPU-seconds | 1 vCPU x ~1,200 s = 1,200 (about 5.2K per month) | 180,000 per month | https://docs.cloud.google.com/free/docs/free-cloud-features |
| Cloud Run GiB-seconds | 2 GiB x ~1,200 s = 2,400 (about 10.4K per month) | 360,000 per month | same |
| Cloud Storage | two small objects per run (about 2.5 MB), under 100 operations | 5 GB-months (US regions), 5,000 Class A, 50,000 Class B per month | same |
| Artifact Registry | one thin layer (a few MB) over the API image's shared layers | 0.5 GB per month | same |
| Cloud Scheduler | not used | n/a | n/a |
| Secret Manager | one access per run | not checked here | https://cloud.google.com/secret-manager/pricing |
| GitHub Actions | one `--wait` runner for at most 70 minutes per week | public-repo minutes are free (BELIEVED) | https://docs.github.com/billing |

Even at a daily cadence the compute is about 20% of the free allowance. **Unverified gap:** the Free Tier page
quoted above lists the Cloud Run allowance under request-based billing and does not mention jobs; Cloud Run's
pricing page (https://cloud.google.com/run/pricing) renders client-side and could not be read here. I BELIEVE
jobs draw on the same allowance. If they do not, the list-price worst case is roughly 3 cents per run (BELIEVED,
not verified), which still violates the rule. So the schedule ships weekly, `--max-retries 0`,
`--task-timeout 3300s`, and the first execution must be checked in Billing > Reports before the schedule is
left on.

## Consequences

- Recordings stop depending on the laptop and on one working tree; budget visibility is shared across every
  project that uses the ledger; the Groq key never leaves Secret Manager.
- One new IAM binding is required (queued for the owner, `scripts/cloud_recording/README.md`): the runtime SA
  needs create and read on `gs://triageiq-prod-260812-models/recorder/` only, as a conditional
  `roles/storage.objectUser` binding. The bucket has uniform bucket-level access (VERIFIED), which conditional
  bindings need. `objectUser` includes delete; the code never deletes, and the 7-day soft-delete policy on the
  bucket (VERIFIED) bounds a mistake.
- The runtime SA that serves production traffic gains write access to a prefix no serving code reads. A
  dedicated service account would be tighter but costs three more IAM changes; revisit if the job grows.
- The ledger in GCS only helps projects that sync it, and no laptop-side pull/push command exists yet (not
  built). Until then laptop projects keep their local ledger and learn of the job's spend only through Groq's
  own 429 (`observe_429`). The job always publishes its spend, so a later sync will see it.
- A recording is only as large as the day's headroom: with the benchmark key sharing the pool, many runs will
  exit "skipped_budget" until the benchmark adopts the ledger (review-iq #510) and the Oct-11 spend ages out.

## Alternatives considered

- **Cloud Scheduler to Cloud Run**: the canonical trigger, but it needs the Cloud Scheduler API enabled and a
  caller SA with `run.jobs.run` (another IAM binding). A weekly GitHub Actions schedule using the identity that
  already has the role needs neither.
- **Persistent disk or Filestore for state**: costs money; GCS snapshots are free at this size.
- **Overwrite a single `ledger.json` with generation preconditions**: needs `storage.objects.delete`-class
  permission for overwrites; append-only snapshots need only create and read.
- **A dedicated service account**: cleaner least-privilege; three IAM changes (create SA, secretAccessor,
  bucket binding) against one.
- **Run on the laptop but sync the ledger to GCS**: keeps the laptop dependency, which is the thing removed.
