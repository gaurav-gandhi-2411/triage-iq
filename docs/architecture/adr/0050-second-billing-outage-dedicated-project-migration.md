# ADR-0050 — Second billing outage: migration off expense-tracker-498014 to a dedicated project

**Status:** Accepted, migration complete and live-verified (billing-status monitoring shipped; GCP Cloud Monitoring dashboard recreation tracked as follow-up, not done in this pass)
**Date:** 2026-08-12
**Decider:** Gaurav Gandhi

## Context

ADR-0038 (2026-08-05) migrated production off `triageiq-portfolio-495022` after its billing
account was closed, onto a co-tenant project (`expense-tracker-498014`) shared with an unrelated
product, under an IAM-scoped deploy identity with zero project-level grants — an explicitly
documented stopgap, not a permanent home.

Today (2026-08-12), `expense-tracker-498014`'s own billing was disabled too
(`gcloud billing projects describe expense-tracker-498014` → `billingEnabled: false`). This is
the second occurrence of the exact same failure class in eight days. Unlike the first outage —
undetected for up to 12 days because nothing watched `/health` on a schedule — this one was
caught within ~15 minutes, incidentally, by the eval-gate CI failing to download models from GCS
with a `403 ... billing account for the owning project is disabled` error while reviewing an
unrelated PR. Production itself was still serving (Cloud Run's warm revision doesn't need GCS
until its next cold start), but was one scale-to-zero cycle away from repeating the first
incident's outage shape.

GG's instruction: don't relink or debug the old project's billing — migrate to a dedicated
project under a new GCP identity (`gaurav.gandhi1129@gmail.com`, billing account
`01285B-91E4CB-70AD7E`) entirely, and this time add the monitoring that would have caught either
outage immediately instead of via CI coincidence.

## Decision

Migrate to `triageiq-prod-260812`, a fresh, dedicated (not co-tenant) project, replicating
ADR-0038's resource-scoped IAM discipline even though co-tenancy is no longer the reason for it —
least-privilege by default, not only when forced to by shared-project blast radius.

**What was provisioned:**

| Resource | Value |
|---|---|
| Project | `triageiq-prod-260812` |
| Billing account | `01285B-91E4CB-70AD7E` ("My Billing Account", open) — `billingEnabled: true` confirmed, no project-linking quota block (the cap that blocked GPU work in a prior session, per project memory, did not recur here) |
| APIs enabled | run, artifactregistry, storage, secretmanager, iamcredentials |
| Artifact Registry | `triageiq` (docker, us-central1) |
| GCS bucket | `gs://triageiq-prod-260812-models` — **not** `triageiq-models`: that name is globally taken (a `409` on creation, presumably by the old project's own bucket of the same name — GCS bucket names are global, not project-scoped). Project-scoped name chosen to avoid a repeat collision on any future migration. |
| Service accounts | `triageiq-deployer` (CI/deploy identity, WIF-only, no key ever created), `triageiq-api-runtime` (Cloud Run runtime identity) |
| IAM | Resource-scoped only, mirroring ADR-0038 exactly: `storage.objectViewer` on the bucket, `artifactregistry.writer` on the AR repo, `secretmanager.secretAccessor` on both secrets (deployer + runtime), `iam.serviceAccountUser` (deployer → runtime SA). `roles/run.developer` deferred until the Cloud Run service exists (see Consequences) — **zero project-level grants**, same as before. |
| WIF | New pool (`gh-actions-pool`) + OIDC provider (`gh-actions-provider`), bound to `triageiq-deployer`, scoped to `attribute.repository == 'gaurav-gandhi-2411/triage-iq'`. `GCP_WIF_PROVIDER` repo variable updated. |
| Secrets | `groq-api-key` recreated from local `.env` (unchanged value). `metrics-token` **rotated** — the old project's Secret Manager is unreadable under its own billing outage (`PERMISSION_DENIED`), so a fresh token was generated and synced to both the new project's Secret Manager and the `SMOKE_TEST_METRICS_TOKEN` GitHub Actions secret (must match for the deploy smoke test's `/metrics` check to pass). |

**Model artifacts: 11/11 verified byte-identical, before AND after upload.** Hashed all 11 files
against the committed `MANIFEST.sha256` before touching anything (all matched); uploaded to the
new bucket; downloaded them back and re-hashed (not trusting `gcloud storage`'s own transfer
checksum alone) — all 11 matched again. Same discipline as the 2026-08-05 migration.

**Code references updated** (repo-wide grep for `expense-tracker-498014` and the old bucket
name): `.github/workflows/{deploy,eval-gate,record-cassette}.yml`, `deploy/scripts/{setup_gcp,
upload_models}.sh`, `scripts/{setup_wif,setup_monitoring,publish_models,verify_model_manifest,
probe_label_anchoring_fix}.py`, `README.md` (non-URL sections). `health-monitor.yml`,
`scripts/11b_verify_priority_calibration.py`, and the `triage-iq-ui` repo (`.env.production`,
`package.json`'s codegen script, `README.md`) all hardcode the Cloud Run service URL directly,
which is project-number-derived and unknowable before the first deploy — updated in a follow-up
commit once the service exists.

**New test coverage:** `publish_models.py` and `verify_model_manifest.py` each carry an
`EXPECTED_PROJECT` hard-stop gate (added after a 2026-07-23 silent-project-drift incident) that
had **zero test coverage at all** until this migration — not even a check on the constant value.
`tests/test_gcp_project_guard.py` now exercises the actual failure path (wrong or empty active
project → `SystemExit(1)` with an actionable message), not just the string constant, per GG's
explicit instruction.

## Consequences

- **What changes:** production moves to a new project, new bucket name, new service accounts,
  new WIF pool, one rotated secret. The application code and its behavior are unchanged — this is
  an infrastructure-identity migration, not a feature or model change.
- **What becomes easier:** a project-linking quota block (the documented risk from a prior GPU
  work session) did NOT recur here — confirmed empirically, not assumed, closing that open
  question for future GCP work under this identity.
- **What becomes harder:** nothing structurally; the same resource-scoped IAM discipline as
  ADR-0038 carries forward.
- **The bootstrap sequencing gap, inherited from ADR-0038 and not novel to this migration:** a
  brand-new Cloud Run service cannot receive a resource-scoped IAM binding before it exists, so
  the very first deploy needs a broader-than-steady-state permission to create the service shell.
  ADR-0038's own `setup_gcp.sh` documents this as "run once manually... before this script's IAM
  step." Confirmed empirically here too: a `workflow_dispatch` test deploy from this branch (the
  documented, CI-driven "test before merging" path — rule 31a, never a local `--prod` deploy)
  failed exactly as predicted with `PERMISSION_DENIED: run.services.get` on the not-yet-existing
  service, since the deployer SA has no project-level Cloud Run grant to fall back on. Resolved
  the same way ADR-0038 documents: one manual `gcloud run deploy` using an Owner-level identity
  (`gaurav.gandhi1129@gmail.com`), deploying the EXACT image the CI run had already built and
  pushed from this same commit (not a separately-built local artifact) — creating the service
  shell with 100% traffic (no existing traffic split to preserve on a first deploy, so no
  `--no-traffic`/`--tag=candidate`). Immediately after, `roles/run.developer` was bound to
  `triageiq-deployer` scoped to the now-existing service, closing the bootstrap gap for every
  deploy after this one — the CI-driven `--no-traffic`/candidate/promote flow in `deploy.yml`
  is untouched and will work normally from here on.
- **Live-verified post-deploy** (`triageiq-api-00001-rc8`, 100% traffic,
  `https://triageiq-api-1014562031321.us-central1.run.app`): `/health` 200 with both repos
  loaded; `/metrics` 200 with the rotated `metrics-token`; real `/triage` calls for both repos —
  k8s ("kubectl exec hangs indefinitely") → `predicted_component=kubectl` at 57.6% confidence,
  5/5 grounded similar issues, zero fabrication; vscode ("Editor crashes on large JSON files") →
  `predicted_component=json` at 48.4% confidence, `resolution_confidence_pct=34.5%` (correctly
  under the product's own <40% low-confidence threshold, `resolution_model_beats_naive=false`,
  matching vscode's known naive-fallback behavior), 5/5 grounded, zero fabrication.

## Billing-status monitoring (the gap both outages shared)

Neither billing outage was caught by a control built to catch it — the first was undetected for
up to 12 days (ADR-0038's own finding: nothing watched `/health` on a schedule at all), the
second was caught by CI's GCS download failing incidentally, not by design. Per rule 85a (ask
what surface a control actually reaches): `/health` monitoring answers "is the service
responding," which is a downstream symptom of a billing outage, not the outage itself — a warm
Cloud Run revision can serve `/health` 200 for hours after billing is cut, exactly as observed
today, so a symptom-only check has a detection lag bounded by "however long until the next cold
start," which is unbounded for a low-traffic service. A direct billing-status check closes that
gap by checking the actual cause, not a downstream, delayable symptom.

**Shipped:** a new `billing-check` job in `.github/workflows/health-monitor.yml`, running
`gcloud billing projects describe triageiq-prod-260812 --format='value(billingEnabled)'` on the
same 30-min-nominal schedule, and failing loudly (GitHub's default scheduled-workflow-failure
email) if the value is ever not `True`. Authenticated via the same WIF pool already wired for CI,
but as a NEW, dedicated, minimal-privilege service account (`triageiq-billing-monitor`) rather
than granting `triageiq-deployer` any new permission — `roles/browser` (the smallest role that
includes `resourcemanager.projects.get`; `roles/billing.viewer` is not bindable at the project
level, confirmed via a rejected `INVALID_ARGUMENT`), keeping the deployer SA's own
zero-project-level-grants property intact.

**Not done in this pass, tracked as follow-up:** the GCP Cloud Monitoring uptime check +
alert-policy dashboard (`scripts/setup_monitoring.sh`) described as the *primary* outage detector
as of 2026-08-10 lived in `expense-tracker-498014` and has not been recreated in
`triageiq-prod-260812` — the script is updated to point at the new project but has not been
re-run. Until it is, `health-monitor.yml`'s two jobs (billing-check + the existing `/health`
check) are the only automated detection running, which is a real, disclosed gap relative to the
2026-08-10 state, not a silent one.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Relink `expense-tracker-498014`'s existing billing account | GG's explicit instruction: don't patch the project that's already failed this way once — migrate off it entirely. Also unverified whether the account itself or just the link was the problem; a fresh identity sidesteps diagnosing someone else's billing account state. |
| Reuse the `triageiq-models` bucket name | Globally taken (409 on creation) — likely by the old project's own bucket of the same name, which this migration doesn't have write access to reclaim (that project's own billing is disabled). A project-scoped bucket name avoids the same collision recurring on any future migration. |
| Read the old project's `metrics-token` value instead of rotating | Blocked — Secret Manager reads on `expense-tracker-498014` return `PERMISSION_DENIED` under its own billing outage, the same failure class as the GCS download failure that surfaced this whole incident. Rotating is strictly reversible (it's an internal auth token, not a customer-facing credential) and was synced to both ends (Secret Manager + GitHub secret) so nothing stays silently mismatched. |
