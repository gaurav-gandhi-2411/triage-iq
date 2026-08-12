# 0038 — Serving-stack migration off closed-billing project, co-tenanting in expense-tracker-498014

**Superseded (2026-08-12) by ADR-0050.** The co-tenancy this ADR designed turned out to have a
real, materialized failure mode this ADR did not anticipate: on 2026-08-12, a parallel session
running an unrelated GCP account teardown swept and soft-deleted `expense-tracker-498014` — its
criticality (`hosts-prod: triageiq`) was recorded only in this document, invisible from the
project's name, its own console listing, or any label, so nothing stopped it from being treated
as an inert side-project and deleted. Recovered via `gcloud projects undelete` within the 30-day
window, no data lost, but this is exactly the co-tenancy risk made concrete rather than
theoretical. Production has since moved to a dedicated project (`triageiq-prod-260812`, ADR-0050)
for exactly this reason. `expense-tracker-498014` is retained temporarily as a labeled
(`do-not-delete: true`) fallback, not decommissioned as of this note.

## Context

TriageIQ's original GCP project, `triageiq-portfolio-495022`, had its billing account
(`01BAF1-D0984B-2D23F2`) closed by Google sometime between 2026-07-24 (the classifier
multi-label cutover, ADR-0036, last confirmed-live traffic) and 2026-08-05 (discovered). Cloud
Run scale-to-zero meant no traffic hit the service in that window to surface the failure
earlier — production was down for an unknown but possibly ~12-day window, undetected, because
nothing was watching `/health` on a schedule. Discovered when a routine Groq TPD headroom check
tried to trigger `record-cassette.yml` and it failed at "Download production models from GCS"
with `AccessDeniedException: ... billing account for the owning project is disabled in state
closed`. GCS reads, Cloud Run cold starts, and Secret Manager access were all blocked project-
wide; `gcloud billing accounts list` confirmed the linked account no longer even appears as an
account GG's identity can see (closed, not just unlinked).

GG does not want to relink `triageiq-portfolio-495022` — that billing account is gone. The
alternative was migrating the serving stack (Cloud Run service, GCS model bucket, Artifact
Registry image repo, service accounts, WIF) to a project already linked to a working billing
account (`014DAE-6B3556-077365`).

## Decision

**Migrate to `expense-tracker-498014`** — an existing, already-billed, personal-finance-tracker
project, chosen over three alternatives after a survey (Cloud Run services, buckets, IAM) of
every project linked to the same billing account:

| Candidate | Region | Existing footprint | Why not chosen |
|---|---|---|---|
| `review-iq-prod` | asia-south1 | 2 services, already hosts an unrelated AetherArt retrain bucket | Explicitly another product's *prod* project; already shows loose tenancy hygiene |
| `agentic-travel-booking-system` | asia-south1 | prod+staging pair of its own | Live prod/staging split of its own, cross-region |
| `iconic-reactor-496423-m4` | asia-south1 | 5 services (StyleMaitri sub-products) | Busiest of the four, cross-region |
| **`expense-tracker-498014`** | **us-central1** | **2 services, boring buckets only** | **Chosen** — smallest footprint, no foreign product's data already co-mingled, same region as TriageIQ's existing Cloud Run/Artifact Registry/WIF setup (zero cross-region latency or new-region AR repo needed) |

**A brand-new dedicated project was ruled out, not preferred-but-blocked-so-settled-for-second-
best**: `gcloud billing projects link` against a freshly created throwaway project hit the same
`Cloud billing quota exceeded` cap the 2026-07-24 GPU investigation already hit on this billing
account (`FAILED_PRECONDITION`, `QuotaFailure` on `billingAccounts/014DAE-6B3556-077365`). GG is
filing a self-service quota-increase request separately (not blocking this migration); if/when
that clears, the intent is to move to a dedicated project rather than staying co-tenant
indefinitely. **This migration is a stopgap, not the target end state.**

### IAM scoping — the specific thing that makes co-tenancy safe

`expense-tracker-498014` is not TriageIQ's project; it runs `expense-tracker` and
`agentgauge-judge` for an unrelated product. The core risk of co-tenanting is a TriageIQ-scoped
service account getting broad enough access to disturb those services. Concretely avoided:

- `triageiq-deployer@expense-tracker-498014.iam.gserviceaccount.com` (the WIF-authenticated CI
  deploy identity) holds **zero project-level IAM roles** — verified via
  `gcloud projects get-iam-policy expense-tracker-498014 --filter="bindings.members:triageiq-deployer@..."`
  returning empty. The original design (still reflected in this repo's history before this ADR)
  granted `roles/run.admin` at the PROJECT level; that would have let this SA manage or delete
  `expense-tracker`/`agentgauge-judge` too. Replaced with 5 resource-scoped bindings:
  - `roles/storage.objectViewer` on bucket `triageiq-models` only (not `roles/storage.objectViewer`
    project-wide, which would read every bucket in the project).
  - `roles/artifactregistry.writer` on Artifact Registry repo `triageiq` only.
  - `roles/run.developer` (not `roles/run.admin`) on Cloud Run service `triageiq-api` only —
    `run.developer` excludes `run.services.setIamPolicy`/`run.services.getIamPolicy`, so even a
    fully compromised deploy SA cannot change who can administer the service, let alone touch
    any other service in the project.
  - `roles/iam.serviceAccountUser` on the dedicated runtime SA
    `triageiq-api-runtime@expense-tracker-498014.iam.gserviceaccount.com` only — not on
    `expense-tracker`'s own default compute SA, and not project-wide `serviceAccountUser` (which
    would let it `actAs` any SA in the project, a real privilege-escalation surface in a shared
    project).
  - `roles/secretmanager.secretAccessor` on the two secrets it needs (`groq-api-key`,
    `metrics-token`) only.
- A **dedicated runtime SA** (`triageiq-api-runtime`) was created instead of reusing
  `expense-tracker`'s default compute SA (`242393598566-compute@developer.gserviceaccount.com`,
  what the original `triageiq-api` service used in the old project) — keeps TriageIQ's runtime
  identity, and therefore its exact permission surface, fully separate from anything
  expense-tracker runs under.
- The bucket is named `triageiq-models`, not project-id-derived — unambiguous ownership in a
  shared project, unlike the old `triageiq-portfolio-495022-models` naming which baked the
  (now-dead) project id into the name.
- First deploy was done manually (as project Owner) from a clean `git archive` of `origin/main`,
  specifically so `triageiq-deployer` never needed a project-level `run.services.create` grant
  even temporarily during bootstrap — every grant it holds today is exactly what ongoing CI
  redeploys need, nothing broader was ever issued and later trimmed.

### What moved

- Cloud Run service `triageiq-api` → new URL `https://triageiq-api-242393598566.us-central1.run.app`
  (project number changed; Cloud Run URLs are project-number-derived, not stable across a
  migration — every hardcoded reference, including the frontend's `VITE_API_BASE_URL` and a
  hardcoded API-docs link in `triage-iq-ui`'s `App.tsx`, had to be updated).
- GCS bucket → `gs://triageiq-models` (was `gs://triageiq-portfolio-495022-models`). All 11
  production artifacts (2 classifiers, 2 resolution predictors, cqr adjustments, 2 dup-index
  FAISS indices, 2 processed parquets) recovered from the local dev checkout and verified
  byte-identical against the committed `MANIFEST.sha256` before upload — the GCS objects
  themselves were never downloadable post-outage (403, same billing block), but nothing was
  actually lost since local copies existed. One pre-existing, already-documented manifest
  mismatch (`cqr_conformal_adjustments.json`, flagged in the ADR-0036 cutover commit as
  predating that session) persists unchanged in the new bucket too — not a regression from this
  migration.
- Artifact Registry repo `triageiq` (docker) recreated fresh; production image rebuilt from a
  clean `origin/main` tree (not the working dev checkout, to avoid baking in dev-only model
  variants like `*_PRE_MULTILABEL.pkl`/`*_multilabel_staged.pkl` that the original
  `COPY data/models/component_classifier_*.pkl` glob in `docker/Dockerfile.prod` would otherwise
  pick up from a dirty local tree).
- WIF pool/provider recreated (`scripts/setup_wif.sh`, same GitHub-repo-scoped
  `attribute-condition` as before); GitHub repo variable `GCP_WIF_PROVIDER` updated.
- `metrics-token` Secret Manager secret **rotated** — the old value lived in the now-inaccessible
  project and could not be read back (Secret Manager, like GCS, requires an active billing
  account even to read). GitHub secret `SMOKE_TEST_METRICS_TOKEN` updated to match the new value.
  `groq-api-key` did not need rotation — the value was already available locally in `.env`.
- New `.github/workflows/health-monitor.yml`: scheduled `/health` check every 30 minutes, fails
  loudly (non-zero exit, GitHub's default failure-notification email) on anything but a 200 with
  `status: "ok"`. This is the control that was missing and let the original outage run
  undetected for up to 12 days — added specifically to close that gap, not a generic nice-to-have.

## Consequences

- **Positive**: production restored; IAM blast radius is provably narrower than the original
  single-project design ever was, even before co-tenancy was a consideration.
- **Negative / residual risk**: TriageIQ's Cloud Run cost, quota, and (very indirectly) any
  outage/incident now shows up inside `expense-tracker-498014`'s billing and quota surface. Given
  TriageIQ's traffic is portfolio-scale, not expected to be material, but worth remembering if
  `expense-tracker-498014`'s own quotas ever get tight.
- **Open item, explicitly not closed by this ADR**: this is a stopgap. GG is filing a GCP
  support request to raise the `Cloud billing quota exceeded` cap on `014DAE-6B3556-077365`. If
  that clears, the plan is a follow-up migration to a dedicated project — repeat this same IAM-
  scoping discipline there, don't regress to project-level grants just because there's no
  co-tenant to protect anymore.
- Anyone reading this cold: **if you're wondering why a triage/issue-classification service's
  Cloud Run URL and Artifact Registry repo live inside a project named `expense-tracker-498014`,
  this is why** — it is not a naming mistake or a merge accident, it's a deliberate, IAM-scoped
  co-tenancy decision made under a production outage, expected to be temporary.

## Alternatives considered

- **Relink the old billing account** — rejected by GG; the account itself is closed, not just
  unlinked, so this isn't actually available regardless of preference.
- **New dedicated project** — the clean option, blocked by `Cloud billing quota exceeded` on the
  only working billing account. Would eliminate co-tenancy risk entirely; deferred pending the
  quota-increase request.
- **`review-iq-prod` / `agentic-travel-booking-system` / `iconic-reactor-496423-m4`** — surveyed
  and rejected in favor of `expense-tracker-498014` per the table above (footprint, region,
  tenancy hygiene).
