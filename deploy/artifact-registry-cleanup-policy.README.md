# Artifact Registry cleanup policy

`deploy.yml` pushes a new tagged image on every push to `main` (a `:<sha>` tag plus `:latest`)
with no retention step. Left unmanaged, this grows without bound — the `triageiq` repo in
`expense-tracker-498014` reached 32 images (~39 GB reported, ~13 GB measured directly from image
manifests once the count came down — the registry's cached `sizeBytes` lags actual deletions by
some unknown interval, so trust a fresh manifest-level check over that field if the numbers look
stale) after about a week of active development.

`artifact-registry-cleanup-policy.json` in this directory keeps the 15 most recent versions
unconditionally (comfortable headroom past any realistic Cloud Run rollback) and deletes anything
both older than 14 days *and* outside that window. It does not know about Cloud Run traffic
directly — Artifact Registry cleanup policies can't express "keep whatever a live revision
references" natively — so the count-based keep rule is the safety margin instead. Before ever
lowering `keepCount`, cross-check against `gcloud run revisions list --service=<name> --format="table(name,active,ImageDigest)"`
for every project this policy is applied to, to confirm the live/candidate image digests fall
well inside the kept window.

## Applying it

```bash
gcloud artifacts repositories set-cleanup-policies triageiq \
  --project=<PROJECT_ID> \
  --location=us-central1 \
  --policy=deploy/artifact-registry-cleanup-policy.json \
  --no-dry-run
```

Already applied to `expense-tracker-498014`'s `triageiq` repo (2026-08-12), alongside a one-time
manual cleanup of 22 stale images. Needs the same command run against `triageiq-prod-260812` once
that project has repository access set up — this file exists so that doesn't require relaying the
JSON through a chat transcript.
