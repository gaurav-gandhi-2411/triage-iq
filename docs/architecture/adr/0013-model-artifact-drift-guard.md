# ADR-0013: Model artifact drift guard

Status: Accepted
Date: 2026-06-21

## Context

The calibration deployment gap (ADR-0004 §Deployment Correction) revealed a structural
failure mode: a model improvement committed locally was never uploaded to GCS, so
production kept serving the old artifact for six weeks. The gap was only discovered by
the eval-regression CI gate, which loaded from GCS and found an ECE of 0.31/0.61 (vs
the claimed 0.14/0.16 for the calibrated model).

At the time the calibrated pkl was committed (2026-05-19), no mechanism existed to
enforce that "committed = uploaded". The deploy workflow pulled from GCS blindly.

The failure mode has two distinct layers:
1. **GCS drift** — a local artifact was changed/committed but not uploaded to GCS.
   The next deploy pulls the old file from GCS and silently serves it.
2. **Image drift** — a GCS artifact was uploaded but the Docker image layer cached
   the old file. Production serves stale bytes even though GCS is correct.

## Decision

Ship a four-piece guard that catches both failure modes at different checkpoints:

### Piece 1 — MANIFEST.sha256 (source of truth)

`data/models/MANIFEST.sha256` — a sha256sum-format manifest listing every production
model artifact and its SHA-256 hash. Format (two spaces between hash and path):

```
{sha256hex}  data/models/component_classifier_microsoft_vscode.pkl
{sha256hex}  data/models/component_classifier_kubernetes_kubernetes.pkl
...
```

The committed manifest IS the source of truth. Any workflow that reads artifacts must
agree with it. `scripts/publish_models.py` is the only write path.

### Piece 2 — publish_models.py (atomic write path)

`scripts/publish_models.py` — run by the engineer who changes a model artifact. Steps
in this strict order:

1. Hash all local artifacts.
2. Write `MANIFEST.sha256` (committed first, before any upload — makes the manifest
   the record of intent, not an after-the-fact summary).
3. Upload each artifact to GCS via `gcloud storage cp`.
4. **Read back each uploaded object** (download to temp, hash locally) and verify it
   matches the just-written manifest. Step 4 catches silent upload corruption and GCS
   encoding issues. Only prints `git add` instructions on full success.

If step 4 fails, the manifest is already committed and GCS is inconsistent — the script
tells the operator to re-run. CI will block deploys until GCS and manifest agree.

### Piece 3 — deploy.yml pre-deploy gate (catches GCS drift)

A `Verify model artifacts against manifest (drift guard)` step runs in `deploy.yml`
**after** GCP authentication and **before** `Download production models from GCS`.
It runs `scripts/verify_model_manifest.py`, which downloads each GCS artifact to a
temp directory and compares hashes to the committed manifest. If any artifact
mismatches, the deploy job fails immediately — before any Docker build or Cloud Run
push. This catches the exact failure mode from the calibration gap.

### Piece 4 — startup assertion in loader.py (catches image drift)

`_check_manifest_drift(models_dir)` is called from `ModelStore.load_all()` during
API startup. It hashes each image-baked artifact and compares to `MANIFEST.sha256`.

- **Does NOT crash** — logs `ARTIFACT_DRIFT:` warnings and continues. The API stays
  up; a stale image is surfaced to operators via logs/alerts rather than taking down
  the service.
- **Scope:** catches image-layer corruption or a Docker cache that captured a stale
  artifact after GCS was correctly updated. The deploy gate (Piece 3) catches GCS
  drift before the image is built; the startup assertion is the last-resort check on
  what actually got baked in.

### Piece 5 — eval structural invariant (catches manifest freshness)

`test_model_manifest_clean` in `eval/test_invariants.py` — asserts that all local
artifacts match `MANIFEST.sha256`. Runs in the eval-gate CI job. Fails if:
- The manifest file is missing (engineer forgot to generate it).
- Any local artifact was modified without re-running `publish_models.py`.

This catches developer workflow errors before a branch is merged to main.

## Consequences

**What changes:**
- `data/models/MANIFEST.sha256` — new committed artifact.
- `scripts/publish_models.py` — new script, replaces ad-hoc `gsutil cp` one-liners.
- `scripts/verify_model_manifest.py` — new CI check script.
- `.github/workflows/deploy.yml` — new pre-deploy gate step (non-blocking on auth
  failure in forks; WIF is always available on main-branch deploys).
- `eval/test_invariants.py` — 6th invariant added.
- `src/triage_iq/api/loader.py` — `_check_manifest_drift()` helper + call in `load_all()`.

**What is not changed:**
- GCS bucket structure — artifacts stay at the same paths.
- Deploy workflow ordering for Docker build/push/deploy — the gate only adds a step
  before the existing download step.
- `DVC` — rejected (ADR-0012). The manifest approach is simpler: a flat text file
  tracked in git, no DVC remote, no `.dvc` metadata files, no install requirement.

**What this does NOT catch:**
- A manifest that is committed but never pushed (push is the engineer's responsibility
  after `publish_models.py` prints the `git commit && git push` instructions).
- Training data drift (`data/processed/`) — out of scope; training data is larger and
  less frequently changed. Add to manifest when training cycle management is formalized.

## Alternatives considered

- **DVC:** Rejected. Adds install complexity (dvc + remote config), `.dvc` pointer
  files alongside the actual artifacts, and CI login for the DVC remote. The manifest
  approach achieves the same integrity guarantee with a single text file.
- **Pre-commit hook to verify manifest:** Would catch local drift on commit. Rejected
  as too aggressive — hashing all model pkls on every commit is slow (~1s per pkl).
  The eval-gate invariant is the right CI checkpoint.
- **Crash on drift in startup assertion:** Rejected. A stale image should surface via
  alerts, not take down the service. The deploy gate is the correct hard stop; the
  startup assertion is defense-in-depth.
- **GCS object versioning + `if-match` headers:** Would prevent silent overwrites.
  Not adopted — the risk we're guarding against is a missing upload, not a race
  condition on GCS writes. The readback-and-verify step in `publish_models.py` is
  sufficient.
