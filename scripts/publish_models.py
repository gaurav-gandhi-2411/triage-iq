from __future__ import annotations
"""Publish local model artifacts to GCS and update MANIFEST.sha256.

Safe ordering (prevents half-complete state from poisoning the guard):
  1. Hash all local artifacts
  2. Write MANIFEST.sha256 (source of truth)
  3. Upload each artifact to GCS
  4. Read back each uploaded object and verify hash matches manifest
  5. Print the git commands to commit the manifest — only on full success

Usage:
    python scripts/publish_models.py [--dry-run]

Requires: gcloud CLI with Application Default Credentials
    gcloud auth application-default login
"""

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "models" / "MANIFEST.sha256"
GCS_PREFIX = "gs://triageiq-models"
EXPECTED_PROJECT = "expense-tracker-498014"

# On Windows, gcloud is a .cmd file; bare "gcloud" requires shell=True or the .cmd suffix.
_GCLOUD = "gcloud.cmd" if sys.platform == "win32" else "gcloud"


def _assert_correct_project() -> None:
    """Hard gate before any gcloud/GCS-touching command in this publish path. The active
    project has drifted silently before (2026-07-23, root-caused: an explicit
    `gcloud config set project` to a different, off-limits project, undetected for hours
    because nothing in between happened to run a project-scoped command). A read-only mistake
    was luck last time; a publish/upload against the wrong project is the failure this guards
    against."""
    result = subprocess.run([_GCLOUD, "config", "get-value", "project"], capture_output=True, text=True)
    active = result.stdout.strip()
    if active != EXPECTED_PROJECT:
        print(f"HARD STOP: active gcloud project is '{active}', expected '{EXPECTED_PROJECT}'.")
        print("Refusing to publish/upload against the wrong project.")
        print(f"Fix: gcloud config set project {EXPECTED_PROJECT}")
        sys.exit(1)

# Artifacts to publish, relative to REPO_ROOT.
# Any addition here must be mirrored in verify_model_manifest.py.
ARTIFACTS: list[str] = [
    "data/models/component_classifier_microsoft_vscode.pkl",
    "data/models/component_classifier_kubernetes_kubernetes.pkl",
    "data/models/resolution_predictor_microsoft_vscode.pkl",
    "data/models/resolution_predictor_kubernetes_kubernetes.pkl",
    "data/models/cqr_conformal_adjustments.json",
    "data/models/dup_index_microsoft_vscode_bge/index.faiss",
    "data/models/dup_index_microsoft_vscode_bge/meta.pkl",
    "data/models/dup_index_kubernetes_kubernetes_bge/index.faiss",
    "data/models/dup_index_kubernetes_kubernetes_bge/meta.pkl",
    "data/processed/microsoft_vscode_temporal_train.parquet",
    "data/processed/kubernetes_kubernetes_temporal_train.parquet",
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gcs_path(local_rel: str) -> str:
    # data/models/foo.pkl -> gs://.../models/foo.pkl
    suffix = local_rel.removeprefix("data/")
    return f"{GCS_PREFIX}/{suffix}"


def main(dry_run: bool = False) -> None:
    _assert_correct_project()
    print("=== publish_models.py ===")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"GCS prefix: {GCS_PREFIX}")
    print(f"Dry run: {dry_run}\n")

    # Step 1: Hash local artifacts
    print("Step 1: Hashing local artifacts …")
    local_hashes: dict[str, str] = {}
    missing: list[str] = []
    for rel in ARTIFACTS:
        p = REPO_ROOT / rel
        if not p.exists():
            print(f"  MISSING  {rel}")
            missing.append(rel)
        else:
            h = _sha256_file(p)
            local_hashes[rel] = h
            print(f"  {h[:16]}  {rel}")

    if missing:
        print(f"\nERROR: {len(missing)} artifact(s) missing locally. Aborting.")
        sys.exit(1)

    # Step 2: Write manifest (source of truth — written before upload)
    print(f"\nStep 2: Writing {MANIFEST_PATH} …")
    lines = [f"{local_hashes[rel]}  {rel}" for rel in ARTIFACTS]
    manifest_content = "\n".join(lines) + "\n"
    if not dry_run:
        # Explicit newline='\n' prevents CRLF on Windows; CI (Linux) reads LF
        MANIFEST_PATH.write_text(manifest_content, encoding="utf-8", newline="\n")
    else:
        print("  [dry-run] would write:")
        for line in lines:
            print(f"    {line}")

    # Step 3: Upload each artifact to GCS
    print("\nStep 3: Uploading artifacts to GCS …")
    for rel in ARTIFACTS:
        gcs = _gcs_path(rel)
        print(f"  {rel} -> {gcs}", end="", flush=True)
        if dry_run:
            print("  [dry-run skipped]")
            continue
        result = subprocess.run(
            [_GCLOUD, "storage", "cp", str(REPO_ROOT / rel), gcs],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"\nERROR uploading {rel}:\n{result.stderr}")
            sys.exit(1)
        print("  OK")

    if dry_run:
        print("\n[dry-run complete — no files written or uploaded]")
        return

    # Step 4: Read back uploaded objects and verify hashes (download to temp, hash locally)
    print("\nStep 4: Verifying GCS readback against manifest …")
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for rel in ARTIFACTS:
            gcs = _gcs_path(rel)
            tmp = Path(tmpdir) / Path(rel).name
            dl = subprocess.run(
                [_GCLOUD, "storage", "cp", gcs, str(tmp)],
                capture_output=True, text=True,
            )
            if dl.returncode != 0:
                print(f"  READBACK FAILED {rel}: {dl.stderr.strip()}")
                failures.append(rel)
                continue
            actual = _sha256_file(tmp)
            expected = local_hashes[rel]
            if actual == expected:
                print(f"  OK  {rel}")
            else:
                print(f"  MISMATCH  {rel}")
                print(f"    expected: {expected[:32]}")
                print(f"    got:      {actual[:32]}")
                failures.append(rel)

    if failures:
        print(f"\nERROR: readback verification failed for {len(failures)} artifact(s):")
        for f in failures:
            print(f"  {f}")
        print("MANIFEST.sha256 was written but GCS may be inconsistent.")
        print("Re-run publish_models.py to retry the upload+verify cycle.")
        sys.exit(1)

    # Step 5: Success — print git commands
    print("\n=== SUCCESS — all artifacts uploaded and verified ===")
    print("\nNext: commit the updated manifest and push:")
    print()
    print("    git add data/models/MANIFEST.sha256")
    print('    git commit -m "chore(models): update model manifest — <describe what changed>"')
    print("    git push")
    print()
    print("The committed MANIFEST.sha256 is the source of truth.")
    print("deploy.yml will verify GCS matches this manifest before every deploy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Hash and plan without writing or uploading")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
