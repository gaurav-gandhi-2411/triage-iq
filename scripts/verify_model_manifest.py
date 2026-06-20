from __future__ import annotations
"""CI model-artifact drift guard.

Reads MANIFEST.sha256 committed to the repo, downloads each GCS artifact to
a temp directory, and compares SHA-256 hashes. Exits non-zero if any artifact
is missing from GCS, fails to download, or has a hash mismatch.

Works on Linux (CI) and Windows (dev). Requires gcloud CLI with ADC:
    gcloud auth application-default login

Usage:
    python scripts/verify_model_manifest.py

Exit codes:
    0  — all artifacts match
    1  — one or more mismatches, missing artifacts, or download failures
"""

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "models" / "MANIFEST.sha256"
GCS_PREFIX = "gs://triageiq-portfolio-495022-models"

# On Windows, gcloud is a .cmd file; bare "gcloud" requires shell=True or the .cmd suffix.
_GCLOUD = "gcloud.cmd" if sys.platform == "win32" else "gcloud"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gcs_path(local_rel: str) -> str:
    suffix = local_rel.removeprefix("data/")
    return f"{GCS_PREFIX}/{suffix}"


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    lines = [
        ln.strip()
        for ln in MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if not lines:
        print("ERROR: manifest is empty")
        sys.exit(1)

    print(f"Manifest: {MANIFEST_PATH} ({len(lines)} entries)")
    print(f"GCS prefix: {GCS_PREFIX}\n")

    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for line in lines:
            expected_hash, local_rel = line.split("  ", 1)
            gcs = _gcs_path(local_rel)
            artifact_name = Path(local_rel).name
            tmp = Path(tmpdir) / artifact_name

            dl = subprocess.run(
                [_GCLOUD, "storage", "cp", gcs, str(tmp)],
                capture_output=True,
                text=True,
            )
            if dl.returncode != 0:
                print(f"  DOWNLOAD FAILED  {local_rel}")
                print(f"    {dl.stderr.strip()}")
                failures.append(local_rel)
                continue

            actual = _sha256_file(tmp)
            if actual == expected_hash:
                print(f"  OK  {expected_hash[:16]}  {local_rel}")
            else:
                print(f"  MISMATCH  {local_rel}")
                print(f"    manifest : {expected_hash[:32]}")
                print(f"    gcs      : {actual[:32]}")
                failures.append(local_rel)

    print()
    if failures:
        print(f"DRIFT DETECTED: {len(failures)}/{len(lines)} artifact(s) do not match the committed manifest.")
        print("Run python scripts/publish_models.py to re-upload and update the manifest.")
        sys.exit(1)
    else:
        print(f"All {len(lines)} artifacts verified — GCS matches MANIFEST.sha256.")


if __name__ == "__main__":
    main()
