"""Human step after a cloud recording run: fetch the latest cassette bundle into your checkout.

    python scripts/cloud_recording/pull_bundle.py --bucket triageiq-prod-260812-models

Restores eval/cassettes/eval_cassette.json and recording_checkpoint.json ONLY if the bundle extends the
cassette lineage your checkout has (same committed eval_cassette.json the job's image was built from).
Then review the diff and commit it through a normal PR; the job never pushes to GitHub. Needs
`pip install google-cloud-storage` and `gcloud auth application-default login` (as an account with read on
gs://<bucket>/recorder/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "groq_budget"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcs_state import SnapshotStore, open_bucket  # noqa: E402
from job_entrypoint import _sha, unpack_bundle  # noqa: E402


def pull(bucket: object, prefix: str, cassette_dir: Path) -> str:
    snap = SnapshotStore(bucket, f"{prefix}/cassette").latest()
    if snap is None:
        return "no cassette bundle has been published yet"
    seq, data = snap
    result = unpack_bundle(data, cassette_dir, _sha(cassette_dir / "eval_cassette.json"))
    return f"bundle {seq:08d}: {result}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="recorder")
    ap.add_argument("--cassette-dir", default=str(Path(__file__).resolve().parents[2] / "eval" / "cassettes"))
    a = ap.parse_args()
    print(pull(open_bucket(a.bucket), a.prefix, Path(a.cassette_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
