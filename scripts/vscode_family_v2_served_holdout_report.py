"""Committed provenance report for the SERVED vscode family_v2 classifier's own holdout evaluation.

The retrain dry run `run4_rule` (ADR-0072 pipeline, code ce55978) produced the bundle that
`data/models/MANIFEST.sha256` pins and production serves. Its metrics live on D: (not committed);
this script copies them into `reports/vscode_family_v2_served_holdout_2026-10.json` so the numbers
checker has a committed source, and recomputes (never trusts) the sha256 of the bundle, checking it
against the manifest entry and the run's staged copy. It fails closed: any mismatch raises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "component_classifier_microsoft_vscode_family_v2_2026-10-22.joblib"
STATEMENT = (
    "Metrics of the dry run run4_rule copied from its metrics.json / window_meta.json. The challenger "
    "bundle is byte-identical (sha256 recomputed at generation time) to the entry in "
    "data/models/MANIFEST.sha256, i.e. to the family_v2 classifier production serves. Not a "
    "re-evaluation: no model was re-scored by this script."
)


class ProvenanceError(RuntimeError):
    """A sha256 or a required input did not match; the report is not written."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_sha(manifest: Path, name: str) -> str:
    """sha256 the manifest pins for the file whose path ends with `name` (exactly one entry)."""
    hits = [ln.split()[0] for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip().endswith(name)]
    if len(hits) != 1:
        raise ProvenanceError(f"{len(hits)} manifest entries end with {name!r} (need exactly 1)")
    return hits[0]


def build_report(run_dir: Path, manifest: Path, repo_bundle: Path | None = None) -> dict[str, Any]:
    """Assemble the report; raises ProvenanceError unless every recomputed sha256 agrees."""
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    window = json.loads((run_dir / "window_meta.json").read_text(encoding="utf-8"))
    code_sha = (run_dir / "code_sha.txt").read_text(encoding="utf-8").split()[0]
    pinned = manifest_sha(manifest, BUNDLE)
    shas = {
        "challenger.joblib": sha256_file(run_dir / "challenger.joblib"),
        f"staging/{BUNDLE}": sha256_file(run_dir / "staging" / BUNDLE),
    }
    if repo_bundle is not None and repo_bundle.exists():
        shas[f"repo data/models/{BUNDLE}"] = sha256_file(repo_bundle)
    bad = {k: v for k, v in shas.items() if v != pinned}
    if bad:
        raise ProvenanceError(f"sha256 differs from the manifest pin {pinned}: {bad}")
    keep = ("strict_top3", "strict_top1", "in_family_top3")
    return {
        "statement": STATEMENT,
        "served_bundle": {"file": BUNDLE, "sha256_manifest": pinned, "sha256_recomputed": shas,
                          "all_match_manifest": True},
        "run": "retrain_dryrun/run4_rule (D:\\triageiq-data, not committed)",
        "code_sha": code_sha,
        "window": {"as_of": window["as_of"], "holdout_start": window["windows"]["holdout_start"],
                   "holdout_end_exclusive": window["windows"]["cutoff"], "counts": window["counts"]},
        "n_holdout": metrics["n_holdout"],
        "n_in_family": metrics["n_in_family"],
        "challenger_served": {k: metrics["challenger"][k] for k in keep},
        "champion_previous_served": {k: metrics["champion"][k] for k in keep},
        "naive_most_common_labels": {k: metrics["naive"][k] for k in keep},
        "paired_vs_champion": metrics["paired_vs_champion"],
        "paired_vs_naive": metrics["paired_vs_naive"],
        "champion_sha256": metrics["champion"]["sha256"],
        "serving_parity_source": "docs/RUN_STATUS.md: 'Serving parity on it: 300/300 identical top-3 incl. order, max abs diff 0.0'",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=Path(r"D:\triageiq-data\retrain_dryrun\run4_rule"))
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "models" / "MANIFEST.sha256")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "vscode_family_v2_served_holdout_2026-10.json")
    a = ap.parse_args(argv)
    rep = build_report(a.run_dir, a.manifest, ROOT / "data" / "models" / BUNDLE)
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {a.out} (bundle sha256 {rep['served_bundle']['sha256_manifest'][:12]}, all match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
