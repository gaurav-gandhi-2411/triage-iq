"""Write eval/cassettes/EXPECTED_ARTIFACT_HASHES.json for a recording set (ADR-0059).

The recorder refuses to run unless the artifacts it resolves hash to this committed file. For a set
with a RECORDING_CONFIG.json the fingerprint carries the config's extra artifacts (family_v2 bundle, k8s
v2 index, resolution evidence), so the file has to be regenerated under the same config.

Every hash is cross-checked against data/models/MANIFEST.sha256 (the file CI verifies against GCS) and the
comparison is printed. Nothing is written if any artifact is missing, absent from the manifest or differs.

Usage: python scripts/write_expected_artifact_hashes.py --set-dir eval/sets/quality_push_2026-10 [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import artifact_fingerprint as af  # noqa: E402
from recording_config import load_recording_config  # noqa: E402

MANIFEST_REL = "data/models/MANIFEST.sha256"


def read_manifest(path: Path) -> dict[str, str]:
    """{relative path: sha256} from a sha256sum-format file (CRLF tolerated)."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, _, rel = line.strip().partition("  ")
            out[rel.strip()] = digest
    return out


def cross_check(hashes: dict[str, str], manifest: dict[str, str]) -> tuple[list[str], list[str]]:
    """(printable table rows, problems). A problem is MISSING, not in the manifest, or different."""
    rows, problems = [], []
    for rel, h in sorted(hashes.items()):
        m = manifest.get(rel)
        status = (
            "MISSING"
            if h == "MISSING"
            else "NOT-IN-MANIFEST"
            if m is None
            else ("OK" if m == h else "MISMATCH")
        )
        if status != "OK":
            problems.append(f"{status}: {rel}")
        rows.append(f"{status:<16} {h[:16]}  {(m or '-')[:16]}  {rel}")
    return rows, problems


def build(root: Path, set_dir: Path) -> tuple[dict[str, str], list[str], list[str], str]:
    """(hashes, table rows, problems, config as JSON) for `set_dir`'s RECORDING_CONFIG under `root`."""
    config = load_recording_config(root, set_dir / "cassettes")
    hashes = af.compute_artifact_hashes(root, config=config)
    rows, problems = cross_check(hashes, read_manifest(root / MANIFEST_REL))
    return hashes, rows, problems, json.dumps(config.__dict__, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set-dir", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT, help="checkout holding data/models")
    ap.add_argument("--note", default=None, help="appended to the file's running note")
    ap.add_argument("--check", action="store_true", help="print the comparison, write nothing")
    args = ap.parse_args(argv)
    hashes, rows, problems, config = build(args.root, args.set_dir)
    print(f"config: {config}\nstatus           expected         manifest         path")
    print("\n".join(rows))
    print(f"{len(hashes)} entries, {len(problems)} problems")
    if problems:
        print("NOT WRITTEN:", *problems, sep="\n  ")
        return 1
    if args.check:
        return 0
    path = af.expected_hashes_path(args.root)
    old = json.loads(path.read_text(encoding="utf-8")).get("note", "") if path.exists() else ""
    new = args.note or f"regenerated for {args.set_dir.name} under {config}"
    af.save_expected_hashes(args.root, hashes, note=f"{old} | {new}" if old else new)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
