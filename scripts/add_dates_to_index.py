"""Add per-issue created_at to a COPY of a similar-issue index directory.

Reads <src>/{index.faiss,meta.pkl}, looks every indexed issue number up in the derived_v2
issues.parquet (raw GitHub JSON as fallback), and writes <dst>/{index.faiss,meta.pkl,manifest.json}.
index.faiss is copied byte-identical (sha-checked); meta.pkl gains a `created_at` array (naive-UTC
datetime64[ns], aligned with issue_numbers). The source dir is never modified. Fails (exit 1,
nothing written) if any issue has no date, or if <dst> already exists.

Usage:
    python scripts/add_dates_to_index.py --src DIR --dst DIR --repo microsoft/vscode \
        --derived-dir D:/triageiq-data/derived_v2 --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_prompt_neighbour_leakage import load_issue_dates, raw_created_at  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_dates(
    numbers: np.ndarray, parquet: dict[int, object], raw_dir: Path | None, repo: str
) -> tuple[np.ndarray, dict[str, int]]:
    """Per-number naive-UTC datetime64[ns] (parquet first, raw JSON fallback) + source counts.

    Raises ValueError listing the numbers that no source dates.
    """
    out = np.full(len(numbers), np.datetime64("NaT"), dtype="datetime64[ns]")
    source = {"parquet": 0, "raw": 0}
    missing: list[int] = []
    for i, n in enumerate(numbers.tolist()):
        ts = parquet.get(int(n))
        src = "parquet"
        if ts is None or pd.isna(ts):
            ts = raw_created_at(raw_dir, repo, int(n)) if raw_dir else None
            src = "raw"
        parsed = pd.NaT if ts is None else pd.to_datetime(ts, utc=True, errors="coerce")
        if parsed is pd.NaT:
            missing.append(int(n))
            continue
        out[i] = parsed.tz_localize(None).to_datetime64()
        source[src] += 1
    if missing:
        raise ValueError(f"{len(missing)} issues have no created_at, first 10: {missing[:10]}")
    return out, source


def add_dates(src: Path, dst: Path, repo: str, derived_dir: Path, raw_dir: Path | None) -> dict:
    """Write the dated copy of `src` to `dst` and return the manifest dict."""
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; refusing to overwrite")
    meta = joblib.load(str(src / "meta.pkl"))
    if "created_at" in meta:
        raise ValueError("source meta already has created_at")
    parquet = load_issue_dates(derived_dir, repo)
    dates, source = resolve_dates(meta["issue_numbers"], parquet, raw_dir, repo)
    dst.mkdir(parents=True)
    shutil.copyfile(src / "index.faiss", dst / "index.faiss")
    if sha256_file(dst / "index.faiss") != sha256_file(src / "index.faiss"):
        raise RuntimeError("index.faiss copy is not byte-identical")
    joblib.dump({**meta, "created_at": dates}, str(dst / "meta.pkl"))
    manifest = {
        "kind": "dated copy of " + str(src),
        "repo": repo,
        "n_rows": int(len(dates)),
        "created_at_min": str(dates.min()),
        "created_at_max": str(dates.max()),
        "date_sources": source,
        "meta_keys": sorted([*meta.keys(), "created_at"]),
        "source_files": {p: sha256_file(src / p) for p in ("index.faiss", "meta.pkl")},
        "files": {p: sha256_file(dst / p) for p in ("index.faiss", "meta.pkl")},
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--repo", required=True, help="owner/name, e.g. microsoft/vscode")
    ap.add_argument("--derived-dir", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    try:
        manifest = add_dates(a.src, a.dst, a.repo, a.derived_dir, a.raw_dir)
    except (ValueError, FileExistsError, RuntimeError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
