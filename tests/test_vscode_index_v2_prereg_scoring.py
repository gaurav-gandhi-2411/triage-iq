"""scripts/vscode_index_v2_prereg_scoring.py helpers: the served-copy drop and the paired bootstrap.

Hermetic: tiny FAISS index and tmp_path parquet/JSON; no model, no network, nothing under data/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import vscode_index_v2_prereg_scoring as sc  # noqa: E402


def test_served_copy_drops_undated_rows_keeping_vectors_aligned(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    vecs = np.eye(3, dtype=np.float32)
    index = faiss.IndexFlatIP(3)
    index.add(vecs)
    faiss.write_index(index, str(src / "index.faiss"))
    joblib.dump({"repo": "microsoft_vscode", "model_key": "bge",
                 "issue_numbers": np.array([10, 20, 30], dtype=np.int64),
                 "texts": ["a", "b", "c"]}, str(src / "meta.pkl"))
    derived = tmp_path / "derived" / "microsoft__vscode"
    derived.mkdir(parents=True)
    pd.DataFrame({"number": [10, 30], "created_at": pd.to_datetime(["2020-01-01", "2021-01-01"], utc=True)}
                 ).to_parquet(derived / "issues.parquet")
    raw = tmp_path / "raw" / "microsoft_vscode"  # no 20.json: #20 is undated everywhere
    raw.mkdir(parents=True)
    dst = tmp_path / "dst"

    info = sc.make_served_dated_copy(src, dst, tmp_path / "derived", tmp_path / "raw")

    assert info["dropped_undated_issue_numbers"] == [20]
    assert (info["n_before"], info["n_after"]) == (3, 2)
    meta = joblib.load(str(dst / "meta.pkl"))
    assert meta["issue_numbers"].tolist() == [10, 30] and meta["texts"] == ["a", "c"]
    assert meta["created_at"][1] == np.datetime64("2021-01-01", "ns")
    kept = faiss.read_index(str(dst / "index.faiss"))
    assert kept.ntotal == 2
    assert np.allclose(kept.reconstruct_n(0, 2), vecs[[0, 2]])  # row 20's vector is gone, order kept
    assert "created_at" not in joblib.load(str(src / "meta.pkl"))  # source untouched
    json.dumps(info)


def test_paired_bootstrap_is_deterministic_and_signed() -> None:
    a = np.array([1] * 80 + [0] * 20, dtype=np.int8)
    b = np.array([1] * 20 + [0] * 80, dtype=np.int8)
    idx = np.random.default_rng(sc.SEED).integers(0, 100, size=(500, 100))
    r1, r2 = sc.paired_bootstrap(a, b, idx), sc.paired_bootstrap(a, b, idx)
    assert r1 == r2
    assert abs(r1["point"] - 0.6) < 1e-9 and r1["ci95"][0] > 0
    assert sc.paired_bootstrap(b, a, idx)["ci95"][1] < 0
