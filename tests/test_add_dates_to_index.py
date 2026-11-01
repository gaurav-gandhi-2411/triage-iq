"""scripts/add_dates_to_index.py: dating a COPY of an index (parquet first, raw JSON fallback).

Hermetic: tiny FAISS index, tmp_path parquet/JSON; no model or network.
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

import add_dates_to_index as dating  # noqa: E402


def _src_index(tmp_path: Path, numbers: list[int]) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    index = faiss.IndexFlatIP(2)
    index.add(np.eye(2, dtype=np.float32)[[i % 2 for i in range(len(numbers))]])
    faiss.write_index(index, str(src / "index.faiss"))
    joblib.dump(
        {
            "repo": "microsoft_vscode",
            "model_key": "bge",
            "issue_numbers": np.array(numbers, dtype=np.int64),
            "texts": [f"t{n}" for n in numbers],
        },
        str(src / "meta.pkl"),
    )
    return src


def _derived(tmp_path: Path, rows: dict[int, str]) -> Path:
    d = tmp_path / "derived" / "microsoft__vscode"
    d.mkdir(parents=True)
    pd.DataFrame({"number": list(rows), "created_at": pd.to_datetime(list(rows.values()), utc=True)}
                 ).to_parquet(d / "issues.parquet")
    return tmp_path / "derived"


def test_dating_script_parquet_plus_raw_fallback_byte_identical_faiss(tmp_path: Path) -> None:
    src = _src_index(tmp_path, [1, 2, 3])
    derived = _derived(tmp_path, {1: "2020-01-01T00:00:00Z", 2: "2021-02-02T00:00:00Z"})
    raw = tmp_path / "raw" / "microsoft_vscode"
    raw.mkdir(parents=True)
    (raw / "3.json").write_text(json.dumps({"created_at": "2022-03-03T03:03:03Z"}), "utf-8")
    dst = tmp_path / "dst"
    rc = dating.main(
        ["--src", str(src), "--dst", str(dst), "--repo", "microsoft/vscode",
         "--derived-dir", str(derived), "--raw-dir", str(tmp_path / "raw")]
    )
    assert rc == 0
    assert (dst / "index.faiss").read_bytes() == (src / "index.faiss").read_bytes()
    meta = joblib.load(str(dst / "meta.pkl"))
    assert meta["created_at"][2] == np.datetime64("2022-03-03T03:03:03", "ns")
    assert meta["created_at"].dtype == np.dtype("datetime64[ns]")
    man = json.loads((dst / "manifest.json").read_text("utf-8"))
    assert man["date_sources"] == {"parquet": 2, "raw": 1}
    assert man["files"]["meta.pkl"] == dating.sha256_file(dst / "meta.pkl")
    assert man["source_files"]["index.faiss"] == man["files"]["index.faiss"]
    assert "created_at" not in joblib.load(str(src / "meta.pkl"))  # source untouched


def test_dating_script_fails_closed_on_missing_date_and_writes_nothing(tmp_path: Path) -> None:
    src = _src_index(tmp_path, [1, 2])
    derived = _derived(tmp_path, {1: "2020-01-01T00:00:00Z"})
    dst = tmp_path / "dst"
    rc = dating.main(["--src", str(src), "--dst", str(dst), "--repo", "microsoft/vscode",
                      "--derived-dir", str(derived)])
    assert rc == 1
    assert not dst.exists()


def test_dating_script_refuses_existing_dst(tmp_path: Path) -> None:
    src = _src_index(tmp_path, [1])
    derived = _derived(tmp_path, {1: "2020-01-01T00:00:00Z"})
    dst = tmp_path / "dst"
    dst.mkdir()
    rc = dating.main(["--src", str(src), "--dst", str(dst), "--repo", "microsoft/vscode",
                      "--derived-dir", str(derived)])
    assert rc == 1
