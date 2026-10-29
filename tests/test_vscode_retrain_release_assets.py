"""scripts/vscode_retrain_release_assets.py: person fields dropped, exact columns kept, missing column fails.

Hermetic: synthetic frames under tmp_path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import vscode_retrain_release_assets as ra  # noqa: E402

PERSON = ["author", "user_login", "assignee", "actor", "email"]


def _issues() -> pd.DataFrame:
    n = 2
    cols = {c: [f"{c}-{i}" for i in range(n)] for c in ra.SPEC["issues.parquet"]}
    cols["number"] = [1, 2]
    cols.update({p: [f"{p}-secret-{i}" for i in range(n)] for p in PERSON})
    return pd.DataFrame(cols)


def _events() -> pd.DataFrame:
    cols = {c: ["x", "y"] for c in ra.SPEC["label_events.parquet"]}
    cols.update({"actor": ["bob", "eve"], "actor_id": [7, 8]})
    return pd.DataFrame(cols)


def _write_src(src: Path, issues: pd.DataFrame, events: pd.DataFrame) -> None:
    src.mkdir()
    # preserve_index keeps pandas metadata in the schema, as the real scrape files have it
    pq.write_table(pa.Table.from_pandas(issues), src / "issues.parquet")
    pq.write_table(pa.Table.from_pandas(events), src / "label_events.parquet")


def test_person_fields_dropped_and_exact_columns_kept(tmp_path: Path) -> None:
    _write_src(tmp_path / "src", _issues(), _events())
    rows = ra.project_dir(tmp_path / "src", tmp_path / "out")
    assert rows == {"issues.parquet": 2, "label_events.parquet": 2}
    for name, columns in ra.SPEC.items():
        table = pq.read_table(tmp_path / "out" / name)
        assert table.column_names == columns
        assert not table.schema.metadata  # pandas metadata would still name the dropped columns
        raw = (tmp_path / "out" / name).read_bytes()
        for secret in (*PERSON, "secret", "actor_id"):
            assert secret.encode() not in raw


def test_missing_required_column_fails(tmp_path: Path) -> None:
    _write_src(tmp_path / "src", _issues().drop(columns=["labels_t72h"]), _events())
    with pytest.raises(ValueError, match="labels_t72h"):
        ra.project_dir(tmp_path / "src", tmp_path / "out")


def test_spec_matches_the_published_column_lists() -> None:
    assert ra.SPEC["issues.parquet"] == [
        "number",
        "created_at",
        "title",
        "body",
        "labels_final",
        "labels_t72h",
        "text_hash",
    ]
    assert ra.SPEC["label_events.parquet"] == ["number", "ts", "action", "label"]
