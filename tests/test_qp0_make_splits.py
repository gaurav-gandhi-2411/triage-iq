from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SPEC = importlib.util.spec_from_file_location("qp0s", _SCRIPTS / "qp0_make_splits.py")
qp0s = importlib.util.module_from_spec(_SPEC)
sys.modules["qp0s"] = qp0s
_SPEC.loader.exec_module(qp0s)

TRAIN_T, VAL_T, TEST_T = "2024-05-01T00:00:00Z", "2025-09-01T00:00:00Z", "2026-03-01T00:00:00Z"


def issues(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    """(number, created_at, text_hash)."""
    return pd.DataFrame(
        {
            "number": [r[0] for r in rows],
            "created_at": pd.to_datetime([r[1] for r in rows], utc=True),
            "closed_at": pd.NaT, "state": "CLOSED", "text_hash": [r[2] for r in rows],
        }
    )  # fmt: skip


def pairs(rows: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {"source": [r[0] for r in rows], "canonical": [r[1] for r in rows]}, dtype="int64"
    )


def run(rows: list[tuple[int, str, str]], pr: list[tuple[int, int]] | None = None):
    return qp0s.make_splits(issues(rows), pairs(pr or []))


def split_of(df: pd.DataFrame, n: int) -> str:
    return df.set_index("number").at[n, "split"]


def test_boundary_timestamps_and_tz() -> None:
    df, _ = run(
        [
            (1, "2025-06-30T23:59:59Z", "a"), (2, "2025-07-01T00:00:00Z", "b"),
            (3, "2025-12-31T23:59:59Z", "c"), (4, "2026-01-01T00:00:00Z", "d"),
            (5, "2025-07-01T01:00:00+02:00", "e"),  # = 2025-06-30T23:00Z: train
        ]
    )  # fmt: skip
    assert [split_of(df, n) for n in (1, 2, 3, 4, 5)] == [
        "train", "validation", "validation", "test", "train",
    ]  # fmt: skip
    naive = issues([(1, TRAIN_T, "a")]).assign(created_at=pd.Timestamp("2024-01-01"))
    with pytest.raises(ValueError, match="timezone"):
        qp0s.make_splits(naive, pairs([]))


def test_texthash_leak_both_directions_and_within_split() -> None:
    df, _ = run(
        [
            (1, TRAIN_T, "dup"), (2, TEST_T, "dup"),  # train text repeated in test: leak
            (3, TEST_T, "unique"),  # not a leak
            (4, TEST_T, "twice"), (5, TEST_T, "twice"),  # only within test: not a leak
            (6, VAL_T, "vt"), (7, TEST_T, "vt"),  # val text repeated in test: test leaks
            (8, TRAIN_T, "tr2"), (9, VAL_T, "tr2"),  # val leaks from train, train does not
        ]
    )  # fmt: skip
    leak = df.set_index("number")["leak_texthash"]
    assert [bool(leak[n]) for n in range(1, 10)] == [
        False, True, False, False, False, False, True, False, True,
    ]  # fmt: skip
    assert df.set_index("number").at[1, "classifier_eligible"]
    assert not df.set_index("number").at[2, "classifier_eligible"]


def test_dup_pair_leak_flags_both_ends() -> None:
    rows = [
        (1, TRAIN_T, "a"),
        (2, TEST_T, "b"),
        (3, TEST_T, "c"),
        (4, TRAIN_T, "d"),
        (5, TEST_T, "e"),
    ]
    df, _ = run(rows, [(2, 1), (4, 3), (5, 3)])  # 2->1 source later; 4->3 canonical later
    f = df.set_index("number")["leak_dup_pair"]
    assert [bool(f[n]) for n in (1, 2, 3, 4, 5)] == [False, True, True, False, False]
    # 5 and 3 are both test: within-split pair is not leakage


def test_dup_task_pairs_selection() -> None:
    rows = [
        (1, TRAIN_T, "x"), (2, VAL_T, "x"), (3, TEST_T, "y"), (4, TEST_T, "z"),
        (5, TRAIN_T, "q"),
    ]  # fmt: skip
    _, dup = run(rows, [(2, 1), (3, 2), (2, 3), (4, 99), (5, 1)])
    got = {(r.source, r.canonical): r for r in dup.itertuples()}
    assert set(got) == {(2, 1), (3, 2), (4, 99)}  # (2,3): canonical newer; (5,1): train source
    assert got[(2, 1)].exact_text_dup is True and got[(3, 2)].exact_text_dup is False
    assert got[(4, 99)].canonical_in_table is False and got[(4, 99)].split_of_canonical is None
    assert got[(3, 2)].split_of_canonical == "validation"


def test_hash_stable_under_row_order_and_deterministic(tmp_path: Path) -> None:
    rows = [(i, [TRAIN_T, VAL_T, TEST_T][i % 3], f"h{i % 4}") for i in range(1, 13)]
    a, da = run(rows, [(3, 1)])
    b, db = run(rows[::-1], [(3, 1)])
    s1, s2 = qp0s.summarise(a, da), qp0s.summarise(b, db)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)
    assert qp0s.digest_numbers(pd.Series([3, 1, 2])) == qp0s.digest_numbers(pd.Series([1, 2, 3]))
    assert s1["train"]["sha256_numbers"] != s1["test"]["sha256_numbers"]


def test_main_writes_outputs_and_report_is_byte_identical(tmp_path: Path) -> None:
    derived = tmp_path / "d"
    derived.mkdir()
    rows = [(1, TRAIN_T, "a"), (2, VAL_T, "a"), (3, TEST_T, "c")]
    issues(rows).to_parquet(derived / "issues.parquet")
    pairs([(3, 1)]).to_parquet(derived / "duplicate_pairs.parquet")
    out, rep = tmp_path / "o", tmp_path / "rep.json"
    args = [
        "--repo",
        "o/r",
        "--derived-dir",
        str(derived),
        "--out-dir",
        str(out),
        "--report",
        str(rep),
    ]
    qp0s.main(args)
    first = rep.read_bytes()
    qp0s.main(args)
    assert rep.read_bytes() == first
    man = json.loads(first)
    assert man["train"]["n"] == man["validation"]["n"] == man["test"]["n"] == 1
    assert man["validation"]["leak_texthash"] == 1 and man["test"]["leak_dup_pair"] == 1
    assert (out / "test.parquet").exists() and (out / "dup_task_pairs.parquet").exists()
    assert set(pd.read_parquet(out / "test.parquet").columns) == set(qp0s.COLS)
    assert '"number"' not in first.decode()  # the report carries no issue ids
