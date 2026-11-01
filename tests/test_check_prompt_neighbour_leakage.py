"""scripts/check_prompt_neighbour_leakage.py: strict created-before rule on synthetic data.

Hermetic: tmp_path parquet/JSON and hand-written prompt strings; no model, retriever or network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "eval", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import check_prompt_neighbour_leakage as leak  # noqa: E402

Q = "2026-03-01T12:00:00+00:00"
ISSUE = {"id": "k8s-500", "repo": "kubernetes/kubernetes", "number": 500, "created_at": Q}


def test_compare_is_strict_and_fails_closed() -> None:
    assert leak.compare(Q, "2026-03-01T11:59:59Z") == leak.BEFORE
    assert leak.compare(Q, "2026-03-01T12:00:00Z") == leak.EQUAL  # same instant is not "before"
    assert leak.compare(Q, "2026-03-01T12:00:01Z") == leak.AFTER
    assert leak.compare(Q, pd.Timestamp("2026-03-01 11:00:00")) == leak.BEFORE  # naive = UTC
    for missing in (None, "", float("nan"), pd.NaT, "not a date"):
        assert leak.compare(Q, missing) == leak.UNKNOWN


def test_prompt_neighbours_reads_similar_and_evidence_numbers() -> None:
    user = (
        "--- ISSUE ---\nBody mentions #999 in passing\n"
        "--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---\n"
        "  #11 (similarity: 0.900): a...\n  #12 (similarity: 0.800): b...\n"
        "\n--- RESOLUTION EVIDENCE (how the top similar issues were resolved) ---\n"
        '#11: fix PR 7 "x"\n'
        "\n--- SYSTEM 3: RESOLUTION TIME ESTIMATE ---\n#13: not a neighbour\n"
    )
    assert leak.prompt_neighbours(user) == ([11, 12], [11])
    no_evidence = user.split("\n--- RESOLUTION EVIDENCE")[0] + "\n--- SYSTEM 3: X ---\n"
    assert leak.prompt_neighbours(no_evidence) == ([11, 12], [])


def test_check_issue_and_summarize_flag_every_failure_shape() -> None:
    dates = {
        1: "2026-01-01T00:00:00Z",  # before
        2: "2026-03-01T12:00:00Z",  # equal
        3: "2026-04-01T00:00:00Z",  # after
        4: None,  # unknown
        500: "2025-01-01T00:00:00Z",  # self-match even though its date is earlier
    }
    items = [
        {"neighbour": 1, "kind": "fix PR", "number": 9, "at": "2026-02-01T00:00:00Z"},
        {"neighbour": 1, "kind": "closing comment", "number": 0, "at": "2026-05-01T00:00:00Z"},
    ]
    rows = leak.check_issue(ISSUE, [1, 2, 3, 4, 500], [1, 2], items, dates)
    assert len(rows) == 9
    s = leak.summarize(rows)
    sim = s["per_repo"]["kubernetes/kubernetes"]["similar"]
    assert sim == {"total": 5, "before": 2, "equal": 1, "after": 1, "unknown": 1, "self_match": 1}
    ev = s["per_repo"]["kubernetes/kubernetes"]["evidence_item"]
    assert ev["total"] == 2 and ev["before"] == 1 and ev["after"] == 1
    # violations = equal/after rows + self-match (similar 2, 3, 500; evidence_nbr 2; the late
    # closing comment); the one undated neighbour is counted apart as unknown
    assert s["n_violations"] == 5 and s["n_unknown"] == 1
    assert {(v["type"], v["neighbour"]) for v in s["violations"]} >= {
        ("similar", 2),
        ("similar", 3),
        ("similar", 500),
        ("similar", 4),
        ("evidence_item", 1),
    }


def test_clean_issue_has_no_violations() -> None:
    dates = {1: "2026-01-01T00:00:00Z", 2: "2026-02-01T00:00:00Z"}
    s = leak.summarize(leak.check_issue(ISSUE, [1, 2], [1], [], dates))
    assert s["n_violations"] == 0 and s["n_unknown"] == 0 and s["violations"] == []


def test_load_issue_dates_drops_conflicting_duplicates(tmp_path: Path) -> None:
    d = tmp_path / "kubernetes__kubernetes"
    d.mkdir()
    pd.DataFrame(
        {
            "number": [1, 2, 2, 3, 3],
            "created_at": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-04"], utc=True
            ),
        }
    ).to_parquet(d / "issues.parquet")
    got = leak.load_issue_dates(tmp_path, "kubernetes/kubernetes")
    assert set(got) == {1, 3}  # 2 is ambiguous -> unknown downstream; 3 is a harmless exact repeat


def test_repo_dates_falls_back_to_raw_json_and_reports_disagreement(tmp_path: Path) -> None:
    raw = tmp_path / "microsoft_vscode"
    raw.mkdir()
    for n, ts in ((10, "2020-01-01T00:00:00Z"), (20, "2026-02-01T00:00:00Z")):
        (raw / f"{n}.json").write_text(json.dumps({"created_at": ts}), encoding="utf-8")
    parquet = {20: pd.Timestamp("2026-02-05T00:00:00Z"), 30: pd.Timestamp("2026-02-06T00:00:00Z")}
    rd = leak.RepoDates("microsoft/vscode", parquet, tmp_path)
    assert rd.get(10) == "2020-01-01T00:00:00Z"  # only in raw
    assert rd.get(20) == parquet[20]  # parquet wins ...
    assert rd.disagreements == [
        {"number": 20, "parquet": str(parquet[20]), "raw": "2026-02-01T00:00:00Z"}
    ]  # ... and the conflict is reported
    assert rd.get(30) == parquet[30]  # parquet only: raw file absent is not a disagreement
    assert rd.get(99) is None  # nowhere -> unknown
    assert rd.source == {"raw": 1, "parquet": 2, "none": 1}
    assert rd.get(10) == "2020-01-01T00:00:00Z" and rd.source["raw"] == 1  # memoised
