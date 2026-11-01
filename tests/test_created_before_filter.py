"""Created-before retrieval filter (GG 3a): semantics, meta round-trip, dating script.

Hermetic: a stub encoder, a tiny in-memory FAISS index and tmp_path files; no model or network.
Query vector is [1, 0]; doc i sits at angle i*step, so rank == doc order (doc 0 most similar).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


from triage_iq.models import similar_issues as si  # noqa: E402
from triage_iq.models.similar_issues import (  # noqa: E402
    MissingNeighbourDateError,
    SimilarIssueRetriever,
)

Q = "2026-03-01T12:00:00+00:00"
NAT = np.datetime64("NaT", "ns")


class _Stub:
    def encode(self, texts, **_kw):  # noqa: ANN001, ANN201
        return np.tile(np.array([1.0, 0.0]), (len(texts), 1))


def _dt(s: str) -> np.datetime64:
    return np.datetime64(pd.Timestamp(s).tz_localize(None).to_datetime64(), "ns")


def _make(dates: list | None, n: int | None = None) -> SimilarIssueRetriever:
    n = n or len(dates or [])
    r = SimilarIssueRetriever.__new__(SimilarIssueRetriever)  # skip the embedding-model load
    r.repo, r.model_key, r.model = "microsoft_vscode", "bge", _Stub()
    ang = np.arange(n) * 0.01
    r.index = faiss.IndexFlatIP(2)
    r.index.add(np.stack([np.cos(ang), np.sin(ang)], 1).astype(np.float32))
    r.issue_numbers = np.arange(1000, 1000 + n, dtype=np.int64)
    r.texts = [f"t{i}" for i in range(n)]
    r.created_at = None if dates is None else np.array(dates, dtype="datetime64[ns]")
    return r


def _nums(hits: list[dict]) -> list[int]:
    return [h["number"] for h in hits]


def test_strictly_before_and_equal_timestamp_excluded() -> None:
    old, same, new = _dt("2026-03-01T11:59:59Z"), _dt(Q), _dt("2026-03-01T12:00:01Z")
    r = _make([new, same, old, old, new, old])
    assert _nums(r.retrieve("q", k=5, created_before=Q)) == [1002, 1003, 1005]


def test_naive_created_before_is_utc_and_timestamp_objects_work() -> None:
    r = _make([_dt("2026-03-01T11:00:00Z"), _dt("2026-03-01T13:00:00Z")])
    assert _nums(r.retrieve("q", k=5, created_before="2026-03-01T12:00:00")) == [1000]
    assert _nums(r.retrieve("q", k=5, created_before=pd.Timestamp(Q))) == [1000]


def test_overfetch_returns_k_when_top_ranks_are_all_too_new() -> None:
    # 200 docs; ranks 0..119 are newer than the query, 120..199 older: needs several doublings
    # past the initial max(4*(k+1), 32) pool.
    dates = [_dt("2027-01-01")] * 120 + [_dt("2020-01-01")] * 80
    r = _make(dates)
    hits = r.retrieve("q", k=5, created_before=Q)
    assert _nums(hits) == [1120, 1121, 1122, 1123, 1124]


def test_fewer_than_k_qualify_returns_what_exists() -> None:
    r = _make([_dt("2027-01-01")] * 50 + [_dt("2020-01-01")] * 2)
    assert _nums(r.retrieve("q", k=5, created_before=Q)) == [1050, 1051]


def test_exclude_number_still_applies_with_filter() -> None:
    r = _make([_dt("2020-01-01")] * 8)
    assert 1000 not in _nums(r.retrieve("q", k=5, exclude_number=1000, created_before=Q))


def test_no_filter_path_is_the_old_behaviour() -> None:
    r = _make([_dt("2027-01-01")] * 10)
    assert _nums(r.retrieve("q", k=3)) == [1000, 1001, 1002]  # future-dated, not filtered


def test_missing_neighbour_date_serving_keeps_strict_raises() -> None:
    r = _make([NAT, _dt("2020-01-01"), _dt("2027-01-01")])
    assert _nums(r.retrieve("q", k=5, created_before=Q)) == [1000, 1001]
    with pytest.raises(MissingNeighbourDateError, match="#1000"):
        r.retrieve("q", k=5, created_before=Q, strict_dates=True)


def test_dateless_index_serving_unfiltered_logs_once_strict_raises(caplog) -> None:
    r = _make(None, n=5)
    with caplog.at_level(logging.WARNING, logger=si.logger.name):
        a = r.retrieve("q", k=3, created_before=Q)
        b = r.retrieve("q", k=3, created_before=Q)
    assert _nums(a) == _nums(b) == [1000, 1001, 1002]
    assert sum("no created_at" in m for m in caplog.messages) == 1
    with pytest.raises(MissingNeighbourDateError):
        r.retrieve("q", k=3, created_before=Q, strict_dates=True)
    # no created_before => strict has nothing to enforce
    assert _nums(r.retrieve("q", k=3, strict_dates=True)) == [1000, 1001, 1002]


def test_serving_created_at_now_is_a_noop() -> None:
    r = _make([_dt("2020-01-01")] * 4 + [_dt("2024-06-01")] * 4)
    now = pd.Timestamp.now(tz="UTC")
    assert r.retrieve("q", k=5, created_before=now) == r.retrieve("q", k=5)


def test_meta_round_trip_with_and_without_dates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(si, "SentenceTransformer", lambda *_a, **_k: _Stub())
    dated = _make([_dt("2020-01-01"), NAT, _dt("2021-05-05T01:02:03Z")])
    dated.save(str(tmp_path / "dated"))
    back = SimilarIssueRetriever.load(str(tmp_path / "dated"))
    assert np.array_equal(back.created_at, dated.created_at, equal_nan=True)
    assert _nums(back.retrieve("q", k=5, created_before=Q)) == [1000, 1001, 1002]

    undated = _make(None, n=3)
    undated.save(str(tmp_path / "undated"))
    meta = joblib.load(str(tmp_path / "undated" / "meta.pkl"))
    assert "created_at" not in meta  # old meta layout preserved
    assert SimilarIssueRetriever.load(str(tmp_path / "undated")).created_at is None


def test_load_rejects_misaligned_dates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(si, "SentenceTransformer", lambda *_a, **_k: _Stub())
    r = _make([_dt("2020-01-01")] * 3)
    r.save(str(tmp_path / "x"))
    meta = joblib.load(str(tmp_path / "x" / "meta.pkl"))
    meta["created_at"] = meta["created_at"][:2]
    joblib.dump(meta, str(tmp_path / "x" / "meta.pkl"))
    with pytest.raises(ValueError, match="length"):
        SimilarIssueRetriever.load(str(tmp_path / "x"))


# --- serving wiring (triage._collect_signals) ---------------------------------------------


def test_collect_signals_passes_the_requests_created_at_to_the_detector() -> None:
    import test_resolution_embeddings_serving as t

    seen: list[dict] = []

    class _Spy(t._PlainRetriever):
        def retrieve(self, text, k=20, exclude_number=None, **kw):  # noqa: ANN001, ANN201
            seen.append(kw)
            return super().retrieve(text, k, exclude_number)

    issue = t._issue()
    t._assistant(t.VSCODE, _Spy(), t._Predictor(t._fit_pca()))._collect_signals(issue)
    assert seen == [{"created_before": issue["created_at"]}]

    seen.clear()
    undated = issue.drop("created_at")
    t._assistant(t.VSCODE, _Spy(), t._Predictor(t._fit_pca()))._collect_signals(undated)
    assert seen == [{}]  # no created_at on the request => no filter argument at all
