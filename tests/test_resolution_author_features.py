"""engineer_features' author-history loop must match the original iterrows implementation exactly.

2026-09-24: the per-request author-history loop was rewritten from all_df.iterrows() to plain
column lists (production System 3 latency: ~3 s per k8s request, ~0.6 s per vscode request).
The resolution estimate feeds the LLM prompt, so any numeric drift would silently change prompts
and break cassette replay. This pins the new code to the original algorithm on the edge cases
that matter: NaN and None authors, created_at ties, a query whose number collides with a
training row, open vs closed rows, and an author-less query (what the API sends).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from triage_iq.models.resolution import engineer_features


def _reference_author_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """The pre-2026-09-24 implementation, verbatim apart from returning only its three columns."""
    all_df = pd.concat([train_df, df], sort=False).drop_duplicates(subset=["number"])
    all_df = all_df.sort_values("created_at")
    author_prior_count: dict = {}
    author_prior_resolutions: dict = {}
    author_count_at = {}
    author_median_at = {}
    for _, row in all_df.iterrows():
        num = row["number"]
        author = row.get("author", "")
        count = author_prior_count.get(author, 0)
        medians = author_prior_resolutions.get(author, [])
        author_count_at[num] = count
        author_median_at[num] = np.median(medians) if medians else np.nan
        author_prior_count[author] = count + 1
        if pd.notna(row.get("resolution_hours")) and row.get("state") == "closed":
            author_prior_resolutions.setdefault(author, []).append(row["resolution_hours"])
    out = pd.DataFrame(index=df.index)
    out["author_prior_count"] = df["number"].map(author_count_at).fillna(0)
    out["is_first_author"] = (out["author_prior_count"] == 0).astype(int)
    out["author_prior_median_hrs"] = df["number"].map(author_median_at).fillna(-1)
    return out


def _train() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    authors = rng.choice(["alice", "bob", "carol", "dave"], size=n).astype(object)
    authors[rng.choice(n, 25, replace=False)] = np.nan   # k8s has 125 NaN authors
    authors[rng.choice(n, 10, replace=False)] = None
    base = pd.Timestamp("2015-01-01", tz="UTC")
    created = [base + pd.Timedelta(days=int(d)) for d in rng.integers(0, 120, size=n)]  # many ties
    return pd.DataFrame({
        "number": np.arange(1, n + 1),
        "title": ["t"] * n,
        "body_clean": ["b"] * n,
        "author": authors,
        "created_at": created,
        "resolution_hours": np.where(rng.random(n) < 0.2, np.nan, rng.gamma(2.0, 50.0, size=n)),
        "state": rng.choice(["closed", "open"], size=n, p=[0.8, 0.2]),
    })


def _assert_same(df: pd.DataFrame, train: pd.DataFrame) -> None:
    got, _ = engineer_features(df, train_df=train)
    want = _reference_author_features(df, train)
    for col in ("author_prior_count", "is_first_author", "author_prior_median_hrs"):
        assert_series_equal(got[col], want[col], check_exact=True, check_names=False)


def test_api_shaped_query_matches_reference() -> None:
    train = _train()
    q = pd.DataFrame([{"number": -1, "title": "x", "body_clean": "y",
                       "created_at": pd.Timestamp("2015-03-01", tz="UTC")}])
    _assert_same(q, train)


def test_training_rows_as_queries_match_reference() -> None:
    train = _train()
    for idx in (0, 7, 42, 199, 399):   # includes NaN/None authors via the fixed seed
        _assert_same(train.iloc[[idx]].copy(), train)


def test_nan_and_none_author_queries_match_reference() -> None:
    train = _train()
    for pick in (train["author"].isna(),):
        for idx in np.flatnonzero(pick.to_numpy())[:6]:
            _assert_same(train.iloc[[idx]].copy(), train)


def test_batch_matches_reference() -> None:
    train = _train()
    _assert_same(train.sample(60, random_state=42).copy(), train)
