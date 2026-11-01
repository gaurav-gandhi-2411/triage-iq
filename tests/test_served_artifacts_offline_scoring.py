from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import served_artifacts_offline_scoring as s  # noqa: E402


def test_majority_topk_is_strict_and_deterministic() -> None:
    train = ["a"] * 5 + ["b"] * 3 + ["c"] * 3 + ["d"] + ["other"] * 9  # other is never ranked
    out = s.majority_topk(train, ["a", "b", "c", "d", "other", "a"], 3)
    assert out["labels"] == ["a", "b", "c"]  # b/c tie broken by name
    assert (out["k"], out["n"]) == (4, 6)  # a, b, c, a hit; d and gold-other miss
    assert out["wilson95"][0] < out["rate"] < out["wilson95"][1]


def test_majority_topk_empty_test_has_no_rate() -> None:
    out = s.majority_topk(["a"], [], 1)
    assert out["rate"] is None and out["wilson95"] is None


def test_build_dup_queries_counts_unanswerable_and_missing_sources() -> None:
    pairs = pd.DataFrame({"source": [10, 10, 11, 12], "canonical": [1, 2, 3, 4]})
    qs, counts = s.build_dup_queries(pairs, indexed={1, 4, 10}, in_table={10, 11})
    assert [q.source for q in qs] == [10, 11]
    assert qs[0].canonicals_indexed == (1,) and qs[1].canonicals_indexed == ()
    assert counts["n_answerable"] == 1 and counts["n_sources_missing_from_table"] == 1
    assert counts["n_query_issue_itself_in_index"] == 1  # source 10 is in the index


def test_recall_at_k_scores_answerable_only_and_all_queries_denominators() -> None:
    qs = [s.DupQuery(1, (100,), 1), s.DupQuery(2, (200,), 1), s.DupQuery(3, (), 1)]
    ranked = {1: [100, 5, 6], 2: [7, 8, 9, 10, 11, 200]}
    out = s.recall_at_k(qs, lambda q: ranked[q.source], ks=(1, 5, 10))
    assert out["recall@1_answerable"]["k"] == 1 and out["recall@1_answerable"]["n"] == 2
    assert out["recall@5_answerable"]["k"] == 1  # query 2's canonical is at rank 6
    assert out["recall@10_answerable"]["k"] == 2
    assert out["recall@10_all_queries"]["n"] == 3  # the unanswerable query counts as a miss


def test_check_manifest_requires_matching_shas(tmp_path: Path) -> None:
    import hashlib

    d = tmp_path / "idx"
    d.mkdir()
    (d / "index.faiss").write_bytes(b"i")
    (d / "meta.pkl").write_bytes(b"m")
    pins = "\n".join(f"{hashlib.sha256(c).hexdigest()}  data/models/idx/{n}" for n, c in (("index.faiss", b"i"), ("meta.pkl", b"m")))
    man = tmp_path / "MANIFEST.sha256"
    man.write_text(pins + "\n", encoding="utf-8")
    assert set(s.check_manifest(d, man)) == {"index.faiss", "meta.pkl"}
    (d / "meta.pkl").write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        s.check_manifest(d, man)
