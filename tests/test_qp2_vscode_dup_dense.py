"""Tests for scripts/qp2_vscode_dup_dense.py: extraction, rank/RRF math, selection, guards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp2_vscode_dup_dense as dd  # noqa: E402
import qp2_vscode_dup_eval as qe  # noqa: E402


def test_code_tokens_whole_and_subwords() -> None:
    toks = dd.code_tokens("call getFooBar and my_var in workbench.api.Foo but plain words")
    assert {"getfoobar", "get", "foo", "bar", "my_var", "my", "var"} <= set(toks)
    assert "workbench_api_foo" in toks and "workbench" in toks
    assert "plain" not in toks and "words" not in toks


def test_error_lines_cap_and_filter() -> None:
    raw = "\n".join(
        ["fine line", "Error: boom", "x" * 400 + " error"] + [f"failed {i}" for i in range(9)]
    )
    got = dd.error_lines(raw)
    assert got[0] == "Error: boom" and len(got) == dd.MAX_ERR_LINES
    assert all(len(g) <= dd.MAX_ERR_LEN for g in got)


def test_variant_texts_w1_is_baseline_and_extras_append() -> None:
    base, titles, raw = ["T. body"], ["T"], ["T\nError: getFoo failed"]
    assert dd.variant_texts(base, titles, raw, 1, False, False) == base
    assert dd.variant_texts(base, titles, raw, 3, False, False)[0].startswith("T. T. T. body")
    out = dd.variant_texts(base, titles, raw, 1, True, True)[0]
    assert "getfoo" in out and out.endswith("Error: getFoo failed")


def test_full_ranks_ties_go_to_more_recent() -> None:
    s = np.array([0.5, 0.9, 0.5, 0.1, 0.99])
    r = dd.full_ranks(s, 4)  # candidates 0..3 only; 4 is outside the time-respecting prefix
    assert list(r) == [3, 1, 2, 4]  # tie between 0 and 2 -> 2 (more recent) ranks first


def test_rrf_scores_and_fusion_rewards_agreement() -> None:
    a, b = np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0])
    f = dd.rrf_scores(a, b, 60)
    assert f[0] == pytest.approx(2 / 61)
    assert f[0] > f[1] and f[0] > f[2]


def test_select_tiebreak_mrr_then_simplicity() -> None:
    def c(k: int, mrr: float, cfg: dict) -> dict:
        return {"cfg": cfg, "res": {"recall@5": {"k": k}, "mrr": mrr}}

    cands = {
        "v_hi": c(10, 0.3, {"kind": "bm25", "w": 1, "ids": 0, "errs": 0}),
        "v_mrr": c(10, 0.4, {"kind": "bm25", "w": 2, "ids": 1, "errs": 0}),
        "h": c(10, 0.4, {"kind": "rrf", "k": 10}),
        "best": c(11, 0.1, {"kind": "dense"}),
    }
    assert dd.select(cands) == "best"
    del cands["best"]
    assert dd.select(cands) == "v_mrr"  # same R@5 as h and same MRR -> simpler (bm25 < rrf)


def test_test_stage_guards() -> None:
    with pytest.raises(qe.TestGuardError):
        dd.main(["--stage", "test"])
    with pytest.raises(qe.TestGuardError):
        dd.main(["--stage", "test", "--final-test"])  # missing hash


def test_sha_json_is_order_independent() -> None:
    assert dd.sha_json({"a": 1, "b": [1, 2]}) == dd.sha_json({"b": [1, 2], "a": 1})
