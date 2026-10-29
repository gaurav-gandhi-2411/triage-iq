"""Tests for scripts/r8_dup_finetune.py and its kernel: split-safe pairs, leak guard, loss mask."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r8_dup_finetune as ft  # noqa: E402


def _split(d: dict[int, str]) -> pd.Series:
    return pd.Series(d)


def test_train_pairs_respect_splits_and_dates() -> None:
    split = _split({1: "train", 2: "train", 3: "train", 4: "validation", 5: "test", 6: "train"})
    created = pd.Series({1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 5})
    dup = pd.DataFrame(
        {
            "source": [2, 3, 4, 3, 5, 6, 2, 2],
            "canonical": [
                1,
                2,
                1,
                4,
                2,
                3,
                1,
                99,
            ],  # 4,5: held-out end; 6->3 newer; dup row; 99 unknown
        }
    )
    got = ft.train_pairs(dup, split, created)
    assert sorted(map(tuple, got.to_numpy())) == [(2, 1), (3, 2)]


def test_components_union_find_min_id() -> None:
    comp = ft.components(pd.DataFrame({"source": [5, 7, 20], "canonical": [3, 5, 9]}))
    assert comp[5] == comp[3] == comp[7] == 3
    assert comp[20] == comp[9] == 9


def test_assert_train_only_rejects_heldout_and_unknown() -> None:
    split = _split({1: "train", 2: "validation", 3: "test"})
    ft.assert_train_only({1}, split)
    for bad in ({1, 2}, {3}, {1, 77}):
        with pytest.raises(ft.LeakageError):
            ft.assert_train_only(bad, split)


def test_pick_negative_band_fallback_and_empty() -> None:
    rng = np.random.default_rng(42)
    ranked = np.arange(100, 150)
    for _ in range(20):
        assert int(ranked[ft.NEG_LO - 1]) <= ft.pick_negative(ranked, rng) <= int(ranked[-1])
    short = np.array([7, 8, 9])
    assert ft.pick_negative(short, rng) in {7, 8, 9}
    assert ft.pick_negative(np.array([], dtype=int), rng) is None


def test_hard_negatives_are_train_older_and_outside_group() -> None:
    n = 80
    numbers = np.arange(1000, 1000 + n)
    created = np.arange(n)
    base = [f"alpha beta word{i % 7} gamma" for i in range(n)]
    fit = np.arange(n) < 60  # last 20 are held out
    pairs = pd.DataFrame({"source": [1050, 1055], "canonical": [1010, 1011]})
    comp = ft.components(pairs)
    t = SimpleNamespace(numbers=numbers, created=created, base=base, fit_mask=fit)
    neg = ft.hard_negatives(t, pairs, comp)
    for s, c, ng in zip(pairs["source"], pairs["canonical"], neg, strict=True):
        assert ng is not None and ng < s and (ng - 1000) < 60  # older and a train issue
        assert ng not in (s, c)
    frame = ft.pair_frame(pairs, neg, comp)
    assert list(frame["comp"]) == [comp[1050], comp[1055]]
    assert all(v < 0 for v in frame["neg_comp"])  # negatives sit in no duplicate group


def test_metadata_is_private() -> None:
    assert ft.dataset_meta()["isPrivate"] is True
    km = ft.kernel_meta()
    assert km["is_private"] is True and km["code_file"] == "dup_finetune.py"
    assert any(d.endswith(ft.DS_SLUG) for d in km["dataset_sources"])


def test_verdict_needs_gain_and_positive_ci() -> None:
    assert ft.verdict(0.025, 0.001)
    assert not ft.verdict(0.015, 0.005)  # gain below +2.0 points
    assert not ft.verdict(0.05, -0.001)  # CI includes zero
    assert not ft.verdict(-0.03, -0.06)


def test_select_ft_tiebreak_order() -> None:
    def c(cfg: str, kind: str, hit: int, mrr: float, k: int = 0) -> dict:
        return {
            "cfg": {"config": cfg, "kind": kind, **({"k": k} if k else {})},
            "res": {"recall@5": {"k": hit}, "mrr": mrr},
        }

    cands = {
        "a": c("F2", "rrf", 400, 0.3, 10),
        "b": c("F1", "rrf", 400, 0.3, 30),
        "c": c("F1", "dense", 400, 0.3),
        "d": c("F4", "rrf", 399, 0.9, 10),
    }
    assert ft.select_ft(cands) == "c"  # tie on hits and MRR -> dense first
    cands["a"]["res"]["mrr"] = 0.31
    assert ft.select_ft(cands) == "a"  # higher MRR beats simplicity


def _frozen() -> dict:
    return json.loads(ft.FROZEN_FT.read_text(encoding="utf-8"))


def _boom(*_a: object, **_k: object) -> None:
    raise AssertionError("test data was touched before the guard passed")


def _args(**kw: object) -> SimpleNamespace:
    base = {"final_test": True, "frozen_sha256": "", "out": "nowhere", "result": "nowhere.json"}
    return SimpleNamespace(**(base | kw))


def test_committed_frozen_file_is_the_validation_selection() -> None:
    fr = _frozen()
    assert fr["selected"] == "F4_rrf10" and fr["config"]["k"] == 10
    assert fr["config"]["bm25"] == fr["reference_config"]["bm25"]  # frozen #345 BM25 variant
    ft.check_frozen(fr, fr["config_sha256"])  # self-consistent


def test_check_frozen_refuses_wrong_hash_and_tampered_body() -> None:
    fr = _frozen()
    with pytest.raises(ft.FrozenMismatchError):
        ft.check_frozen(fr, "0" * 64)
    tampered = {**fr, "config": {**fr["config"], "k": 30}}
    with pytest.raises(ft.FrozenMismatchError):  # body edited, hash left as committed
        ft.check_frozen(tampered, fr["config_sha256"])


def test_final_test_requires_flags_and_hash_before_touching_data(monkeypatch) -> None:
    monkeypatch.setattr(ft.dd, "Task", _boom)
    for kw in ({"final_test": False, "frozen_sha256": "x"}, {"frozen_sha256": ""}):
        with pytest.raises(ft.qe.TestGuardError):
            ft.cmd_final_test(_args(**kw))
    with pytest.raises(ft.FrozenMismatchError):
        ft.cmd_final_test(_args(frozen_sha256="f" * 64))


def test_final_test_refuses_rerun_when_result_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ft.dd, "Task", _boom)
    done = tmp_path / "res.json"
    done.write_text("{}")
    with pytest.raises(ft.qe.TestGuardError):
        ft.cmd_final_test(_args(frozen_sha256=_frozen()["config_sha256"], result=str(done)))


def test_check_artifacts_refuses_changed_embedding(tmp_path: Path) -> None:
    fr = _frozen()
    (tmp_path / "F4_model").mkdir()
    (tmp_path / "F4_emb.npy").write_bytes(b"not the frozen embedding")
    (tmp_path / "F4_model" / "model.safetensors").write_bytes(b"x")
    (tmp_path / "ids.parquet").write_bytes(b"x")
    with pytest.raises(ft.FrozenMismatchError, match="embedding"):
        ft.check_artifacts(fr, tmp_path)
