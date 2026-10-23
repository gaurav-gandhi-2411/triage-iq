from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp1vlib", _S / "qp1_vscode_lib.py")
vs = importlib.util.module_from_spec(_SPEC)
sys.modules["qp1vlib"] = vs
_SPEC.loader.exec_module(vs)

CLASSES = ["editor", "git", "terminal", "other"]
OTHER = 3


def test_class_list_hash_is_asserted_against_adr_0070() -> None:
    train = pd.DataFrame({"final": [frozenset({"git"})] * 30})
    with pytest.raises(AssertionError, match="ADR-0070"):
        vs.define_classes(train)  # 1-class synthetic list cannot match the frozen 85-class hash


def test_gold_maps_through_parent_of_and_unknown_to_other() -> None:
    assert vs.to_class("terminal-input", CLASSES) == "terminal"  # prefix rule
    assert vs.to_class("editor", CLASSES) == "editor"
    assert vs.to_class("NES", CLASSES) == "other"  # parent not in the class list
    assert vs.to_class("chat-billing", CLASSES) == "other"  # `chat` is not in this toy list
    d = pd.DataFrame(
        {
            "primary": ["git", "NES"],
            "final": [frozenset({"git"}), frozenset({"NES", "terminal-tabs"})],
        }
    )
    y, sets = vs.gold(d, CLASSES)
    assert y == ["git", "other"] and sets[1] == frozenset({"other", "terminal"})


def test_strict_never_credits_other_but_lenient_does() -> None:
    # issue 0: gold git, predicted git. issue 1: gold other, predicted other. issue 2: gold git, miss
    p = np.array([[0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7], [0.5, 0.1, 0.1, 0.3]])
    y = np.array([1, OTHER, 1])
    sm = np.zeros((3, 4), dtype=bool)
    sm[np.arange(3), y] = True
    h = vs.hit_arrays(y, sm, vs.rank_proba(p, OTHER), OTHER)
    assert h["strict_top1"].tolist() == [True, False, False]
    assert h["lenient_top1"].tolist() == [True, True, False]
    assert h["strict_top3"].tolist() == [True, False, True]  # git is 3rd for issue 2
    assert h["lenient_top3"].tolist() == [True, True, True]
    assert h["abstain"].tolist() == [False, True, False]
    assert h["gold_in_family"].tolist() == [True, False, True]


def test_family_only_abstention_leads_lenient_list_with_other() -> None:
    p = np.array([[0.4, 0.3, 0.3, 0.0], [0.9, 0.05, 0.05, 0.0]])
    rk = vs.rank_proba(p, OTHER, abstain=p.max(axis=1) < 0.5)
    assert rk.strict[0].tolist() == [0, 1, 2] and rk.lenient[0].tolist() == [OTHER, 0, 1]
    assert rk.lenient[1].tolist() == rk.strict[1].tolist() == [0, 1, 2]


def test_score_test_requires_final_and_writer_is_test_once(tmp_path: Path) -> None:
    p = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]] * 5)
    y = ["editor", "other"] * 5
    sets = [frozenset({c}) for c in y]
    rk = vs.rank_proba(p, OTHER)
    with pytest.raises(vs.ev.TestGuardError):
        vs.score("test", False, y, sets, rk, CLASSES, n_boot=10)
    s = vs.score("test", True, y, sets, rk, CLASSES, n_boot=10)
    assert s["strict_top1"]["point"] == 0.5 and s["lenient_top1"]["point"] == 0.5
    assert s["in_family_top1"]["point"] == 1.0 and s["coverage_bound_strict"]["k"] == 5
    assert s["abstention"]["precision"]["rate"] is None  # nothing abstained
    assert s["abstention"]["recall"]["rate"] == 0.0
    manifest = tmp_path / "m.json"
    manifest.write_text("{}")
    pred = tmp_path / "p.parquet"
    pred.write_bytes(b"x")
    w = vs.ev.ResultsWriter(tmp_path / "r.json", manifest)
    w.add("m", "test", s, pred, final=True)
    with pytest.raises(vs.ev.TestGuardError):
        w.add("m", "test", s, pred, final=True)


def test_abstention_tau_and_conformal_choice() -> None:
    pmax = np.array([0.2, 0.25, 0.9, 0.95, 0.8, 0.85])
    is_other = np.array([True, True, False, False, False, False])
    tau = vs.best_abstention_tau(pmax, is_other)
    assert ((pmax < tau) == is_other).all()
    table = [
        {"spec": {"name": "b"}, "coverage": {"mean": 0.91}, "set_size_mean": {"mean": 3.0}},
        {"spec": {"name": "a"}, "coverage": {"mean": 0.89}, "set_size_mean": {"mean": 2.0}},
        {"spec": {"name": "c"}, "coverage": {"mean": 0.99}, "set_size_mean": {"mean": 1.0}},
    ]
    assert vs.conformal_choose(table, 0.90)["spec"]["name"] == "a"  # c is outside 0.90 +/- 0.02
    assert vs.conformal_choose(table[2:], 0.90)["spec"]["name"] == "c"  # none qualifies: closest


def test_majority_prior_and_selection_key() -> None:
    p, prior = vs.majority_proba(["git", "git", "other"], CLASSES, 2)
    assert p.shape == (2, 4) and prior[1] == pytest.approx(2 / 3)
    hi = {"strict_top3": {"point": 0.5}, "macro_f1": {"point": 0.1}}
    lo = {"strict_top3": {"point": 0.5}, "macro_f1": {"point": 0.1}}
    assert vs.sel_key(hi, vs.IDS["v0"]) < vs.sel_key(lo, vs.IDS["v1"])  # tie -> simpler


def test_conformal_select_alpha_sets_target_window() -> None:
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), size=400)
    y = np.array([int(rng.choice(4, p=r)) for r in p])
    for alpha in (0.1, 0.05):
        res = vs.conformal_select(p, y, reps=3, alpha=alpha)
        assert res["alpha"] == alpha
        assert vs.conformal_choose(res["table"], 1 - alpha)["spec"] == res["selected"]
