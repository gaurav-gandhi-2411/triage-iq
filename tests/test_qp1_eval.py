from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp1e", _S / "qp1_eval.py")
ev = importlib.util.module_from_spec(_SPEC)
sys.modules["qp1e"] = ev
_SPEC.loader.exec_module(ev)

CLASSES = ["a", "b", "c", "d"]
P = np.array(
    [[0.5, 0.3, 0.15, 0.05], [0.1, 0.2, 0.3, 0.4], [0.4, 0.1, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25]]
)
Y = ["a", "c", "b", "d"]
SETS = [frozenset("a"), frozenset("cd"), frozenset("b"), frozenset("ad")]
YI = np.array([0, 2, 1, 3])


def test_topk_and_any_valid_hand_computed() -> None:
    h = ev.hit_arrays(Y, SETS, P, CLASSES)
    assert h["top1"].tolist() == [True, False, False, False]
    assert h["top3"].tolist() == [True, True, False, False]  # tie row ranks a, b, c: d missed
    assert h["anyvalid_top1"].tolist() == [True, True, False, True]
    assert h["anyvalid_top3"].tolist() == [True, True, False, True]


def test_macro_f1_over_classes_present_only() -> None:
    pred = P.argmax(axis=1)  # a, d, a, a
    assert ev.macro_f1(YI, pred, 4) == pytest.approx(0.125)  # only class a scores 0.5
    # a class absent from y_true is ignored even if predicted
    assert ev.macro_f1(np.array([0, 0]), np.array([0, 1]), 4) == pytest.approx(2 / 3)


def test_nll_and_ece_hand_computed() -> None:
    assert ev.nll(P, YI) == pytest.approx(
        -(math.log(0.5) + math.log(0.3) + math.log(0.1) + math.log(0.25)) / 4
    )
    assert ev.ece(P, YI) == pytest.approx(0.3875)  # .125 + .2 + .0625 across bins 7, 6, 3


def test_aps_scores_quantile_and_sets() -> None:
    s = ev.aps_scores(P, YI)
    assert s.tolist() == pytest.approx([0.5, 0.7, 1.0, 1.0])
    assert ev.conformal_qhat(s, 0.05) == math.inf  # ceil(5 * .95) = 5 > n = 4
    assert ev.conformal_qhat(s, 0.5) == pytest.approx(1.0)  # 3rd smallest
    assert ev.conformal_qhat(s, 0.6) == pytest.approx(0.7)  # 2nd smallest
    mask = ev.aps_sets(P, 0.7)
    assert mask.sum(axis=1).tolist() == [1, 2, 2, 2]
    assert mask[0].tolist() == [True, False, False, False] and mask[1].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert ev.aps_sets(P, 0.0).sum(axis=1).tolist() == [1, 1, 1, 1]  # top-1 is always kept


def test_conformal_coverage_known_answer() -> None:
    r = ev.conformal_report(P, Y, SETS, CLASSES, 0.7)
    assert r["coverage_primary"]["k"] == 2 and r["coverage_any_valid"]["k"] == 3
    assert r["set_size_mean"] == 1.75 and r["set_size_median"] == 2.0


def test_temperature_fit_decreases_nll() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(300, 4)) * 4  # overconfident relative to labels below
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(4, p=0.4 * row + 0.15) for row in p])  # noisy labels
    t = ev.fit_temperature(p, y)
    assert t > 1.0
    assert ev.nll(ev.apply_temperature(p, t), y) < ev.nll(p, y)
    assert ev.ece(ev.apply_temperature(p, t), y) < ev.ece(p, y)
    assert ev.apply_temperature(p, 1.0) == pytest.approx(p)


def test_bootstrap_determinism_and_paired_diff() -> None:
    a = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=float)
    stat = lambda i: float(a[i].mean())  # noqa: E731
    r1, r2 = ev.bootstrap(stat, 10), ev.bootstrap(stat, 10)
    assert r1 == r2 and r1["point"] == 0.7 and r1["ci95"][0] <= 0.7 <= r1["ci95"][1]
    cont = np.sqrt(np.arange(1.0, 41.0))  # continuous values: different seeds give different CIs
    c = lambda i: float(cont[i].mean())  # noqa: E731
    assert ev.bootstrap(c, 40, seed=7) != ev.bootstrap(c, 40)
    d = ev.paired_diff(stat, stat, 10)
    assert d["point"] == 0.0 and d["ci95"] == [0.0, 0.0]
    b = np.zeros(10)
    d2 = ev.paired_diff(stat, lambda i: float(b[i].mean()), 10)
    assert d2["point"] == 0.7 and d2["ci95"][0] > 0.2


def test_wilson_known_value_via_evaluate_and_cis() -> None:
    out = ev.evaluate("validation", False, Y, SETS, P, CLASSES, n_boot=50)
    assert out["top1"]["point"] == 0.25 and out["n"] == 4
    assert [round(x, 4) for x in ev.wilson(95, 100)] == [0.8882, 0.9785]


def test_test_guard_and_test_once(tmp_path: Path) -> None:
    with pytest.raises(ev.TestGuardError):
        ev.evaluate("test", False, Y, SETS, P, CLASSES, n_boot=10)
    assert ev.evaluate("test", True, Y, SETS, P, CLASSES, n_boot=10)["split"] == "test"
    manifest, preds, rep = tmp_path / "m.json", tmp_path / "p.parquet", tmp_path / "r.json"
    manifest.write_text('{"x": 1}')
    preds.write_bytes(b"abc")
    w = ev.ResultsWriter(rep, manifest)
    with pytest.raises(ev.TestGuardError):
        w.add("m1", "test", {"top1": 1}, preds)  # final not passed
    w.add("m1", "validation", {"top1": 0.5}, preds)
    w.add("m1", "test", {"top1": 0.4}, preds, final=True)
    w.save()
    saved = json.loads(rep.read_text())["models"]["m1"]
    assert saved["test"]["pred_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert saved["split_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    w2 = ev.ResultsWriter(rep, manifest)  # a fresh process reading the same report file
    with pytest.raises(ev.TestGuardError, match="already evaluated"):
        w2.add("m1", "test", {"top1": 0.9}, preds, final=True)
    w2.add("m2", "test", {"top1": 0.3}, preds, final=True)  # a different model id is fine
