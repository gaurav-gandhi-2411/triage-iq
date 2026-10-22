from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp1ec", _S / "qp1_embedding_candidates.py")
ec = importlib.util.module_from_spec(_SPEC)
sys.modules["qp1ec"] = ec
_SPEC.loader.exec_module(ec)
import qp1_eval as ev  # noqa: E402

CLASSES = ["a", "b", "other"]


def test_knn_vote_toy_primary_and_any() -> None:
    # 2 queries x 4 references; reference classes a, a, b, b
    sims = np.array([[0.9, 0.8, 0.1, 0.0], [0.0, 0.1, 0.7, 0.6]])
    y = np.array([0, 0, 1, 1])
    sets = np.zeros((4, 3), dtype=bool)
    sets[np.arange(4), y] = True
    sets[0, 2] = True  # reference 0 also carries `other`
    p = ec.knn_proba(sims, y, sets, k=2, mode="primary")
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0].argmax() == 0 and p[1].argmax() == 1
    assert p[0, 0] == pytest.approx(1.0, abs=1e-4)  # both neighbours of query 0 are class a
    pa = ec.knn_proba(sims, y, sets, k=2, mode="any")
    assert pa[0, 2] == pytest.approx(0.9 / 2.6, abs=1e-4)  # only reference 0 carries `other`
    assert pa[1, 2] < 1e-4


def test_knn_negative_similarity_is_ignored() -> None:
    sims = np.array([[-0.5, -0.2, 0.3]])
    y, sets = np.array([0, 0, 1]), np.eye(3, dtype=bool)
    p = ec.knn_proba(sims, y, sets, k=3, mode="primary")
    assert p.argmax() == 1 and p[0, 0] < 1e-4


def test_cross_fit_never_uses_held_out_rows_and_covers_all() -> None:
    y = np.repeat(np.arange(3), 20)
    calls: list[tuple[set[int], set[int]]] = []

    def fp(tr: np.ndarray, he: np.ndarray) -> np.ndarray:
        calls.append((set(tr.tolist()), set(he.tolist())))
        out = np.full((len(he), 3), 1 / 3)
        # a memoriser: if a held row were in tr it would be predicted perfectly
        for j, i in enumerate(he):
            if i in set(tr.tolist()):
                out[j] = np.eye(3)[y[i]]
        return out

    oof = ec.cross_fit(fp, y, 3, folds=5)
    assert len(calls) == 5 and all(not (t & h) for t, h in calls)
    assert set().union(*(h for _, h in calls)) == set(range(len(y)))
    assert np.allclose(oof, 1 / 3)  # no row ever saw its own label


def test_stack_features_clip_and_shape() -> None:
    p = np.array([[0.0, 1.0], [0.5, 0.5]])
    f = ec.stack_features(p, p)
    assert f.shape == (2, 4) and np.isfinite(f).all()
    assert f[0, 0] == pytest.approx(np.log(1e-4))


def test_sel_key_orders_top3_then_f1_then_simplicity() -> None:
    base = {"model": "c3a_lr_bge", "top3": 0.9, "macro_f1": 0.5, "param": 4}
    better = {**base, "top3": 0.91}
    tie_f1 = {**base, "macro_f1": 0.6}
    simpler = {**base, "model": "c4_knn_bge"}
    assert min([base, better, tie_f1], key=ec.sel_key) is better
    assert min([base, tie_f1], key=ec.sel_key) is tie_f1
    assert min([base, simpler], key=ec.sel_key) is simpler


def test_paired_vs_c1_identical_models_have_zero_difference() -> None:
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3), size=60)
    y = [CLASSES[i] for i in rng.integers(0, 3, 60)]
    sets = [frozenset([c]) for c in y]
    r = ec.paired_vs_c1(p, p.copy(), y, sets, CLASSES, n_boot=50)
    assert all(v["point"] == 0 and not v["ci_excludes_zero"] for v in r.values())


def test_head_proba_aligns_integer_and_string_labels() -> None:
    from sklearn.linear_model import LogisticRegression

    x = np.array([[0.0], [1.0], [2.0], [3.0]])
    m_int = LogisticRegression().fit(x, [0, 0, 2, 2])
    m_str = LogisticRegression().fit(x, ["a", "a", "other", "other"])
    for m in (m_int, m_str):
        p = ec.head_proba(m, x, CLASSES)
        assert p.shape == (4, 3) and np.allclose(p[:, 1], 0) and np.allclose(p.sum(axis=1), 1)


def test_test_guard_second_evaluation_refused(tmp_path: Path) -> None:
    man, rep, pred = tmp_path / "m.json", tmp_path / "r.json", tmp_path / "p.parquet"
    man.write_text("{}")
    pred.write_bytes(b"x")
    w = ev.ResultsWriter(rep, man)
    with pytest.raises(ev.TestGuardError):
        w.add("c4_knn_bge", "test", {}, pred, final=False)
    w.add("c4_knn_bge", "test", {}, pred, final=True)
    with pytest.raises(ev.TestGuardError):
        w.add("c4_knn_bge", "test", {}, pred, final=True)
