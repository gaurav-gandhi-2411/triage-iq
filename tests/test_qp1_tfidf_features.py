from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp1f", _S / "qp1_tfidf_features.py")
tf = importlib.util.module_from_spec(_SPEC)
sys.modules["qp1f"] = tf
_SPEC.loader.exec_module(tf)
t1 = tf.t1

T0 = pd.Timestamp("2024-03-01T00:00:00Z")
WORDS = {"sig/a": "alpha kernel", "sig/b": "bravo network", "sig/rare": "rare zebra"}
TRAIN, VAL, TEST = range(1, 111), range(111, 141), range(141, 171)


def make_data(tmp_path: Path):
    rows, events, split_rows = [], [], []
    for n in [*TRAIN, *VAL, *TEST]:
        lab = "sig/rare" if (n % 20 == 0 and n < 100) else "sig/a" if n % 2 else "sig/b"
        title = f"{WORDS[lab]} issue{n % 5}"
        body = f"{WORDS[lab]} see pkg/kubelet/cm/x.go kubectl apply failed to get pod{n % 3} token{n % 7}"
        rows.append({
            "number": n, "created_at": T0, "labels_final": [lab], "labels_t72h": [lab],
            "text_hash": f"h{n}", "title": title, "body": body,
        })  # fmt: skip
        events.append(
            {"number": n, "ts": T0 + pd.Timedelta(hours=1), "action": "labeled", "label": lab}
        )
        sp = "train" if n in TRAIN else "validation" if n in VAL else "test"
        split_rows.append({"number": n, "split": sp, "classifier_eligible": True})
    d, s = tmp_path / "derived", tmp_path / "splits"
    d.mkdir()
    s.mkdir()
    pd.DataFrame(rows).to_parquet(d / "issues.parquet")
    pd.DataFrame(events).assign(ts=lambda x: pd.to_datetime(x["ts"], utc=True)).to_parquet(
        d / "label_events.parquet"
    )
    sp_df = pd.DataFrame(split_rows)
    for name in ("train", "validation", "test"):
        sp_df[sp_df["split"] == name].to_parquet(s / f"{name}.parquet", index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    return d, s, manifest


def test_extra_tokens_known_output() -> None:
    text = "crash in pkg/kubelet/cm/container_manager.go and staging/src/k8s.io/client-go/tools. Run kubectl apply -f x; kubeadm is down. Error: failed to get pod 12345 abcdef0123456789 now"  # noqa: E501
    toks = tf.extra_tokens(text).split()
    assert {"p:pkg", "p:kubelet", "p:cm", "pp:pkg/kubelet", "pp:pkg/kubelet/cm"} <= set(toks)
    assert "p:k8s" in toks and "p:client-go" in toks
    assert "c:kubectl" in toks and "c:kubectl_apply" in toks
    assert "c:kubeadm" in toks and "c:kubeadm_is" not in toks  # only allow-listed sub-commands
    assert "e:error_failed_to_get_pod" in toks  # digits and long hex stripped, 5-word stem cap
    assert tf.extra_tokens("nothing to see here") == ""


def test_featurizer_blocks_and_widths() -> None:
    texts = pd.Series(
        [
            f"title {i % 3} words\nbody about pkg/kubelet/x kubectl get and more {i % 4}"
            for i in range(30)
        ]
    )
    base = tf.Featurizer(False, False).fit_transform(texts)
    full = tf.Featurizer(True, True)
    x = full.fit_transform(texts)
    assert (
        x.shape[0] == 30 and x.shape[1] > base.shape[1] and len(full.blocks) == 4
    )  # title, body, char, paths
    assert (
        len(tf.Featurizer(True, False).blocks) == 3 and len(tf.Featurizer(False, True).blocks) == 2
    )
    assert full.transform(texts).shape == x.shape


def test_selection_key_and_candidate_split() -> None:
    rows = [
        {
            "C": 256,
            "class_weight": None,
            "title_sep": False,
            "paths": False,
            "n_features": 100,
            "top3": 0.9,
            "macro_f1": 0.5,
        },
        {
            "C": 64,
            "class_weight": None,
            "title_sep": True,
            "paths": False,
            "n_features": 300,
            "top3": 0.9,
            "macro_f1": 0.5,
        },
        {
            "C": 64,
            "class_weight": None,
            "title_sep": False,
            "paths": True,
            "n_features": 200,
            "top3": 0.9,
            "macro_f1": 0.5,
        },
        {
            "C": 64,
            "class_weight": None,
            "title_sep": False,
            "paths": True,
            "n_features": 150,
            "top3": 0.8,
            "macro_f1": 0.9,
        },
    ]
    assert tf.cfg(tf.pick(rows)) == (
        64,
        None,
        False,
        True,
    )  # top3 & f1 tie -> smaller C -> fewer features
    assert tf.pick(rows[:2])["C"] == 64
    assert (
        len(tf.GRID) == 24 and sum(1 for g in tf.GRID if not (g[2] or g[3])) == 6
    )  # 6 plain rows = candidate 1b


def test_grid_resumes_and_final_test_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d, s, manifest = make_data(tmp_path)
    models, report, c1_report = tmp_path / "models", tmp_path / "r.json", tmp_path / "c1.json"
    t1.select(d, s, models)  # candidate 1 artefacts (probabilities) on the fixture
    t1.final_test(d, s, models, c1_report, manifest, _ceiling(tmp_path), n_boot=20)
    grid = [
        (64, "balanced", False, False),
        (64, None, True, False),
        (64, None, False, True),
        (256, "balanced", True, True),
    ]
    fits: list[int] = []
    real = t1.fit_lr
    monkeypatch.setattr(t1, "fit_lr", lambda *a, **k: (fits.append(1), real(*a, **k))[1])
    monkeypatch.setattr(tf, "GRID", grid[:2])
    st = tf.run_grid(d, s, models, 1e9)
    assert st["complete"] and len(fits) == 2
    monkeypatch.setattr(tf, "GRID", grid)
    fits.clear()
    st = tf.run_grid(d, s, models, 1e9)
    assert (
        len(fits) == 2 and len(st["rows"]) == 4 and st["complete"]
    )  # resumed: only the missing rows
    assert "candidate_2" in st["selection"] and "plain_extC_best" in st["selection"]
    seen_test: list[bool] = []
    real_load = t1.load_split
    monkeypatch.setattr(
        t1,
        "load_split",
        lambda dd, ss, names: (seen_test.append("test" in names), real_load(dd, ss, names))[1],
    )
    out = tf.final_test(d, s, models, report, manifest, c1_report, n_boot=20)
    assert seen_test[-1] is True and seen_test.count(True) == 1  # test labels loaded once, last
    saved = json.loads(report.read_text())
    assert (
        "test" in saved["models"]["tfidf_feat_v1"]
        and "artifact_mb" in saved["models"]["tfidf_feat_v1"]["test"]["timing"]
    )
    assert set(out["paired_vs_candidate_1"]["tfidf_feat_v1"]) == {"top1", "top3", "anyvalid_top1"}
    assert (
        "selected" in out["conformal_test"]["tfidf_lr_v1"]
        and "fixed_top3" in out["conformal_test"]["tfidf_feat_v1"]
    )
    assert saved["candidate_1_reused"]["source"] == "c1.json"
    assert "LAC" in saved["statement"] and "widened" in saved["statement"]
    with pytest.raises(tf.ev.TestGuardError, match="already evaluated"):
        tf.final_test(d, s, models, report, manifest, c1_report, n_boot=20)


def _ceiling(tmp_path: Path) -> Path:
    p = tmp_path / "ceiling.json"
    p.write_text(json.dumps({"baselines": {"test": {"majority_top1": {"rate": 0.3}, "majority_any_valid": {"rate": 0.3}}},
                             "ceiling_proxies": {"top1_ceiling_estimate": 0.9}}))  # fmt: skip
    return p
