from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp1t", _S / "qp1_tfidf_lr.py")
tl = importlib.util.module_from_spec(_SPEC)
sys.modules["qp1t"] = tl
_SPEC.loader.exec_module(tl)

T0 = pd.Timestamp("2024-03-01T00:00:00Z")
WORDS = {"sig/a": "alpha kernel", "sig/b": "bravo network", "sig/rare": "rare zebra"}
TRAIN, VAL, TEST = range(1, 111), range(111, 141), range(141, 171)


def make_data(tmp_path: Path):
    """Synthetic corpus: sig/a, sig/b (>= 30 train issues) and sig/rare (5 -> `other`)."""
    rows, events, split_rows = [], [], []
    for n in [*TRAIN, *VAL, *TEST]:
        lab = "sig/rare" if (n % 20 == 0 and n < 100) else "sig/a" if n % 2 else "sig/b"
        text = f"{WORDS[lab]} {WORDS[lab]} issue token{n % 7} common{n % 3}"
        rows.append({
            "number": n, "created_at": T0, "labels_final": [lab, "kind/bug"], "labels_t72h": [lab],
            "text_hash": f"h{n}", "title": text[:20], "body": text,
        })  # fmt: skip
        events.append(
            {"number": n, "ts": T0 + pd.Timedelta(hours=1), "action": "labeled", "label": lab}
        )
        sp = "train" if n in TRAIN else "validation" if n in VAL else "test"
        split_rows.append({"number": n, "split": sp, "classifier_eligible": n != 5})
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
    ceiling = tmp_path / "ceiling.json"
    ceiling.write_text(json.dumps({"baselines": {"test": {"majority_top1": {"rate": 0.3}, "majority_any_valid": {"rate": 0.3}}},
                                   "ceiling_proxies": {"top1_ceiling_estimate": 0.9}}))  # fmt: skip
    return d, s, manifest, ceiling


def test_selection_key_prefers_top3_then_f1_then_smaller_c() -> None:
    rows = [{"C": 16, "top3": 0.9, "macro_f1": 0.5}, {"C": 4, "top3": 0.9, "macro_f1": 0.6},
            {"C": 64, "top3": 0.85, "macro_f1": 0.9}, {"C": 1, "top3": 0.9, "macro_f1": 0.6}]  # fmt: skip
    assert min(rows, key=tl.sel_key)["C"] == 1  # top3 ties, f1 ties, smaller C wins
    assert min(rows[:3], key=tl.sel_key)["C"] == 4


def test_classes_threshold_other_and_text_input() -> None:
    train = pd.DataFrame({"final": [frozenset({"sig/a"})] * 30 + [frozenset({"sig/x"})] * 29})
    modelled, classes = tl.define_classes(train)
    assert modelled == ["sig/a"] and classes == ["sig/a", "other"]  # 29 < 30
    d = pd.DataFrame(
        {
            "primary": ["sig/a", "sig/x"],
            "final": [frozenset({"sig/a"}), frozenset({"sig/x", "sig/a"})],
        }
    )
    y, sets = tl.gold(d, modelled)
    assert y == ["sig/a", "other"] and sets[1] == frozenset({"sig/a", "other"})


def test_select_never_reads_test_labels_and_final_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d, s, manifest, ceiling = make_data(tmp_path)
    seen: list[set[int]] = []
    real = tl.cl.per_issue

    def spy(issues, events, fam):  # sentinel: record every issue whose labels are computed
        seen.append(set(issues["number"]))
        return real(issues, events, fam)

    monkeypatch.setattr(tl.cl, "per_issue", spy)
    models, report = tmp_path / "models", tmp_path / "report.json"
    info = tl.select(d, s, models)
    assert seen and not set(TEST) & set().union(*seen)  # stage 1 never touched a test issue
    assert not (models / "tfidf_lr" / "test.parquet").exists()
    assert set(info["classes"]) == {"sig/a", "sig/b", "other"} and len(info["grid"]) == 8
    assert 5 not in set(
        pd.read_parquet(models / "tfidf_lr" / "train.parquet")["number"]
    )  # ineligible
    n_calls = len(seen)
    r = tl.final_test(d, s, models, report, manifest, ceiling, n_boot=20)
    assert set(TEST) <= set().union(*seen[n_calls:])  # test labels read only in the final stage
    assert r["top1"]["point"] > 0.9 and r["n"] == len(TEST)  # separable synthetic classes
    saved = json.loads(report.read_text())
    assert saved["models"][tl.MODEL_ID]["test"]["pred_file"] == "test.parquet"
    assert "validation" in saved["models"][tl.MODEL_ID] and len(saved["validation_grid"]) == 8
    cal = saved["models"][tl.MODEL_ID]["test"]["calibration"]
    assert cal["temperature"] > 0 and "ece_after" in cal
    with pytest.raises(tl.ev.TestGuardError, match="already evaluated"):
        tl.final_test(d, s, models, report, manifest, ceiling, n_boot=20)


def test_proba_alignment_and_saved_format(tmp_path: Path) -> None:
    class M:  # model that never saw class `b`
        classes_ = np.array(["a", "c"])

        def predict_proba(self, x):  # noqa: ANN001, ANN201
            return np.array([[0.7, 0.3]] * x.shape[0])

    p = tl.proba_full(M(), np.zeros((2, 1)), ["a", "b", "c"])
    assert p.tolist() == [[0.7, 0.0, 0.3]] * 2
    path = tl.save_proba(tmp_path / "x" / "p.parquet", pd.Series([7, 9]), p, ["a", "b", "c"])
    assert pd.read_parquet(path).columns.tolist() == ["number", "a", "b", "c"]
