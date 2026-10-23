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
_SPEC = importlib.util.spec_from_file_location("qp1fe", _S / "qp1_finetuned_encoder.py")
fe = importlib.util.module_from_spec(_SPEC)
sys.modules["qp1fe"] = fe
_SPEC.loader.exec_module(fe)
import qp1_eval as ev  # noqa: E402

T0 = pd.Timestamp("2024-03-01T00:00:00Z")
TRAIN, VAL, TEST = range(1, 111), range(111, 141), range(141, 171)
TEST_ONLY_LABEL = "sig/testonly"


def make_data(tmp_path: Path, with_events: bool = True) -> tuple[Path, Path]:
    rows, events, split_rows = [], [], []
    for n in [*TRAIN, *VAL, *TEST]:
        lab = TEST_ONLY_LABEL if n in TEST and n % 3 == 0 else "sig/a" if n % 2 else "sig/b"
        text = f"{lab} issue token{n % 7}"
        rows.append({"number": n, "created_at": T0, "labels_final": [lab], "labels_t72h": [lab],
                     "text_hash": f"h{n}", "title": text[:12], "body": text})  # fmt: skip
        events.append({"number": n, "ts": T0 + pd.Timedelta(hours=1), "action": "labeled", "label": lab})  # fmt: skip
        sp = "train" if n in TRAIN else "validation" if n in VAL else "test"
        split_rows.append({"number": n, "split": sp, "classifier_eligible": n != 150})
    d, s = tmp_path / "derived", tmp_path / "splits"
    d.mkdir()
    s.mkdir()
    pd.DataFrame(rows).to_parquet(d / "issues.parquet")
    if with_events:
        ev_df = pd.DataFrame(events).assign(ts=lambda x: pd.to_datetime(x["ts"], utc=True))
        ev_df.to_parquet(d / "label_events.parquet")
    sp_df = pd.DataFrame(split_rows)
    for name in ("train", "validation", "test"):
        sp_df[sp_df["split"] == name].to_parquet(s / f"{name}.parquet", index=False)
    return d, s


def test_dataset_excludes_test_labels(tmp_path: Path) -> None:
    d, s = make_data(tmp_path)
    fr = fe.build_frames(d, s)
    assert list(fr["test_texts"].columns) == ["number", "text"]  # no label column of any kind
    test_nums = set(fr["test_texts"]["number"])
    assert test_nums == {n for n in TEST if n != 150}  # all eligible test issues, nothing else
    assert not test_nums & set(fr["train"]["number"]) | test_nums & set(fr["validation"]["number"])
    assert TEST_ONLY_LABEL not in fr["classes"]  # test-only label never reaches the class list
    blob = json.dumps({k: fr[k].to_dict("list") for k in ("train", "validation")})
    assert TEST_ONLY_LABEL not in blob
    assert set(fr["train"].columns) == {"number", "text", "primary", "labels"}


def test_test_texts_reads_no_label_table(tmp_path: Path) -> None:
    d, s = make_data(tmp_path, with_events=False)  # label_events.parquet does not exist
    out = fe.eligible_test_texts(d, s)
    assert len(out) == len(TEST) - 1 and out["text"].str.contains("issue").all()


def test_align_by_id_not_by_position() -> None:
    classes = ["a", "b"]
    df = pd.DataFrame({"number": [3, 1, 2], "a": [0.3, 0.1, 0.2], "b": [0.7, 0.9, 0.8]})
    got = fe.align(df, pd.Series([1, 2, 3]), classes)
    assert np.allclose(got[:, 0], [0.1, 0.2, 0.3])
    with pytest.raises(KeyError):
        fe.align(df, pd.Series([1, 4]), classes)
    with pytest.raises(ValueError):
        fe.align(pd.concat([df, df.iloc[:1]]), pd.Series([1]), classes)
    with pytest.raises(ValueError):
        fe.align(df.rename(columns={"b": "c"}), pd.Series([1]), classes)


def test_final_test_is_guarded_and_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d, s = make_data(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    d_all = fe.tl.load_split(d, s, ("train", "validation"))
    modelled, classes = fe.tl.define_classes(d_all[d_all["split"] == "train"])
    va = d_all[(d_all["split"] == "validation") & d_all["classifier_eligible"]].reset_index(
        drop=True
    )
    te = fe.tl.load_split(d, s, ("test",))
    te = te[te["classifier_eligible"]].reset_index(drop=True)
    rng = np.random.default_rng(0)

    def probs(df: pd.DataFrame) -> pd.DataFrame:
        p = rng.dirichlet(np.ones(len(classes)), len(df))
        return pd.DataFrame(p, columns=classes).assign(number=df["number"].to_numpy())[["number", *classes]].iloc[::-1]  # fmt: skip

    probs(va).to_parquet(out / "validation_probs.parquet", index=False)
    probs(te).to_parquet(out / "test_probs.parquet", index=False)
    c1 = tmp_path / "c1"
    c1.mkdir()
    probs(te).to_parquet(c1 / "test.parquet", index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    p = {"derived": d, "splits": s, "out": out, "c1_dir": c1, "manifest": manifest, "report": tmp_path / "r.json"}  # fmt: skip
    w = ev.ResultsWriter(p["report"], manifest)
    w.data["classes"] = classes
    w.save()
    res = fe.final_test(p, n_boot=20, latency=False)
    assert (
        fe.MODEL_ID in res
        and "ci_excludes_zero" in res[fe.MODEL_ID]["paired_vs_candidate1"]["top3"]
    )
    assert res[fe.MODEL_ID]["conformal_exploratory"]["label"].startswith("EXPLORATORY")
    with pytest.raises(ev.TestGuardError):  # second test evaluation of the same model id
        fe.final_test(p, n_boot=20, latency=False)
    # and the guard fires before any test label is read
    monkeypatch.setattr(fe.tl, "load_split", lambda *a, **k: pytest.fail("test labels read"))
    with pytest.raises(ev.TestGuardError):
        fe.final_test(p, n_boot=20, latency=False)


def test_dataset_text_equals_serving_recipe(tmp_path: Path) -> None:
    from triage_iq.data.preprocess import clean_text
    from triage_iq.models.component_classifier import _build_text

    d, s = make_data(tmp_path)
    iss = pd.read_parquet(d / "issues.parquet")
    iss["title"] = "  Title " + iss["number"].astype(str) + " "
    iss["body"] = "<!-- tpl -->  real body\n```\ncode\n```\n" + iss["number"].astype(str)
    iss.to_parquet(d / "issues.parquet")
    fr = fe.build_frames(d, s)
    by_num = iss.set_index("number")
    for key in ("train", "validation", "test_texts"):
        for n, text in zip(fr[key]["number"].head(5), fr[key]["text"].head(5), strict=True):
            title, body = by_num.loc[n, "title"], by_num.loc[n, "body"]
            want = _build_text(pd.Series([title]), pd.Series([clean_text(body)[0]])).iloc[0]
            assert text == want
            assert "<!--" not in text and text.startswith("Title ")
