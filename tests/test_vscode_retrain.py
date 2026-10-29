from __future__ import annotations

import copy
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
sys.path.insert(0, str(_S.parent / "src"))


def _load(name: str):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, _S / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rl = _load("vscode_retrain_lib")
cli = _load("vscode_retrain")


def _ci(point: float, lo: float) -> dict:
    return {"point": point, "ci95": [lo, point + 0.02], "n": 1000}


def good() -> dict:
    """Challenger clearly better than champion and naive on 1000 holdout issues."""
    return {
        "n_holdout": 1000,
        "paired_vs_champion": {"strict_top3": _ci(0.05, 0.02), "in_family_top3": _ci(0.03, 0.0)},
        "paired_vs_naive": {"strict_top3": _ci(0.30, 0.25)},
    }


# ---------------------------------------------------------------------------------- gate


def test_better_model_is_accepted() -> None:
    d = rl.gate_decision(good(), 20000, 80)
    assert d["decision"] == "accept"
    assert d["reasons"] == [] and all(c["passed"] for c in d["checks"])


def test_worse_model_is_refused() -> None:
    m = good()
    m["paired_vs_champion"]["strict_top3"] = _ci(-0.04, -0.07)
    d = rl.gate_decision(m, 20000, 80)
    assert d["decision"] == "refuse"
    assert any("beats_champion_margin" in r for r in d["reasons"])


def test_equal_model_is_refused() -> None:
    m = good()
    m["paired_vs_champion"]["strict_top3"] = {"point": 0.0, "ci95": [-0.01, 0.01], "n": 1000}
    assert rl.gate_decision(m, 20000, 80)["decision"] == "refuse"


def test_gain_below_margin_is_refused_even_if_ci_positive() -> None:
    m = good()
    m["paired_vs_champion"]["strict_top3"] = _ci(0.005, 0.001)  # significant but < 1 point
    d = rl.gate_decision(m, 20000, 80)
    assert d["decision"] == "refuse" and any("margin" in r for r in d["reasons"])


def test_not_beating_naive_is_refused() -> None:
    m = good()
    m["paired_vs_naive"]["strict_top3"] = _ci(0.01, -0.01)
    assert rl.gate_decision(m, 20000, 80)["decision"] == "refuse"


def test_in_family_regression_is_refused() -> None:
    m = good()
    m["paired_vs_champion"]["in_family_top3"] = _ci(-0.05, -0.08)
    assert rl.gate_decision(m, 20000, 80)["decision"] == "refuse"


@pytest.mark.parametrize("path", [
    ("paired_vs_champion",), ("paired_vs_naive", "strict_top3", "ci95"), ("n_holdout",),
])  # fmt: skip
def test_missing_input_is_refused(path: tuple[str, ...]) -> None:
    m = good()
    cur = m
    for k in path[:-1]:
        cur = cur[k]
    del cur[path[-1]]
    d = rl.gate_decision(m, 20000, 80)
    assert d["decision"] == "refuse" and d["reasons"][0].startswith("input_error")


@pytest.mark.parametrize("args", [(None, 20000, 80), ({}, 20000, 80), (good(), None, 80), (good(), 20000, None)])  # fmt: skip
def test_absent_data_is_refused(args: tuple) -> None:
    assert rl.gate_decision(*args)["decision"] == "refuse"


def test_nan_is_refused() -> None:
    m = copy.deepcopy(good())
    m["paired_vs_champion"]["strict_top3"]["point"] = float("nan")
    assert rl.gate_decision(m, 20000, 80)["decision"] == "refuse"


def test_too_little_data_is_refused() -> None:
    assert rl.gate_decision(good(), 100, 80)["decision"] == "refuse"
    m = good()
    m["n_holdout"] = 50
    assert rl.gate_decision(m, 20000, 80)["decision"] == "refuse"
    assert rl.gate_decision(good(), 20000, 5)["decision"] == "refuse"


def test_gate_stage_refuses_when_inputs_missing(tmp_path: Path) -> None:
    res = cli.stage_gate(Namespace(out=tmp_path))
    assert res["decision"] == "refuse"
    assert json.loads((tmp_path / "gate_decision.json").read_text())["decision"] == "refuse"


def test_stage_refuses_unless_gate_accepted_this_exact_candidate(tmp_path: Path) -> None:
    (tmp_path / "challenger.joblib").write_bytes(b"x")
    (tmp_path / "gate_decision.json").write_text(json.dumps({"decision": "refuse"}))
    a = Namespace(out=tmp_path, manifest=tmp_path / "m")
    with pytest.raises(RuntimeError, match="nothing staged"):
        cli.stage_stage(a)
    (tmp_path / "gate_decision.json").write_text(
        json.dumps({"decision": "accept", "candidate_sha256": "0" * 64})  # hash of a different file
    )
    with pytest.raises(RuntimeError, match="nothing staged"):
        cli.stage_stage(a)
    assert not (tmp_path / "staging").exists()


# ------------------------------------------------------------------- window and split


def test_windows_boundaries_are_contiguous_and_settled() -> None:
    w = rl.make_windows(pd.Timestamp("2026-10-22"))
    assert w.cutoff == pd.Timestamp("2026-09-22", tz="UTC")  # as_of - 30 d
    assert w.holdout_start == pd.Timestamp("2026-07-24", tz="UTC")
    assert w.calib_start == pd.Timestamp("2026-05-25", tz="UTC")
    assert w.train_start == rl.FLOOR  # 48-month rolling start is before the ADR-0070 floor


def test_window_start_rolls_after_the_floor_is_older_than_window() -> None:
    w = rl.make_windows(pd.Timestamp("2030-01-31"), 48)
    assert w.cutoff == pd.Timestamp("2030-01-01", tz="UTC")
    assert w.train_start == pd.Timestamp("2026-01-01", tz="UTC")  # cutoff - 48 months


def test_assign_split_half_open_boundaries() -> None:
    w = rl.make_windows(pd.Timestamp("2026-10-22"))
    ts = pd.Series([
        w.train_start - pd.Timedelta(seconds=1), w.train_start,
        w.calib_start - pd.Timedelta(seconds=1), w.calib_start,
        w.holdout_start - pd.Timedelta(seconds=1), w.holdout_start,
        w.cutoff - pd.Timedelta(seconds=1), w.cutoff,
    ])  # fmt: skip
    got = rl.assign_split(ts, w).tolist()
    assert got == ["", "train", "train", "calibration", "calibration", "holdout", "holdout", ""]


def _write_derived(tmp: Path, w: rl.Windows) -> None:
    rows = [
        # (number, created, labels, text_hash)
        (1, w.train_start + pd.Timedelta(days=1), ["git"], "a"),
        (2, w.calib_start + pd.Timedelta(days=1), ["git"], "a"),  # repeats train text -> dropped
        (3, w.holdout_start + pd.Timedelta(days=1), ["terminal-input"], "c"),
        (4, w.holdout_start + pd.Timedelta(days=2), ["bug"], "d"),  # only excluded label -> dropped
        (5, w.cutoff + pd.Timedelta(days=1), ["git"], "e"),  # unsettled (after cutoff)
        (6, w.train_start - pd.Timedelta(days=1), ["git"], "f"),  # before the window
    ]
    iss = pd.DataFrame({
        "number": [r[0] for r in rows], "created_at": [r[1] for r in rows],
        "title": "t", "body": "b", "labels_final": [r[2] for r in rows],
        "labels_t72h": [r[2] for r in rows], "text_hash": [r[3] for r in rows],
    })  # fmt: skip
    iss.to_parquet(tmp / "issues.parquet")
    pd.DataFrame({
        "number": [1], "ts": [w.train_start + pd.Timedelta(days=1)],
        "action": ["labeled"], "label": ["git"], "actor": ["u"],
    }).to_parquet(tmp / "label_events.parquet")  # fmt: skip


def test_assemble_applies_window_population_and_leak_rules(tmp_path: Path) -> None:
    w = rl.make_windows(pd.Timestamp("2026-10-22"))
    _write_derived(tmp_path, w)
    d = rl.assemble(tmp_path, w)
    assert dict(zip(d["number"], d["split"], strict=True)) == {1: "train", 3: "holdout"}


def test_assemble_empty_window_raises(tmp_path: Path) -> None:
    w = rl.make_windows(pd.Timestamp("2026-10-22"))
    _write_derived(tmp_path, w)
    with pytest.raises(ValueError, match="no issues"):
        rl.assemble(tmp_path, rl.make_windows(pd.Timestamp("2019-01-01"), 1))


def test_assemble_stage_roundtrips_label_sets_through_parquet(tmp_path: Path) -> None:
    w = rl.make_windows(pd.Timestamp("2026-10-22"))
    src = tmp_path / "src"
    src.mkdir()
    _write_derived(src, w)
    a = Namespace(derived_dir=src, as_of="2026-10-22", window_months=48, out=tmp_path / "out")
    meta = cli.stage_assemble(a)
    d, _ = cli._load(a.out)
    assert meta["counts"] == {"train": 1, "holdout": 1}
    assert d.loc[d["number"] == 3, "final"].iloc[0] == frozenset({"terminal-input"})


# ------------------------------------------------------- champion pinned to MANIFEST.sha256


def _champion(tmp: Path, content: bytes = b"champion") -> tuple[Path, Path]:
    """A models dir holding the served pkl, and a manifest pinning exactly its sha256."""
    models = tmp / "models"
    models.mkdir()
    (models / "component_classifier_microsoft_vscode.pkl").write_bytes(content)
    man = tmp / "MANIFEST.sha256"
    man.write_text(
        f"{rl.file_sha256(models / 'component_classifier_microsoft_vscode.pkl')}  "
        "data/models/component_classifier_microsoft_vscode.pkl\n"
        f"{'1' * 64}  data/models/other.pkl\n"
    )
    return models, man


def test_champion_matching_manifest_passes(tmp_path: Path) -> None:
    models, man = _champion(tmp_path)
    actual, expected = rl.verify_champion(models, man)
    assert (
        actual == expected == rl.file_sha256(models / "component_classifier_microsoft_vscode.pkl")
    )


def test_champion_sha_mismatch_errors_and_stages_nothing(tmp_path: Path) -> None:
    models, man = _champion(tmp_path)
    (models / "component_classifier_microsoft_vscode.pkl").write_bytes(b"stale copy")
    with pytest.raises(rl.ChampionShaError, match="!= manifest"):
        rl.verify_champion(models, man)
    out = tmp_path / "out"
    out.mkdir()
    a = Namespace(champion=models, manifest=man, out=out, derived_dir=tmp_path)
    with pytest.raises(rl.ChampionShaError):
        cli.stage_evaluate(a)  # refuses before reading any stage output or scoring
    rc = cli.main(["evaluate", "--derived-dir", str(tmp_path), "--champion", str(models),
                   "--manifest", str(man), "--out", str(out)])  # fmt: skip
    assert rc == cli.ERROR
    assert not (out / "metrics.json").exists() and not (out / "staging").exists()


@pytest.mark.parametrize("manifest_text", [None, "", f"{'2' * 64}  data/models/other.pkl\n"])
def test_champion_missing_manifest_or_entry_fails_closed(
    tmp_path: Path, manifest_text: str | None
) -> None:
    models, man = _champion(tmp_path)
    if manifest_text is None:
        man.unlink()
    else:
        man.write_text(manifest_text)
    with pytest.raises(rl.ChampionShaError):
        rl.verify_champion(models, man)


def test_champion_duplicate_manifest_entry_fails_closed(tmp_path: Path) -> None:
    models, man = _champion(tmp_path)
    man.write_text(man.read_text() * 2)
    with pytest.raises(rl.ChampionShaError, match="exactly one"):
        rl.verify_champion(models, man)


def test_gate_refuses_when_metrics_champion_sha_not_manifest_sha(tmp_path: Path) -> None:
    m = good()
    m["champion"] = {"sha256": "a" * 64, "manifest_sha256": "b" * 64}
    (tmp_path / "metrics.json").write_text(json.dumps(m))
    (tmp_path / "train_info.json").write_text(json.dumps(
        {"n_train": 20000, "classes": list(range(80)), "classes_sha256": "x",
         "challenger_sha256": "y"}))  # fmt: skip
    (tmp_path / "window_meta.json").write_text("{}")
    res = cli.stage_gate(Namespace(out=tmp_path))
    assert res["decision"] == "refuse"
    assert any("champion_sha_mismatch" in r for r in res["reasons"])
    m["champion"]["manifest_sha256"] = "a" * 64
    (tmp_path / "metrics.json").write_text(json.dumps(m))
    res = cli.stage_gate(Namespace(out=tmp_path))
    assert res["decision"] == "accept" and res["champion_manifest_sha256"] == "a" * 64
