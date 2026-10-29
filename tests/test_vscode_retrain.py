from __future__ import annotations

import copy
import importlib.util
import json
import re
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
    a = Namespace(
        champion=models, manifest=man, out=out, derived_dir=tmp_path, champion_manifest_key=None
    )
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


# ------------------------------------------------ ADR-0073 meta-label exclusion (D92.2)

META = ("ai-generated", "ai-triaged", "2026-themes")


LABEL_TABLE = (
    Path(__file__).resolve().parents[1] / "docs/architecture/adr/0073-vscode-label-decisions.md"
)


DECISIONS = ("EXCLUDE", "BORDERLINE-EXCLUDE", "BORDERLINE-KEEP", "KEEP")


def _table() -> tuple[list[tuple[str, str, str]], list[str], list[str]]:
    """Parse the ADR-0073 decision file: (label, decision, NEW) rows, frozen list, KEEP list."""
    rows: list[tuple[str, str, str]] = []
    lists: dict[str, list[str]] = {}
    section = ""
    for line in LABEL_TABLE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) == 6 and cells[3] in DECISIONS:
            rows.append((cells[0], cells[3], cells[4]))
        elif (
            section.startswith(("## Already", "## KEEP")) and line and not line.startswith("Reason")
        ):
            key = "frozen" if section.startswith("## Already") else "keep"
            lists[key] = [re.sub(r" \(\d+\)$", "", x) for x in line.split("; ")]
    return rows, lists["frozen"], lists["keep"]


def test_meta_exclusion_equals_the_adr_table() -> None:
    rows, frozen, keep = _table()
    new = {lab.lower() for lab, dec, flag in rows if flag == "yes"}
    assert new == rl.RETRAIN_META_EXCLUDE
    for lab, dec, flag in rows:
        assert (flag == "yes") == dec.endswith("EXCLUDE"), lab
        assert rl.tax.not_excluded(lab), lab  # NEW means not already excluded by ADR-0070
    assert frozen and all(rl.tax.is_excluded(x) for x in frozen)
    assert keep and all(rl.tax.not_excluded(x) for x in keep)
    everything = [r[0] for r in rows] + frozen + keep
    assert len(everything) == len(set(everything)) > 600  # one entry per distinct label
    assert not {x.lower() for x in keep} & rl.RETRAIN_META_EXCLUDE
    assert {"recent-regression", "stable-anomaly", "tree-sitter-autoindent-candidate"} <= new


def test_meta_exclusion_matching() -> None:
    assert frozenset(META) <= rl.RETRAIN_META_EXCLUDE
    assert all(not rl.retrain_family(x) for x in (*META, "AI-Generated", "2026-THEMES"))
    for ok in ("git", "terminal-input", "ai-generated-x", "UBB"):
        assert rl.retrain_family(ok)  # other labels unaffected, incl. near-names
    assert not rl.retrain_family("bug")  # the ADR-0070 list still applies


def test_meta_labels_never_become_classes_and_meta_only_issues_are_dropped(
    tmp_path: Path,
) -> None:
    w = rl.make_windows(pd.Timestamp("2026-10-22"))
    t0 = w.train_start + pd.Timedelta(days=1)
    rows = []  # (labels): 35 git+meta, 35 meta-only (would reach >= 30 as a class), 35 terminal
    for i in range(35):
        rows.append((["git", META[i % 3]], f"g{i}"))
        rows.append(([META[i % 3], META[(i + 1) % 3]], f"m{i}"))
        rows.append((["terminal-input"], f"t{i}"))
    rows.append((["AI-Generated"], "case"))
    n = len(rows)
    pd.DataFrame({
        "number": range(1, n + 1), "created_at": [t0 + pd.Timedelta(minutes=i) for i in range(n)],
        "title": "t", "body": "b", "labels_final": [r[0] for r in rows],
        "labels_t72h": [r[0] for r in rows], "text_hash": [r[1] for r in rows],
    }).to_parquet(tmp_path / "issues.parquet")  # fmt: skip
    pd.DataFrame({
        "number": [1], "ts": [t0], "action": ["labeled"], "label": ["git"], "actor": ["u"],
    }).to_parquet(tmp_path / "label_events.parquet")  # fmt: skip
    d = rl.assemble(tmp_path, w)
    assert len(d) == 70  # the 36 meta-only issues (35 + the upper-case one) have no family
    assert all(not (set(s) & set(META)) for s in d["final"])
    assert {frozenset(s) for s in d["final"]} == {frozenset({"git"}), frozenset({"terminal-input"})}
    classes = rl.regenerate_classes(d)
    assert set(META).isdisjoint(classes)
    assert "git" in classes and classes[-1] == "other"


# ------------------------------------- #326: bundle downloaded under a name that is not its manifest name

BUNDLE_KEY = "data/models/component_classifier_microsoft_vscode_family_v2_2026-10-22.joblib"


def _bundle(tmp: Path, content: bytes = b"bundle") -> tuple[Path, Path]:
    """champion.joblib (download name) and a manifest pinning its sha256 under BUNDLE_KEY."""
    f = tmp / "champion.joblib"
    f.write_bytes(content)
    man = tmp / "MANIFEST.sha256"
    man.write_text(f"{rl.file_sha256(f)}  {BUNDLE_KEY}\n{'1' * 64}  data/models/other.pkl\n")
    return f, man


def test_downloaded_name_without_key_still_fails_closed(tmp_path: Path) -> None:
    f, man = _bundle(tmp_path)
    with pytest.raises(rl.ChampionShaError, match="champion.joblib"):
        rl.verify_champion(f, man)


def test_explicit_manifest_key_verifies_by_sha_not_by_file_name(tmp_path: Path) -> None:
    f, man = _bundle(tmp_path)
    actual, expected = rl.verify_champion(f, man, BUNDLE_KEY)
    assert actual == expected == rl.file_sha256(f)


def test_explicit_manifest_key_still_requires_matching_sha(tmp_path: Path) -> None:
    f, man = _bundle(tmp_path)
    f.write_bytes(b"tampered bundle")
    with pytest.raises(rl.ChampionShaError, match="!= manifest"):
        rl.verify_champion(f, man, BUNDLE_KEY)


@pytest.mark.parametrize(
    "key",
    [
        "data/models/other.pkl",  # listed in the manifest but not the vscode classifier
        "data/models/component_classifier_microsoft_vscode_missing.joblib",  # not in the manifest
        "../data/models/component_classifier_microsoft_vscode_x.joblib",
        "data/models/component_classifier_microsoft_vscode_x.joblib extra",
        "",
    ],
)
def test_explicit_manifest_key_must_be_a_listed_vscode_classifier(tmp_path: Path, key: str) -> None:
    f, man = _bundle(tmp_path)
    with pytest.raises(rl.ChampionShaError):
        rl.verify_champion(f, man, key)


def test_cli_passes_manifest_key_through_to_evaluation(tmp_path: Path) -> None:
    f, man = _bundle(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    base = ["evaluate", "--derived-dir", str(tmp_path), "--champion", str(f),
            "--manifest", str(man), "--out", str(out)]  # fmt: skip
    # Without the key the sha check refuses before any scoring (exit 2, nothing written).
    assert cli.main(base) == cli.ERROR
    # With the key the sha check passes and the run proceeds to the (absent) stage inputs, which
    # is still an error here, but NOT the champion error: it is reached only after verification.
    a = Namespace(champion=f, manifest=man, out=out, derived_dir=tmp_path,
                  champion_manifest_key=BUNDLE_KEY)  # fmt: skip
    with pytest.raises(
        RuntimeError, match="assemble output missing"
    ):  # past verify_champion, into reading stage outputs
        cli.stage_evaluate(a)
