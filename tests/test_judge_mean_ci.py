"""scripts/judge_mean_ci.py: checkpoint-keyed judge selection, per-repo CIs, weighted mean (Kish n_eff).

Hermetic: synthetic cassette/checkpoint/eval set under tmp_path; one test replays the committed 64-issue
cassette (JSON reads only) and checks the default output is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import judge_mean_ci as jm  # noqa: E402


def _judge_entry(parent: str, total_each: int) -> dict:
    resp = {d: total_each for d in jm.DIMENSIONS}
    return {
        "request_messages": [{"content": "x GOLD STANDARD: y"}],
        "response": {"content": "noise " + json.dumps(resp) + " tail"},
        "judge_provenance": {"parent_synthesis_key": parent},
    }


def _write_set(tmp_path: Path, scores: dict[str, int], stale: bool = True) -> Path:
    """Set dir with eval_set.jsonl, cassettes/eval_cassette.json and cassettes/recording_checkpoint.json."""
    cdir = tmp_path / "cassettes"
    cdir.mkdir(parents=True)
    entries = {}
    done = {}
    rows = []
    for n, (issue_id, each) in enumerate(scores.items()):
        key = f"syn{n}"
        entries[f"j{n}"] = _judge_entry(key, each)
        done[f"{issue_id}::m::p::a"] = {"issue_id": issue_id, "synthesis_cache_key": key}
        repo = "kubernetes/kubernetes" if issue_id.startswith("k8s") else "microsoft/vscode"
        rows.append(
            {"id": issue_id, "repo": repo, "meta": {"weight": 1.0 + 3.0 * (repo[0] == "k")}}
        )
    if stale:  # a superseded judge entry the checkpoint no longer points at
        entries["old"] = _judge_entry("syn-gone", 0)
    (cdir / "eval_cassette.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
    (cdir / "recording_checkpoint.json").write_text(json.dumps({"done": done}), encoding="utf-8")
    (tmp_path / "eval_set.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return cdir / "eval_cassette.json"


def test_only_checkpoint_keyed_judge_entries_count(tmp_path: Path) -> None:
    cassette = _write_set(tmp_path, {"k8s-1": 2, "k8s-2": 3, "vscode-1": 1})
    out = jm.build(cassette)
    assert out["n"] == 3  # the stale all-zero entry is not counted
    assert out["mean"] == pytest.approx(12.0)  # (12 + 18 + 6) / 3
    assert "per_repo" not in out  # legacy shape unless --detail


def test_detail_adds_per_repo_and_weighted_mean_with_kish_neff(tmp_path: Path) -> None:
    cassette = _write_set(tmp_path, {"k8s-1": 2, "k8s-2": 3, "vscode-1": 1})
    out = jm.build(cassette, with_detail=True)
    assert out["per_repo"]["kubernetes/kubernetes"]["n"] == 2
    assert out["per_repo"]["microsoft/vscode"]["mean"] == pytest.approx(6.0)
    # weights 4, 4, 1: (4*12 + 4*18 + 1*6) / 9 = 14.0 ; n_eff = 81 / 33
    assert out["weighted"]["mean"] == pytest.approx(14.0)
    assert out["weighted"]["kish_n_eff"] == pytest.approx(81 / 33, abs=0.01)


def test_missing_judge_entry_for_a_checkpoint_record_fails_loudly(tmp_path: Path) -> None:
    cassette = _write_set(tmp_path, {"k8s-1": 2, "k8s-2": 3})
    done = json.loads((cassette.parent / "recording_checkpoint.json").read_text(encoding="utf-8"))
    done["done"]["k8s-3::m::p::a"] = {"issue_id": "k8s-3", "synthesis_cache_key": "none"}
    (cassette.parent / "recording_checkpoint.json").write_text(json.dumps(done), encoding="utf-8")
    with pytest.raises(SystemExit, match="no judge entry"):
        jm.build(cassette)


def test_baseline_mismatch_is_rejected(tmp_path: Path) -> None:
    cassette = _write_set(tmp_path, {"k8s-1": 2})
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"overall": {"n": 1, "mean": 9.0}}), encoding="utf-8")
    with pytest.raises(AssertionError):
        jm.build(cassette, baseline=base)


def test_default_output_is_the_committed_report_for_the_64_set() -> None:
    # Reads the committed cassette/baseline only. Compared as parsed JSON (line endings vary by checkout).
    committed = json.loads((ROOT / "reports" / "judge_mean_ci.json").read_text(encoding="utf-8"))
    assert jm.build(jm.CASSETTE, baseline=jm.BASELINE) == committed
