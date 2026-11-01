from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vscode_family_v2_served_holdout_report as r  # noqa: E402


def _run(tmp: Path, payload: bytes, pin: bytes) -> Path:
    run = tmp / "run"
    (run / "staging").mkdir(parents=True)
    (run / "challenger.joblib").write_bytes(payload)
    (run / "staging" / r.BUNDLE).write_bytes(payload)
    blk = {"strict_top3": {"point": 0.5}, "strict_top1": {"point": 0.3}, "in_family_top3": {"point": 0.7}}
    (run / "metrics.json").write_text(json.dumps({
        "n_holdout": 10, "n_in_family": 8, "challenger": blk, "champion": {**blk, "sha256": "x"}, "naive": blk,
        "paired_vs_champion": {}, "paired_vs_naive": {}}), encoding="utf-8")
    (run / "window_meta.json").write_text(json.dumps({
        "as_of": "2026-10-22", "windows": {"holdout_start": "a", "cutoff": "b"}, "counts": {"holdout": 10}}), encoding="utf-8")
    (run / "code_sha.txt").write_text("abc123\n", encoding="utf-8")
    (tmp / "MANIFEST.sha256").write_text(f"{hashlib.sha256(pin).hexdigest()}  data/models/{r.BUNDLE}\n", encoding="utf-8")
    return run


def test_report_built_when_the_recomputed_sha_matches_the_manifest(tmp_path: Path) -> None:
    run = _run(tmp_path, b"bundle", b"bundle")
    rep = r.build_report(run, tmp_path / "MANIFEST.sha256")
    assert rep["served_bundle"]["all_match_manifest"] is True
    assert rep["challenger_served"]["strict_top3"] == {"point": 0.5}
    assert rep["code_sha"] == "abc123"


def test_a_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    run = _run(tmp_path, b"bundle", b"another bundle")
    with pytest.raises(r.ProvenanceError):
        r.build_report(run, tmp_path / "MANIFEST.sha256")


def test_a_missing_manifest_entry_fails_closed(tmp_path: Path) -> None:
    run = _run(tmp_path, b"bundle", b"bundle")
    (tmp_path / "MANIFEST.sha256").write_text("", encoding="utf-8")
    with pytest.raises(r.ProvenanceError):
        r.build_report(run, tmp_path / "MANIFEST.sha256")
