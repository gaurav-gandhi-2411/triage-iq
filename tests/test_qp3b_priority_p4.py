from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import pytest

_T = Path(__file__).resolve().parent
_S = _T.parent / "scripts"
sys.path.insert(0, str(_S))
for _n, _p in (
    ("qp3bp3t", _T / "test_qp3b_priority_p3.py"),
    ("qp3bp4", _S / "qp3b_priority_p4.py"),
):
    _spec = importlib.util.spec_from_file_location(_n, _p)
    _m = importlib.util.module_from_spec(_spec)
    sys.modules[_n] = _m
    _spec.loader.exec_module(_m)
p4, fixture = sys.modules["qp3bp4"], sys.modules["qp3bp3t"]


def test_p4_validate_test_once_and_artifacts(tmp_path: Path) -> None:
    derived, splits, models, report, manifest = fixture.make_data(tmp_path)
    p4.pr.validate(derived, splits, models, report, manifest)
    p4.p3.validate_p3(derived, splits, models, report, manifest)
    p4.pr.final_test(derived, splits, models, report, manifest, n_boot=50)
    p4.p3.final_test_p3(derived, splits, models, report, manifest, n_boot=50)
    sel = p4.validate_p4(derived, splits, models, report, manifest)
    res = p4.final_test_p4(derived, splits, models, report, manifest, n_boot=50)
    assert sel["model_id"] == p4.P4_ID and res["vs_P0_majority"]["difference_paired"]["point"] > 0
    out = models / "priority" / "p4"
    assert (
        joblib.load(out / "model.joblib").classes_.tolist() and (out / "featurizer.joblib").exists()
    )
    kept = json.loads(report.read_text(encoding="utf-8"))["test"]
    assert "primary" in kept and "p3" in kept and "p4" in kept  # P2/P3 untouched
    with pytest.raises(p4.ev.TestGuardError):
        p4.final_test_p4(derived, splits, models, report, manifest, n_boot=50)
