from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp1v", _S / "qp1_vscode_phase1.py")
vs = importlib.util.module_from_spec(_SPEC)
sys.modules["qp1v"] = vs
_SPEC.loader.exec_module(vs)


def _writer(tmp_path: Path, data: dict) -> object:
    manifest = tmp_path / "m.json"
    manifest.write_text("{}")
    (tmp_path / "r.json").write_text(json.dumps(data))
    return vs.ev.ResultsWriter(tmp_path / "r.json", manifest)


def test_alpha_0_1_results_move_under_superseded_key(tmp_path: Path) -> None:
    w = _writer(
        tmp_path,
        {"models": {}, "conformal_selection": {"alpha": 0.1}, "conformal_test": {"alpha": 0.1}},
    )
    vs.supersede_alpha_0_1(w)
    assert "conformal_test" not in w.data and "conformal_selection" not in w.data
    assert set(w.data["superseded_alpha_0_1"]) == {"conformal_selection", "conformal_test"}
    w.data["conformal_selection"] = {"alpha": 0.05}
    vs.supersede_alpha_0_1(w)  # alpha 0.05 results are never moved
    assert w.data["conformal_selection"]["alpha"] == 0.05


def test_conformal_test_runs_once_and_needs_frozen_adr(tmp_path: Path) -> None:
    a = vs.argparse.Namespace(report=tmp_path / "r.json")
    _writer(tmp_path, {"models": {}, "conformal_test": {"alpha": 0.05}})
    with pytest.raises(vs.ev.TestGuardError):  # second alpha-0.05 test evaluation refused
        vs.stage_conformal_test(a, tmp_path, tmp_path, tmp_path)
    _writer(
        tmp_path, {"models": {}, "conformal_selection": {"alpha": 0.1, "selected": {"name": "lac"}}}
    )
    with pytest.raises(RuntimeError, match="frozen in ADR-0071"):  # alpha mismatch -> no test read
        vs.stage_conformal_test(a, tmp_path, tmp_path, tmp_path)
