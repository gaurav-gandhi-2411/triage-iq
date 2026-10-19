"""The committed vscode expansion results report must equal a fresh recomputation from the committed cassettes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(
        "vscode_expansion_results", ROOT / "scripts" / "vscode_expansion_results.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vscode_expansion_results_report_is_current() -> None:
    mod = _load()
    committed = json.loads(mod.OUT.read_text(encoding="utf-8"))
    assert mod.build() == committed


def test_pooled_arm_is_the_union_of_the_two_arms() -> None:
    r = json.loads(_load().OUT.read_text(encoding="utf-8"))
    assert r["pooled74"]["n"] == r["committed11"]["n"] + r["expansion63"]["n"] == 74
    weighted = (r["committed11"]["mean"] * 11 + r["expansion63"]["mean"] * 63) / 74
    assert abs(weighted - r["pooled74"]["mean"]) < 1e-3
