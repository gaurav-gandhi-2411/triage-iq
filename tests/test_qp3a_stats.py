"""Pure statistics helpers of scripts/qp3a_component_decision_eval.py.

The committed report is NOT re-derived here: that needs the manifest-matching data/models
classifiers (gitignored, not in CI), so CI must not depend on them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "qp3a_component_decision_eval.py"
_spec = importlib.util.spec_from_file_location("qp3a_eval", _PATH)
qp3a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qp3a)


def test_mcnemar_exact_known_values() -> None:
    assert qp3a.mcnemar_exact(0, 0) == 1.0
    assert qp3a.mcnemar_exact(5, 4) == pytest.approx(1.0)
    assert qp3a.mcnemar_exact(0, 4) == pytest.approx(0.125)  # 2 * 0.5**4
    assert qp3a.mcnemar_exact(9, 1) == pytest.approx(2 * 11 / 1024)


def test_wilson_matches_textbook_value_and_handles_extremes() -> None:
    lo, hi = qp3a.wilson(8, 10)
    assert (lo, hi) == pytest.approx((0.4902, 0.9433), abs=1e-3)
    assert qp3a.wilson(0, 0) == [0.0, 0.0]
    assert qp3a.wilson(10, 10)[1] == 1.0


def test_bootstrap_is_deterministic_and_brackets_the_point_estimate() -> None:
    off = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=float)
    on = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 0], dtype=float)
    ci = qp3a.paired_bootstrap_ci(off, on)
    assert ci == qp3a.paired_bootstrap_ci(off, on)
    assert ci[0] <= 0.3 <= ci[1]
