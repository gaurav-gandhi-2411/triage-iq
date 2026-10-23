from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp1_k8s_strict_rescore as rs  # noqa: E402

# classes a, b, c, other(3); 5 issues
P = np.array(
    [
        [0.7, 0.2, 0.05, 0.05],  # gold a, hit top1
        [0.1, 0.2, 0.3, 0.4],  # gold b: top3 = other,c,b -> top3 hit only
        [0.1, 0.1, 0.1, 0.7],  # gold other, predicted other: lenient hit, strict miss
        [0.05, 0.05, 0.1, 0.8],  # gold other
        [0.5, 0.3, 0.1, 0.1],  # gold c, miss (c ranked 3rd -> top3 hit)
    ]
)
Y = np.array([0, 1, 3, 3, 2])


def test_strict_vs_lenient_hits() -> None:
    h = rs.hits(Y, P, 3)
    assert h["lenient_top1"].tolist() == [True, False, True, True, False]
    assert h["strict_top1"].tolist() == [True, False, False, False, False]
    assert h["lenient_top3"].tolist() == [True, True, True, True, True]
    assert h["strict_top3"].tolist() == [True, True, False, False, True]


def test_rescore_summary_numbers() -> None:
    r = rs.rescore(Y, P, 3, n_boot=50)
    assert r["n"] == 5 and r["n_gold_other"] == 2
    assert r["gold_other_share"]["rate"] == 0.4 and r["strict_coverage_bound"] == 0.6
    assert r["strict_top3"]["k"] == 3 and r["in_family_top3"]["k"] == 3
    assert r["in_family_top3"]["wilson95"][1] <= 1.0
