"""Conformal score families and the validation-only selection protocol (Phase 1.2, D82).

PRE-REGISTERED PROTOCOL (2026-10-20; the score family was WIDENED after test numbers had been
seen: the non-randomised APS of qp1_eval.py gave near-vacuous sets on test, and a verifier's peek
showed LAC / randomised APS are much smaller. The choice below is therefore made on VALIDATION
ONLY and the report states this). Scores, for a probability row p with classes ranked by p
(stable), rank r (1-based) and u ~ Uniform(0, 1) drawn from numpy default_rng, seed 42:
 (1) aps_nonrandomised: cumulative mass up to and including the class; top-1 always included;
 (2) aps_randomised: mass strictly above the class + u * p_class;
 (3) raps: (2) + lambda * max(0, r - k_reg), lambda in {0.001, 0.01, 0.1}, k_reg in {1, 2, 3};
 (4) lac: 1 - p_class.
A class is in the set iff its score <= qhat, the finite-sample conformal quantile of the
calibration scores of the PRIMARY label at alpha = 0.05. Selection: for each score, 50 seeded
random 50/50 splits of the eligible labelled VALIDATION set; calibrate on half A, measure
coverage (primary label) and set size on half B; average over the repetitions. Choose the score
with the smallest mean set size among those whose mean coverage >= 0.95; if none reaches 0.95
choose the highest coverage; ties -> smaller mean set size (then name). The chosen score is then
calibrated on the FULL validation set and applied to test only in the explicit final-test step
(final=True), next to the pre-registered non-randomised APS and fixed top-3 / top-4 sets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_eval as ev  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED PROTOCOL", 1)[1].strip()
REPS, SEED = 50, 42
RAPS_GRID = [(lam, k) for lam in (0.001, 0.01, 0.1) for k in (1, 2, 3)]


def specs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"name": "aps_nonrandomised"}, {"name": "aps_randomised"}]
    out += [{"name": f"raps_l{lam}_k{k}", "lam": lam, "k_reg": k} for lam, k in RAPS_GRID]
    return [*out, {"name": "lac"}]


def score_matrix(p: np.ndarray, u: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Score of EVERY class for every row (n x C); smaller = more conforming."""
    if spec["name"] == "lac":
        return 1.0 - p
    order = ev.ranked(p)
    sp = np.take_along_axis(p, order, axis=1)
    cum = np.cumsum(sp, axis=1)
    if spec["name"] == "aps_nonrandomised":
        s = cum
    else:
        s = cum - sp + u[:, None] * sp
        if spec["name"].startswith("raps"):
            s = s + spec["lam"] * np.maximum(0, np.arange(1, p.shape[1] + 1) - spec["k_reg"])
    out = np.empty_like(s)
    np.put_along_axis(out, order, s, axis=1)
    return out


def uniforms(n: int, salt: int, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed + salt).random(n)


def predict_mask(p: np.ndarray, qhat: float, spec: dict[str, Any], u: np.ndarray) -> np.ndarray:
    mask = score_matrix(p, u, spec) <= qhat + 1e-12
    if spec["name"] == "aps_nonrandomised":
        mask[np.arange(len(p)), p.argmax(axis=1)] = True  # pre-registered: never empty
    return mask


def calibrate(
    p: np.ndarray, y: np.ndarray, spec: dict[str, Any], u: np.ndarray, alpha: float = ev.ALPHA
) -> float:
    return ev.conformal_qhat(score_matrix(p, u, spec)[np.arange(len(y)), y], alpha)


def mask_report(mask: np.ndarray, y: np.ndarray, sets: np.ndarray) -> dict[str, Any]:
    """Coverage (primary and any-valid, Wilson) and size summary of a set mask."""
    n, size = len(y), mask.sum(axis=1)
    k1, k2 = int(mask[np.arange(n), y].sum()), int((mask & sets).any(axis=1).sum())
    cov = lambda k: {"k": k, "n": n, "rate": k / n, "wilson95": list(ev.wilson(k, n))}  # noqa: E731
    return {
        "coverage_primary": cov(k1), "coverage_any_valid": cov(k2),
        "set_size_mean": float(size.mean()), "set_size_median": float(np.median(size)),
        "empty_set_rate": float((size == 0).mean()),
    }  # fmt: skip


def repeat_eval(
    p: np.ndarray, y: np.ndarray, spec: dict[str, Any], reps: int = REPS, seed: int = SEED
) -> dict:
    """Mean / sd over seeded 50/50 validation splits of coverage and set size."""
    cov, mean_sz, med_sz = [], [], []
    for r in range(reps):
        rng = np.random.default_rng(seed * 1000 + r)
        perm = rng.permutation(len(y))
        a, b = perm[: len(y) // 2], perm[len(y) // 2 :]
        qhat = calibrate(p[a], y[a], spec, rng.random(len(a)))
        mask = predict_mask(p[b], qhat, spec, rng.random(len(b)))
        size = mask.sum(axis=1)
        cov.append(mask[np.arange(len(b)), y[b]].mean())
        mean_sz.append(size.mean())
        med_sz.append(np.median(size))
    f = lambda v: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}  # noqa: E731
    return {
        "spec": spec,
        "coverage": f(cov),
        "set_size_mean": f(mean_sz),
        "set_size_median": f(med_sz),
    }


def choose(table: list[dict[str, Any]], target: float = 1 - ev.ALPHA) -> dict[str, Any]:
    """Selection rule (see module docstring)."""
    ok = [r for r in table if r["coverage"]["mean"] >= target]
    if ok:
        return min(ok, key=lambda r: (r["set_size_mean"]["mean"], r["spec"]["name"]))
    return min(
        table, key=lambda r: (-r["coverage"]["mean"], r["set_size_mean"]["mean"], r["spec"]["name"])
    )


def select_score(
    proba_val: np.ndarray, y_val: np.ndarray, reps: int = REPS, seed: int = SEED
) -> dict:
    """Validation-only selection: takes validation arrays and nothing else."""
    table = [repeat_eval(proba_val, y_val, s, reps, seed) for s in specs()]
    return {"table": table, "selected": choose(table)["spec"]}


def final_sets(
    proba_val,
    y_val,
    proba_test,
    y_test,
    sets_test,
    selected,
    final: bool,  # noqa: ANN001
) -> dict[str, Any]:
    """Full-validation calibration applied to test (final=True only): selected, APS and top-k."""
    ev.guard("test", final)
    out: dict[str, Any] = {}
    for key, spec in (("selected", selected), ("pre_registered_aps_nonrandomised", specs()[0])):
        q = calibrate(proba_val, y_val, spec, uniforms(len(y_val), 0))
        out[key] = {"spec": spec, "qhat": q} | mask_report(
            predict_mask(proba_test, q, spec, uniforms(len(y_test), 1)), y_test, sets_test
        )
    order = ev.ranked(proba_test)
    for k in (3, 4):
        m = np.zeros(proba_test.shape, dtype=bool)
        np.put_along_axis(m, order[:, :k], True, axis=1)
        out[f"fixed_top{k}"] = mask_report(m, y_test, sets_test)
    return out
