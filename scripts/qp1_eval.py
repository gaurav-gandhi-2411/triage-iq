"""Shared Phase 1 evaluation harness (spec section 6) for the kubernetes sig/* task (ADR-0067).

PRE-REGISTERED DESIGN (2026-10-20; do not change after seeing test results). Pure functions on
(y_true_primary, y_true_sets, proba, classes); labels are strings already mapped to the modelled
classes + `other`.
 * top-1 / top-3 accuracy against the PRIMARY label; any-valid top-1 (prediction in the true
   label set) and any-valid top-3 (any of the top-3 in the set); macro-F1 over the classes present
   in y_true; NLL (probabilities clipped at 1e-12); ECE with 15 equal-width bins on the top-1
   confidence. Ties in probability are broken by class order (stable sort).
 * Conformal sets by APS, non-randomised: a label's score is the cumulative probability mass of
   all classes ranked at or above it; calibration scores use the PRIMARY label on VALIDATION;
   qhat = the ceil((n + 1)(1 - alpha))-th smallest score (inf when that exceeds n), alpha = 0.05;
   a class is in the set iff its score <= qhat, and the top-1 class is always included (non-empty
   sets; conservative). Reported: coverage w.r.t. primary, coverage w.r.t. any-valid (set meets
   the true set), mean / median set size.
 * Temperature scaling: T fitted on VALIDATION by minimising NLL of softmax(log p / T), T in
   [0.05, 20]; ECE reported before and after.
 * Bootstrap 95% percentile CIs over issues: 2000 resamples, seed 42, paired differences use the
   same resampled indices; Wilson 95% for proportions.
 * Guard: test metrics are computed only with final=True, and ResultsWriter refuses a second test
   evaluation of the same model id in the same report file. Each entry records the model id, the
   sha256 of the prediction file and the sha256 of the frozen split manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qp0_fixlink_audit import wilson  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
N_BOOT, SEED, ECE_BINS, ALPHA = 2000, 42, 15, 0.05
CLIP = 1e-12


class TestGuardError(RuntimeError):
    """Raised when test metrics are requested without final=True or evaluated twice."""

    __test__ = False  # not a pytest class despite the name


def guard(split: str, final: bool) -> None:
    if split == "test" and not final:
        raise TestGuardError("test metrics require final=True (explicit --final-test)")


def to_idx(labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    pos = {c: i for i, c in enumerate(classes)}
    return np.array([pos[x] for x in labels], dtype=int)


def set_mask(y_sets: Sequence[frozenset[str] | set[str]], classes: Sequence[str]) -> np.ndarray:
    pos = {c: i for i, c in enumerate(classes)}
    m = np.zeros((len(y_sets), len(classes)), dtype=bool)
    for i, s in enumerate(y_sets):
        for c in s:
            m[i, pos[c]] = True
    return m


def ranked(proba: np.ndarray) -> np.ndarray:
    return np.argsort(-proba, axis=1, kind="stable")


def hit_arrays(y_primary, y_sets, proba, classes) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Per-issue boolean hits: top1, top3, anyvalid_top1, anyvalid_top3."""
    y, order, sm = to_idx(y_primary, classes), ranked(proba), set_mask(y_sets, classes)
    rows = np.arange(len(y))
    return {
        "top1": order[:, 0] == y,
        "top3": (order[:, :3] == y[:, None]).any(axis=1),
        "anyvalid_top1": sm[rows, order[:, 0]],
        "anyvalid_top3": sm[rows[:, None], order[:, :3]].any(axis=1),
    }


def f1_per_class(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """F1 per class (nan for classes absent from y)."""
    tp = np.bincount(y[pred == y], minlength=n_classes).astype(float)
    sup, npred = np.bincount(y, minlength=n_classes), np.bincount(pred, minlength=n_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(sup + npred > 0, 2 * tp / (sup + npred), 0.0)
    return np.where(sup > 0, f1, np.nan)


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    """Mean F1 over the classes present in y_true."""
    return float(np.nanmean(f1_per_class(y, pred, n_classes)))


def nll(proba: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(proba[np.arange(len(y)), y], CLIP, None)).mean())


def ece(proba: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error on top-1 confidence, equal-width bins."""
    conf, correct = proba.max(axis=1), proba.argmax(axis=1) == y
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(
        (b == k).sum() / len(y) * abs(correct[b == k].mean() - conf[b == k].mean())
        for k in range(bins) if (b == k).any()
    ))  # fmt: skip


def aps_scores(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative mass of all classes ranked at or above the true class."""
    order = ranked(proba)
    cum = np.cumsum(np.take_along_axis(proba, order, axis=1), axis=1)
    pos = np.argmax(order == y[:, None], axis=1)
    return cum[np.arange(len(y)), pos]


def conformal_qhat(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return math.inf if k > n else float(np.sort(scores)[k - 1])


def aps_sets(proba: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean n x C prediction-set mask (score <= qhat; top-1 always included)."""
    order = ranked(proba)
    cum = np.cumsum(np.take_along_axis(proba, order, axis=1), axis=1)
    keep = cum <= qhat + 1e-12
    keep[:, 0] = True
    mask = np.zeros_like(keep)
    np.put_along_axis(mask, order, keep, axis=1)
    return mask


def conformal_report(proba, y_primary, y_sets, classes, qhat) -> dict[str, Any]:  # noqa: ANN001
    y, sm = to_idx(y_primary, classes), set_mask(y_sets, classes)
    ps = aps_sets(proba, qhat)
    size = ps.sum(axis=1)
    cov_p, cov_a = ps[np.arange(len(y)), y], (ps & sm).any(axis=1)
    k1, k2 = int(cov_p.sum()), int(cov_a.sum())
    return {
        "qhat": qhat, "n": len(y),
        "coverage_primary": {"k": k1, "n": len(y), "rate": k1 / len(y), "wilson95": list(wilson(k1, len(y)))},
        "coverage_any_valid": {"k": k2, "n": len(y), "rate": k2 / len(y), "wilson95": list(wilson(k2, len(y)))},
        "set_size_mean": float(size.mean()), "set_size_median": float(np.median(size)),
    }  # fmt: skip


def apply_temperature(proba: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(proba, CLIP, None)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(proba_val: np.ndarray, y: np.ndarray) -> float:
    """T minimising validation NLL (bounded scalar search, deterministic)."""
    res = minimize_scalar(
        lambda t: nll(apply_temperature(proba_val, t), y), bounds=(0.05, 20.0), method="bounded"
    )
    return float(res.x)


def bootstrap(stat: Callable[[np.ndarray], float], n: int, n_boot: int = N_BOOT, seed: int = SEED):
    """Point estimate and 95% percentile CI of stat(index array) over issue resamples."""
    rng = np.random.default_rng(seed)
    draws = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": float(stat(np.arange(n))), "ci95": [float(lo), float(hi)], "n": n}


def paired_diff(stat_a, stat_b, n: int, n_boot: int = N_BOOT, seed: int = SEED):  # noqa: ANN001
    """CI of stat_a - stat_b on the SAME resampled indices."""
    return bootstrap(lambda i: stat_a(i) - stat_b(i), n, n_boot, seed)


def evaluate(split, final, y_primary, y_sets, proba, classes, n_boot: int = N_BOOT) -> dict:  # noqa: ANN001
    """All point metrics with bootstrap CIs. Test requires final=True."""
    guard(split, final)
    y, h, n = (
        to_idx(y_primary, classes),
        hit_arrays(y_primary, y_sets, proba, classes),
        len(y_primary),
    )
    pred = proba.argmax(axis=1)
    out = {k: bootstrap(lambda i, a=v: float(a[i].mean()), n, n_boot) for k, v in h.items()}
    out["macro_f1"] = bootstrap(lambda i: macro_f1(y[i], pred[i], len(classes)), n, n_boot)
    out["nll"] = bootstrap(lambda i: nll(proba[i], y[i]), n, n_boot)
    out["ece"] = bootstrap(lambda i: ece(proba[i], y[i]), n, n_boot)
    out["top1_wilson95"] = list(wilson(int(h["top1"].sum()), n))
    out["n"], out["split"] = n, split
    return out


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ResultsWriter:
    """JSON report with per-model entries; refuses a second test evaluation of one model id."""

    def __init__(self, report_path: Path, manifest_path: Path) -> None:
        self.path, self.manifest = Path(report_path), Path(manifest_path)
        self.data: dict[str, Any] = {"models": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.data["split_manifest_sha256"] = file_sha256(self.manifest)

    def add(
        self, model_id: str, split: str, metrics: dict, pred_file: Path, final: bool = False
    ) -> None:
        guard(split, final)
        entry = self.data["models"].setdefault(model_id, {})
        if split == "test" and "test" in entry:
            raise TestGuardError(f"test already evaluated for {model_id} in {self.path.name}")
        entry[split] = {
            **metrics,
            "pred_file": Path(pred_file).name,
            "pred_sha256": file_sha256(pred_file),
        }
        entry["split_manifest_sha256"] = self.data["split_manifest_sha256"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=1, sort_keys=True, default=str), encoding="utf-8"
        )
