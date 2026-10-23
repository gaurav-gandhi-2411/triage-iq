"""Reusable pieces of the vscode Phase 1 runner (qp1_vscode_phase1.py): taxonomy gold mapping,
strict / lenient / in-family scoring, abstention threshold, V2 views, model selection key and the
conformal selection loop. Pure functions; the design is pre-registered in qp1_vscode_phase1.py.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp0_ceiling_labels as cl  # noqa: E402
import qp1_conformal as qc  # noqa: E402
import qp1_eval as ev  # noqa: E402
import qp2_vscode_taxonomy as tax  # noqa: E402

OTHER = "other"
FROZEN_SHA = "95129c95809c21574ed34d41a46358149322680a7ccb0e42e90f4b62e7b761ea"  # ADR-0070
IDS = {
    "v0": "majority_prior_v0",
    "v1": "tfidf_lr_v1",
    "v2": "tfidf_lr_family_only_v2",
    "prod": "production_vscode_ovr",
}
COMPLEXITY = {IDS["v0"]: 0, IDS["v1"]: 1, IDS["v2"]: 2}
CONF_ALPHA, CONF_TARGET, CONF_TOL = 0.1, 0.90, 0.02
CONF_SPECS = [
    {"name": "aps_nonrandomised"},
    {"name": "aps_randomised"},
    {"name": "raps_l0.1_k1", "lam": 0.1, "k_reg": 1},
    {"name": "lac"},
]

# ----------------------------------------------------------------------------- taxonomy and data


def classes_sha256(classes: list[str]) -> str:
    """sha256 of the newline-joined sorted class list (ADR-0070 decision 2; `other` excluded)."""
    return hashlib.sha256("\n".join(sorted(classes)).encode()).hexdigest()


def define_classes(train: pd.DataFrame) -> list[str]:
    """Parent-area classes carried by >= 30 TRAIN issues (labels_final), `other` appended last.

    Asserts the list equals ADR-0070's frozen one (by hash): the test run uses it as is.
    """
    fam = tax.family(
        tax.window_counts(pd.DataFrame({"labels_final": train["final"]}), tax.parent_of)
    )
    got = classes_sha256(fam)
    if got != FROZEN_SHA:
        raise AssertionError(f"class list sha256 {got} != ADR-0070 {FROZEN_SHA}")
    return [*sorted(fam), OTHER]


def to_class(label: str, classes: list[str]) -> str:
    """parent_of(label) if it is a modelled class, else `other`."""
    p = tax.parent_of(label)
    return p if p in classes and p != OTHER else OTHER


def gold(d: pd.DataFrame, classes: list[str]) -> tuple[list[str], list[frozenset[str]]]:
    """Primary-label class and the set of classes of all final labels, per issue."""
    prim = [to_class(p, classes) for p in d["primary"]]
    sets = [frozenset(to_class(x, classes) for x in s) for s in d["final"]]
    return prim, sets


def load_split(derived: Path, splits_dir: Path, name: str) -> pd.DataFrame:
    """All issues of ONE split with per_issue labels (family = non-excluded), text and flags."""
    sp = pd.read_parquet(splits_dir / f"{name}.parquet", columns=["number", "classifier_eligible"])
    cols = ["number", "created_at", "title", "body", "labels_final", "labels_t72h", "text_hash"]
    issues = pd.read_parquet(derived / "issues.parquet", columns=cols)
    issues = (
        issues[issues["number"].isin(sp["number"])].sort_values("number").reset_index(drop=True)
    )
    events = pd.read_parquet(derived / "label_events.parquet")
    events = events[events["number"].isin(issues["number"])]
    d = cl.per_issue(issues, events, tax.not_excluded).merge(sp, on="number")
    d = d.merge(issues[["number", "title", "body"]], on="number")
    d["text"] = d["title"].fillna("") + "\n" + d["body"].fillna("").str.slice(0, 4000)
    d["split"] = name
    return d.reset_index(drop=True)


def labelled_eligible(d: pd.DataFrame) -> pd.DataFrame:
    """Evaluation / training population: >= 1 non-excluded label and classifier_eligible."""
    keep = d["final"].map(bool).astype(bool) & d["classifier_eligible"].astype(bool)
    return d[keep].reset_index(drop=True)


# ------------------------------------------------------------------------------------- scoring


class Ranked:
    """Per-issue top-3 class indices for the strict and the lenient (k8s-convention) reading."""

    def __init__(self, strict: np.ndarray, lenient: np.ndarray) -> None:
        self.strict, self.lenient = strict[:, :3], lenient[:, :3]


def rank_proba(p: np.ndarray, other: int, abstain: np.ndarray | None = None) -> Ranked:
    """Rankings from probabilities; with `abstain`, the lenient list leads with `other`."""
    strict = ev.ranked(p)[:, :3]
    if abstain is None:
        return Ranked(strict, strict)
    lenient = np.where(
        abstain[:, None], np.column_stack([np.full(len(p), other), strict[:, :2]]), strict
    )
    return Ranked(strict, lenient)


def hit_arrays(y: np.ndarray, sm: np.ndarray, rk: Ranked, other: int) -> dict[str, np.ndarray]:
    """Per-issue booleans: strict / lenient top-1 and top-3, strict any-valid, abstain, in-family."""
    rows = np.arange(len(y))
    gold_in = y != other
    sm_in = sm.copy()
    sm_in[:, other] = False
    return {
        "strict_top1": gold_in & (rk.strict[:, 0] == y),
        "strict_top3": gold_in & (rk.strict == y[:, None]).any(axis=1),
        "lenient_top1": rk.lenient[:, 0] == y,
        "lenient_top3": (rk.lenient == y[:, None]).any(axis=1),
        "anyvalid_strict_top1": sm_in[rows, rk.strict[:, 0]],
        "anyvalid_strict_top3": sm_in[rows[:, None], rk.strict].any(axis=1),
        "abstain": rk.lenient[:, 0] == other,
        "gold_in_family": gold_in,
    }


def prop(k: int, n: int) -> dict[str, Any]:
    return {"k": int(k), "n": int(n), "rate": k / n if n else None, "wilson95": wilson_list(k, n)}


def wilson_list(k: int, n: int) -> list[float] | None:
    w = ev.wilson(int(k), int(n))
    return None if w is None else list(w)


def score(split, final, y_primary, y_sets, rk, classes, n_boot=ev.N_BOOT) -> dict[str, Any]:  # noqa: ANN001
    """All Phase 1 vscode metrics for one model on one split (test requires final=True)."""
    ev.guard(split, final)
    other = classes.index(OTHER)
    y, sm = ev.to_idx(y_primary, classes), ev.set_mask(y_sets, classes)
    h = hit_arrays(y, sm, rk, other)
    n, fam = len(y), h["gold_in_family"]
    out: dict[str, Any] = {"split": split, "n": n, "n_in_family": int(fam.sum())}
    for k in ("strict_top1", "strict_top3", "lenient_top1", "lenient_top3"):
        out[k] = ev.bootstrap(lambda i, a=h[k]: float(a[i].mean()), n, n_boot)
        out[k]["wilson95"] = wilson_list(h[k].sum(), n)
    for k in ("anyvalid_strict_top1", "anyvalid_strict_top3"):
        out[k] = ev.bootstrap(lambda i, a=h[k]: float(a[i].mean()), n, n_boot)
    for k in ("strict_top1", "strict_top3"):  # accuracy among issues the family can express
        a = h[k][fam]
        out[f"in_family_{k.split('_')[1]}"] = ev.bootstrap(
            lambda i, a=a: float(a[i].mean()), len(a), n_boot
        )
        out[f"in_family_{k.split('_')[1]}"]["wilson95"] = wilson_list(a.sum(), len(a))
    pred = rk.lenient[:, 0]
    out["macro_f1"] = ev.bootstrap(lambda i: ev.macro_f1(y[i], pred[i], len(classes)), n, n_boot)
    ab, g_oth = h["abstain"], ~fam
    out["abstention"] = {
        "rate": prop(ab.sum(), n),
        "precision": prop((ab & g_oth).sum(), ab.sum()),
        "recall": prop((ab & g_oth).sum(), g_oth.sum()),
    }
    out["coverage_bound_strict"] = prop(fam.sum(), n)
    return out


def paired(h_a: dict[str, np.ndarray], h_b: dict[str, np.ndarray], n_boot: int) -> dict[str, Any]:
    """Paired bootstrap CI of (A - B) for strict top-1 / top-3, in-family top-3, lenient top-3."""
    n, fam = len(h_a["strict_top1"]), h_a["gold_in_family"]
    out = {}
    for k in ("strict_top1", "strict_top3", "lenient_top3"):
        out[k] = ev.paired_diff(
            lambda i, a=h_a[k]: float(a[i].mean()),
            lambda i, b=h_b[k]: float(b[i].mean()),
            n,
            n_boot,
        )
    ia = h_a["strict_top3"][fam].astype(float)
    ib = h_b["strict_top3"][fam].astype(float)
    out["in_family_top3"] = ev.paired_diff(
        lambda i: float(ia[i].mean()), lambda i: float(ib[i].mean()), len(ia), n_boot
    )
    return out


def majority_proba(y_train: list[str], classes: list[str], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Train class frequencies as a probability row broadcast to n issues."""
    cnt = np.array([y_train.count(c) for c in classes], dtype=float)
    prior = cnt / cnt.sum()
    return np.tile(prior, (n, 1)), prior


def best_abstention_tau(pmax: np.ndarray, is_other: np.ndarray) -> float:
    """tau (validation max-prob quantile grid) maximising abstention F1; ties -> higher precision."""
    best = (-1.0, -1.0, 0.0)
    for tau in np.unique(np.quantile(pmax, np.linspace(0.0, 1.0, 201))):
        ab = pmax < tau
        tp = float((ab & is_other).sum())
        prec = tp / ab.sum() if ab.sum() else 0.0
        rec = tp / max(is_other.sum(), 1)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if (f1, prec) > best[:2]:
            best = (f1, prec, float(tau))
    return best[2]


def v2_views(p: np.ndarray, tau: float, classes: list[str]) -> tuple[Ranked, np.ndarray]:
    """Ranked + abstain flags for the family-only model."""
    ab = p.max(axis=1) < tau
    return rank_proba(p, classes.index(OTHER), ab), ab


def sel_key(sc: dict[str, Any], model_id: str) -> tuple[float, float, int]:
    """Lower is better: validation strict top-3, macro-F1, then the simpler model."""
    return (-sc["strict_top3"]["point"], -sc["macro_f1"]["point"], COMPLEXITY[model_id])


# ---------------------------------------------------------------------------------- conformal


def conformal_select(
    p: np.ndarray,
    y: np.ndarray,
    reps: int = qc.REPS,
    seed: int = qc.SEED,
    alpha: float = CONF_ALPHA,
) -> dict[str, Any]:
    """Validation-only conformal score selection at `alpha`; target coverage 1 - alpha."""
    table = []
    for spec in CONF_SPECS:
        cov, size = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed * 1000 + r)
            perm = rng.permutation(len(y))
            a, b = perm[: len(y) // 2], perm[len(y) // 2 :]
            q = qc.calibrate(p[a], y[a], spec, rng.random(len(a)), alpha)
            mask = qc.predict_mask(p[b], q, spec, rng.random(len(b)))
            cov.append(mask[np.arange(len(b)), y[b]].mean())
            size.append(mask.sum(axis=1).mean())
        table.append({
            "spec": spec,
            "coverage": {"mean": float(np.mean(cov)), "sd": float(np.std(cov, ddof=1))},
            "set_size_mean": {"mean": float(np.mean(size)), "sd": float(np.std(size, ddof=1))},
        })  # fmt: skip
    return {"table": table, "selected": conformal_choose(table, 1 - alpha)["spec"], "alpha": alpha}


def conformal_choose(table: list[dict[str, Any]], target: float = CONF_TARGET) -> dict[str, Any]:
    """Coverage within target +/- 0.02, then smallest mean size, then name; else closest coverage."""
    ok = [r for r in table if abs(r["coverage"]["mean"] - target) <= CONF_TOL]
    if ok:
        return min(ok, key=lambda r: (r["set_size_mean"]["mean"], r["spec"]["name"]))
    return min(table, key=lambda r: (abs(r["coverage"]["mean"] - target), r["spec"]["name"]))
