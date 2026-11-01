"""vscode monthly retrain (ADR-0072): window/split, challenger fit, same-holdout evaluation, gate.

Pure functions; the CLI is scripts/vscode_retrain.py. Reuses the ADR-0070 taxonomy and the
qp1_vscode_lib scoring pieces unchanged. Nothing here uploads, deploys or touches production.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # triage_iq (served model)
import qp0_ceiling_labels as cl  # noqa: E402
import qp1_conformal as qc  # noqa: E402
import qp1_eval as ev  # noqa: E402
import qp1_tfidf_lr as t1  # noqa: E402
import qp1_vscode_lib as vl  # noqa: E402
import qp2_vscode_taxonomy as tax  # noqa: E402

SEED = 42
SETTLE_DAYS, CALIB_DAYS, HOLDOUT_DAYS = 30, 60, 60  # label settling lag; calibration; holdout
WINDOW_MONTHS, FLOOR = 48, pd.Timestamp("2023-01-01", tz="UTC")  # ADR-0070: never shorter than 2023
MARGIN = 0.01  # pre-declared: challenger strict top-3 must beat champion by >= 1.0 point
NONINF = 0.02  # in-family top-3 may not be worse than champion by more than 2 points (CI lower)
MIN_HOLDOUT, MIN_TRAIN, MIN_CLASSES = 500, 5000, 20  # sanity floors; below them the CI is noise
N_BOOT = 2000
# ADR-0073 (GG D92.2 + D93): labels describing an issue's status or workflow rather than the code
# area it concerns, NEW relative to the ADR-0070 list. Must equal the NEW = yes rows of
# docs/architecture/adr/0073-vscode-label-decisions.md (a test checks it). Frozen: another
# exclusion needs a new ADR. Compared lower-case, as taxonomy.is_excluded does.
RETRAIN_META_EXCLUDE = frozenset({
    "2026-themes", "a11y-partner", "ack", "agent-bakeoff", "agent-host-mitigated", "ai-generated",
    "ai-triaged", "approved", "copilot-triage-needed", "iteration-plan-draft", "modernization",
    "needs-classifier-review", "needs-verification", "new release", "on-unit-test",
    "papercut 🩸", "pr-test-failure-triage", "recent-regression",
    "release-cherry-pick", "service-outage", "stable-anomaly", "test-disabled", "triage-miss",
    "tree-sitter-autoindent-candidate", "untested", "ux-feedback-wanted", "windows-verification",
    "❗ untested",
})  # fmt: skip


@dataclass(frozen=True)
class Windows:
    """Half-open [start, end) UTC intervals on created_at."""

    train_start: pd.Timestamp
    calib_start: pd.Timestamp
    holdout_start: pd.Timestamp
    cutoff: pd.Timestamp


def make_windows(as_of: pd.Timestamp, window_months: int = WINDOW_MONTHS) -> Windows:
    """cutoff = as_of - 30 d (labels settled); holdout = last 60 d, calibration the 60 d before."""
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    cutoff = as_of.normalize() - pd.Timedelta(days=SETTLE_DAYS)
    return Windows(
        train_start=max(FLOOR, cutoff - pd.DateOffset(months=window_months)),
        calib_start=cutoff - pd.Timedelta(days=HOLDOUT_DAYS + CALIB_DAYS),
        holdout_start=cutoff - pd.Timedelta(days=HOLDOUT_DAYS),
        cutoff=cutoff,
    )


def assign_split(created: pd.Series, w: Windows) -> pd.Series:
    """'train' | 'calibration' | 'holdout' by created_at; '' outside the window."""
    conds = [
        (created >= w.train_start) & (created < w.calib_start),
        (created >= w.calib_start) & (created < w.holdout_start),
        (created >= w.holdout_start) & (created < w.cutoff),
    ]
    return pd.Series(
        np.select(conds, ["train", "calibration", "holdout"], default=""), index=created.index
    )


def retrain_family(label: str) -> bool:
    """True for a label that may become a class: ADR-0070 `not_excluded` and not ADR-0073 meta."""
    return tax.not_excluded(label) and label.lower() not in RETRAIN_META_EXCLUDE


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(derived: Path, w: Windows) -> pd.DataFrame:
    """Labelled (>= 1 `retrain_family` label) settled issues in the window, with split and text.

    Calibration / holdout rows whose text_hash occurs in an earlier split are dropped (leakage).
    """
    cols = ["number", "created_at", "title", "body", "labels_final", "labels_t72h", "text_hash"]
    iss = pd.read_parquet(
        derived / "issues.parquet", columns=cols,
        filters=[("created_at", ">=", w.train_start), ("created_at", "<", w.cutoff)],
    )  # fmt: skip
    if iss.empty:
        raise ValueError("no issues in the retrain window")
    iss = iss.sort_values("number").reset_index(drop=True)
    events = pd.read_parquet(derived / "label_events.parquet")
    events = events[events["number"].isin(iss["number"])]
    d = cl.per_issue(iss, events, retrain_family)
    d = d.merge(iss[["number", "title", "body"]], on="number")
    d = d[d["final"].map(bool)].reset_index(drop=True)
    d["split"] = assign_split(d["created_at"], w)
    d = d[d["split"] != ""].reset_index(drop=True)
    d["text"] = d["title"].fillna("") + "\n" + d["body"].fillna("").str.slice(0, 4000)
    seen: set[str] = set()
    keep = np.ones(len(d), bool)
    for name in ("train", "calibration", "holdout"):  # chronological: later splits lose repeats
        idx = np.flatnonzero(d["split"].to_numpy() == name)
        hashes = d["text_hash"].iloc[idx].tolist()
        if name != "train":
            keep[idx] = [h not in seen for h in hashes]
        seen.update(hashes)
    return d[keep].reset_index(drop=True)


def regenerate_classes(train: pd.DataFrame) -> list[str]:
    """ADR-0070 rule on THIS window: parents carried by >= 30 train issues; `other` last."""
    counts = tax.window_counts(pd.DataFrame({"labels_final": train["final"]}), tax.parent_of)
    return [*tax.family(counts), vl.OTHER]


def _top3_in(p: np.ndarray, y: np.ndarray, other: int) -> np.ndarray:
    """Strict top-3 hit per issue (gold != other and in the 3 best classes)."""
    return (y != other) & (np.argsort(-p, axis=1, kind="stable")[:, :3] == y[:, None]).any(axis=1)


def train_challenger(
    tr: pd.DataFrame, ca: pd.DataFrame, classes: list[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Family-only TF-IDF+LR (ADR-0070 V2 recipe). Grid, tau and conformal use CALIBRATION only."""
    other = classes.index(vl.OTHER)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    xtr = vec.fit_transform(tr["text"])
    ytr, _ = vl.gold(tr, classes)
    yca = ev.to_idx(vl.gold(ca, classes)[0], classes)
    xca = vec.transform(ca["text"])
    keep = np.array([c != vl.OTHER for c in ytr])
    best: tuple[Any, ...] | None = None
    rows: list[dict[str, Any]] = []
    for c, cw in t1.GRID:
        m = t1.fit_lr(xtr[keep], [c2 for c2 in ytr if c2 != vl.OTHER], c, cw)
        top3 = float(_top3_in(t1.proba_full(m, xca, classes), yca, other).mean())
        rows.append({"C": c, "class_weight": cw, "calibration_strict_top3": top3})
        key = (-top3, c)  # higher top-3, then smaller C (as sel_key without the F1 tiebreak)
        if best is None or key < best[0]:
            best = (key, m, c, cw)
    assert best is not None
    model = best[1]
    pca = t1.proba_full(model, xca, classes)
    tau = vl.best_abstention_tau(pca.max(axis=1), yca == other)
    fam = yca != other
    sel = vl.conformal_select(pca[fam], yca[fam])
    spec = sel["selected"]
    u = qc.uniforms(int(fam.sum()), 0)
    qhat = qc.calibrate(pca[fam], yca[fam], spec, u, vl.CONF_ALPHA)
    bundle = {
        "model_kind": "tfidf_lr_family_only_v2", "vectorizer": vec, "model": model,
        "classes": classes, "tau": tau, "C": best[2], "class_weight": best[3],
        "conformal": {"spec": spec, "qhat": qhat, "alpha": vl.CONF_ALPHA, "population": "in-family"},
    }  # fmt: skip
    return bundle, rows


def bundle_proba(bundle: dict[str, Any], texts: pd.Series, classes: list[str]) -> np.ndarray:
    """Bundle probabilities mapped into `classes` (a bundle class outside it -> `other`, max)."""
    p = t1.proba_full(bundle["model"], bundle["vectorizer"].transform(texts), bundle["classes"])
    out = np.zeros((len(texts), len(classes)))
    for j, c in enumerate(bundle["classes"]):
        k = classes.index(vl.to_class(str(c), classes))
        out[:, k] = np.maximum(out[:, k], p[:, j])
    return out


class ChampionShaError(RuntimeError):
    """The champion file is not the one MANIFEST.sha256 pins; the evaluation refuses to run."""


def champion_file(champion: Path) -> Path:
    """The file whose bytes define the champion: the served .pkl in a models dir, else the bundle."""
    return champion / "component_classifier_microsoft_vscode.pkl" if champion.is_dir() else champion


# A manifest key a champion may be pinned to: the vscode classifier (served pickle or a retrain
# bundle) under data/models/. Narrow on purpose: a typo or another model's line must not pass.
CHAMPION_KEY_RE = re.compile(
    r"data/models/component_classifier_microsoft_vscode[A-Za-z0-9_.-]*\.(pkl|joblib)"
)


def verify_champion(
    champion: Path, manifest: Path, manifest_key: str | None = None
) -> tuple[str, str]:
    """Return (actual, expected) sha256; raise ChampionShaError unless they match (fail closed).

    The manifest key is `manifest_key` when given (a downloaded bundle can carry a file name, such
    as champion.joblib, that is not its manifest name), else data/models/<champion file name>. So
    a stale checkout copy, a wrong file or a bundle the manifest does not list all stop the run
    instead of moving the bar silently; an explicit key only changes WHICH line is compared.
    """
    f = champion_file(champion)
    key = manifest_key if manifest_key is not None else f"data/models/{f.name}"
    if manifest_key is not None and not CHAMPION_KEY_RE.fullmatch(manifest_key):
        raise ChampionShaError(f"champion manifest key {manifest_key!r} is not a vscode classifier")
    try:
        lines = Path(manifest).read_text(encoding="utf-8").splitlines()
        actual = file_sha256(f)
    except OSError as e:
        raise ChampionShaError(f"cannot read manifest or champion: {e}") from e
    parts = [ln.split(None, 1) for ln in lines]
    found = [p[0] for p in parts if len(p) == 2 and p[1].strip() == key]
    if len(found) != 1:
        raise ChampionShaError(
            f"expected exactly one {key} entry in {manifest}, found {len(found)}"
        )
    expected = found[0].lower()
    if actual != expected:
        raise ChampionShaError(f"champion sha256 {actual} != manifest {expected} for {key}")
    return actual, expected


def champion_proba(champion: Path, d: pd.DataFrame, classes: list[str]) -> tuple[np.ndarray, str]:
    """Champion scores on `d`: a models dir (served OVR classifier) or a retrain bundle .joblib."""
    if champion.is_dir():
        f = champion / "component_classifier_microsoft_vscode.pkl"
        return vl.production_proba(d, classes, champion), file_sha256(f)
    return bundle_proba(joblib.load(champion), d["text"], classes), file_sha256(champion)


def _ci(a: np.ndarray, n_boot: int) -> dict[str, Any]:
    return ev.bootstrap(lambda i: float(a[i].mean()), len(a), n_boot)


def _pd(a: np.ndarray, b: np.ndarray, n_boot: int) -> dict[str, Any]:
    return ev.paired_diff(
        lambda i: float(a[i].mean()), lambda i: float(b[i].mean()), len(a), n_boot
    )


def evaluate_holdout(
    ho: pd.DataFrame, classes: list[str], p_ch: np.ndarray, tau: float, p_champ: np.ndarray,
    p_naive: np.ndarray, n_boot: int = N_BOOT,
) -> dict[str, Any]:  # fmt: skip
    """Challenger, champion, naive on the SAME holdout rows; paired bootstrap of the differences."""
    other = classes.index(vl.OTHER)
    y_p, y_s = vl.gold(ho, classes)
    y, sm = ev.to_idx(y_p, classes), ev.set_mask(y_s, classes)
    views = {
        "challenger": vl.v2_views(p_ch, tau, classes)[0],
        "champion": vl.rank_proba(p_champ, other),
        "naive": vl.rank_proba(p_naive, other),
    }
    h = {k: vl.hit_arrays(y, sm, v, other) for k, v in views.items()}
    fam = h["challenger"]["gold_in_family"]
    out: dict[str, Any] = {"n_holdout": len(y), "n_in_family": int(fam.sum())}
    for k in views:
        out[k] = {
            "strict_top3": _ci(h[k]["strict_top3"], n_boot),
            "strict_top1": _ci(h[k]["strict_top1"], n_boot),
            "in_family_top3": _ci(h[k]["strict_top3"][fam], n_boot),
        }
    for base in ("champion", "naive"):
        out[f"paired_vs_{base}"] = {
            "strict_top3": _pd(h["challenger"]["strict_top3"], h[base]["strict_top3"], n_boot),
            "in_family_top3": _pd(
                h["challenger"]["strict_top3"][fam], h[base]["strict_top3"][fam], n_boot
            ),
        }
    return out


def conformal_holdout_report(
    ho: pd.DataFrame, classes: list[str], p_ch: np.ndarray, conformal: dict[str, Any]
) -> dict[str, Any]:
    """Saved calibration-only conformal spec + qhat applied to the holdout (no re-selection or
    re-calibration). REPORT-ONLY: the publish gate never reads this key (ADR-0072, ADR-0069)."""
    other = classes.index(vl.OTHER)
    y_p, y_s = vl.gold(ho, classes)
    y, sm = ev.to_idx(y_p, classes), ev.set_mask(y_s, classes)
    mask = qc.predict_mask(p_ch, conformal["qhat"], conformal["spec"], qc.uniforms(len(y), 1))
    fam = y != other
    return {
        "exploratory": "ADR-0069: unconfirmed until the post-test confirmation set",
        "alpha": conformal["alpha"], "spec": conformal["spec"], "qhat": float(conformal["qhat"]),
        "n": len(y), "n_in_family": int(fam.sum()),
        "all_labelled": qc.mask_report(mask, y, sm),
        "in_family_only": qc.mask_report(mask[fam], y[fam], sm[fam]),
    }  # fmt: skip


class GateInputError(ValueError):
    """A gate input is missing, non-numeric or non-finite; the gate refuses."""


def _num(d: dict[str, Any], *path: str) -> float:
    cur: Any = d
    try:
        for k in path:
            cur = cur[k]
        v = float(cur)
    except (KeyError, TypeError, ValueError) as e:
        raise GateInputError(f"missing or non-numeric input: {'.'.join(map(str, path))}") from e
    if not math.isfinite(v):
        raise GateInputError(f"non-finite input: {'.'.join(map(str, path))}")
    return v


def gate_decision(m: dict[str, Any] | None, n_train: int | None, n_classes: int | None) -> dict[str, Any]:  # fmt: skip
    """Accept only if EVERY check passes; any missing/ambiguous input -> refuse (fail closed)."""
    thresholds = {"margin": MARGIN, "noninferiority": NONINF, "min_holdout": MIN_HOLDOUT,
                  "min_train": MIN_TRAIN, "min_classes": MIN_CLASSES}  # fmt: skip
    checks: list[dict[str, Any]] = []
    try:
        if not m or n_train is None or n_classes is None:
            raise GateInputError("metrics, n_train or n_classes absent")
        diff = _num(m, "paired_vs_champion", "strict_top3", "point")
        lo = _num(m, "paired_vs_champion", "strict_top3", "ci95", 0)
        nlo = _num(m, "paired_vs_naive", "strict_top3", "ci95", 0)
        flo = _num(m, "paired_vs_champion", "in_family_top3", "ci95", 0)
        n_ho = _num(m, "n_holdout")
        checks = [
            ("holdout_size", n_ho >= MIN_HOLDOUT, f"{int(n_ho)} >= {MIN_HOLDOUT}"),
            ("train_size", n_train >= MIN_TRAIN, f"{n_train} >= {MIN_TRAIN}"),
            ("class_count", n_classes >= MIN_CLASSES, f"{n_classes} >= {MIN_CLASSES}"),
            ("beats_champion_margin", diff >= MARGIN, f"diff {diff:.4f} >= {MARGIN}"),
            ("beats_champion_ci", lo > 0, f"paired CI lower {lo:.4f} > 0"),
            ("beats_naive_ci", nlo > 0, f"paired CI lower vs naive {nlo:.4f} > 0"),
            ("in_family_noninferior", flo > -NONINF, f"in-family CI lower {flo:.4f} > -{NONINF}"),
        ]
        checks = [{"check": a, "passed": bool(b), "detail": c} for a, b, c in checks]
        accept = all(c["passed"] for c in checks)
        reasons = [f"{c['check']}: {c['detail']}" for c in checks if not c["passed"]]
    except GateInputError as e:
        accept, reasons = False, [f"input_error: {e}"]
    return {"decision": "accept" if accept else "refuse", "reasons": reasons,
            "checks": checks, "thresholds": thresholds}  # fmt: skip
