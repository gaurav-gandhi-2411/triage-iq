"""R10: vscode resolution time as a survival problem WITHOUT embeddings (CPU only).

PRE-REGISTERED 2026-10-29 in docs/eval/r10_vscode_survival_prereg_2026-10.md (committed before any
validation number existed; its sha256 is stored in the report's freeze record). Summary: four
families (penalised Cox, log-normal / Weibull AFT, gradient-boosted Cox, discrete-time hazard
LightGBM) on creation-time non-embedding features, frozen temporal split of scripts/qp0_make_splits.py;
selection on validation (within family: Harrell C; across families: IPCW mean absolute error);
ONE champion is scored once on test against the SERVED constant (3.8374 d, ADR-0064) and the
naive train-closed median. Criteria C1 (Harrell C CI > 0.5), C2 (median-AE paired gain CI > 0),
C3 (IPCW-MAE paired gain CI > 0), both baselines. Seed 42. Nothing in production changes.

Usage (repo root, throwaway venv with scikit-survival + lifelines):
    python scripts/r10_vscode_survival.py                # validation + freeze record
    python scripts/r10_vscode_survival.py --final-test   # test, once (refuses a second read)
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_eval as ev  # noqa: E402
import qp4_features as feat  # noqa: E402
import qp4_resolution as q4  # noqa: E402

ROOT = Path(r"D:\triageiq-data")
SLUG = "microsoft__vscode"
REPO_ROOT = Path(__file__).resolve().parent.parent
PREREG = REPO_ROOT / "docs" / "eval" / "r10_vscode_survival_prereg_2026-10.md"
MANIFEST = REPO_ROOT / "reports" / f"qp0_split_manifest_{SLUG}_2026-10.json"
REPORT = REPO_ROOT / "reports" / "r10_vscode_survival_2026-10.json"
SERVED_TRAIN = REPO_ROOT / "data" / "processed" / "microsoft_vscode_temporal_train.parquet"
SEED, N_BOOT, N_BOOT_C, TAU_H, CAP_H, FLOOR_H = 42, ev.N_BOOT, 500, 365 * 24.0, 730 * 24.0, 0.01
GRID_H = np.unique(np.concatenate([np.geomspace(0.05, CAP_H, 300), q4.EDGES[1 : q4.NB]]))
HORIZONS_D = {"1d": 1, "7d": 3, "30d": 5, "90d": 6}  # index into the 9 finite bin edges
ASSOC = {"NONE": 0, "CONTRIBUTOR": 1, "COLLABORATOR": 2, "MEMBER": 3}
FAMILIES = ("F1_coxnet", "F2_aft", "F3_gb_cox", "F4_hazard_lgbm")
GRIDS: dict[str, list[dict[str, Any]]] = {
    "F1_coxnet": [{"alpha": a, "l1_ratio": r} for a in (0.001, 0.01, 0.1) for r in (0.1, 0.5)],
    "F2_aft": [{"dist": d, "penalizer": p} for d in ("lognormal", "weibull") for p in (0.01, 0.1)],
    "F3_gb_cox": [{"n_estimators": n, "max_depth": m} for n in (100, 200) for m in (2, 3)],
    "F4_hazard_lgbm": [
        {"num_leaves": nl, "learning_rate": 0.05, "n_estimators": ne, "min_child_samples": 50}
        for nl in (15, 31)
        for ne in (200, 400)
    ],
}
LOG_COLS = ["title_len", "body_len", "body_lines", "n_urls", "n_images", "a_n_prior", "a_n_done",
            "a_med_res_h", "load_all", "arr7", "clo7", "rec_med_res_h", "rec_n", "n_lab_t0"]  # fmt: skip
META = ["number", "split", "eligible", "leak", "created_at", "T", "event", "follow_h"]
TRACE_RE = re.compile(r"Traceback|Exception|Error:|\bat [\w.$<>]+\(.*\)|\.(?:ts|js):\d+")
REPRO_RE = re.compile(r"steps to reproduce|repro", re.I)
VERSION_RE = re.compile(r"version\s*:", re.I)


class FinalTestError(RuntimeError):
    """Raised when the final test stage is requested a second time."""


# ---- data and features ----------------------------------------------------------------------
def make_targets(created: pd.Series, closed: pd.Series, res_h: pd.Series, cutoff: pd.Timestamp):
    """(T hours, event, follow-up hours): closed -> resolution (floored); open -> censored at cutoff."""
    follow = (cutoff - created).dt.total_seconds() / 3600
    event = closed.notna().to_numpy() & res_h.notna().to_numpy()
    t = np.where(event, np.maximum(res_h.fillna(0).to_numpy(), FLOOR_H), follow.to_numpy())
    return t, event, follow.to_numpy()


def text_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Own-text and calendar features known at creation."""
    body, title = df["body"].fillna(""), df["title"].fillna("")
    cr = df["created_at"].dt
    ang = lambda v, p: 2 * np.pi * v.to_numpy() / p  # noqa: E731
    return pd.DataFrame({
        "assoc": df["author_assoc"].map(ASSOC).fillna(-1).to_numpy(),
        "title_len": title.str.len().to_numpy(), "body_len": body.str.len().to_numpy(),
        "body_lines": body.str.count("\n").to_numpy(),
        "has_code": body.str.contains("```", regex=False).astype(int).to_numpy(),
        "has_trace": body.map(lambda s: int(bool(TRACE_RE.search(s)))).to_numpy(),
        "n_urls": body.map(lambda s: len(feat.URL_RE.findall(s))).to_numpy(),
        "n_images": body.str.count(r"!\[").to_numpy(),
        "has_repro": body.map(lambda s: int(bool(REPRO_RE.search(s)))).to_numpy(),
        "has_version": body.map(lambda s: int(bool(VERSION_RE.search(s)))).to_numpy(),
        "is_insiders": body.str.contains("insiders", case=False, regex=False).astype(int).to_numpy(),
        "title_q": title.str.contains("?", regex=False).astype(int).to_numpy(),
        "hour_sin": np.sin(ang(cr.hour, 24)), "hour_cos": np.cos(ang(cr.hour, 24)),
        "dow_sin": np.sin(ang(cr.dayofweek, 7)), "dow_cos": np.cos(ang(cr.dayofweek, 7)),
        "month_sin": np.sin(ang(cr.month, 12)), "month_cos": np.cos(ang(cr.month, 12)),
    }, index=df.index)  # fmt: skip


def load_features(created: np.ndarray, closed: np.ndarray, res_h: np.ndarray) -> pd.DataFrame:
    """Repo load and recent speed at each t_q, from issues strictly earlier / already closed.

    created / closed are datetime64[ns] (closed NaT = open). Closures count only if closed < t_q.
    """
    day = np.timedelta64(1, "D")
    cr_s = np.sort(created)
    ok = ~pd.isna(closed)
    order = np.argsort(closed[ok])
    cl_s, res_s = closed[ok][order], res_h[ok][order]
    before = np.searchsorted(cr_s, created, "left")
    closed_before = np.searchsorted(cl_s, created, "left")
    lo90 = np.searchsorted(cl_s, created - 90 * day, "left")
    med, cnt = np.full(len(created), np.nan), closed_before - lo90
    for i in np.flatnonzero(cnt > 0):
        med[i] = np.median(res_s[lo90[i] : closed_before[i]])
    return pd.DataFrame({
        "load_all": before - closed_before,
        "arr7": before - np.searchsorted(cr_s, created - 7 * day, "left"),
        "clo7": closed_before - np.searchsorted(cl_s, created - 7 * day, "left"),
        "rec_med_res_h": med, "rec_n": cnt,
    })  # fmt: skip


def creation_labels(
    events: pd.DataFrame, created: pd.Series, numbers: pd.Series, top_pool: np.ndarray, k: int = 20
) -> tuple[np.ndarray, pd.DataFrame]:
    """Labels applied within 60 s of creation: count per issue and 0/1 flags for the k most
    frequent among the rows flagged True in top_pool (train eligible). events: number, ts, action, label.
    """
    ev_l = events[events["action"] == "labeled"].merge(
        pd.DataFrame({"number": numbers, "c": created, "pool": top_pool}), on="number"
    )
    ev_l = ev_l[(ev_l["ts"] - ev_l["c"]) <= pd.Timedelta(seconds=60)].drop_duplicates(
        ["number", "label"]
    )
    top = ev_l[ev_l["pool"]]["label"].value_counts().index[:k]
    n_lab = numbers.map(ev_l.groupby("number").size()).fillna(0).to_numpy()
    flags = pd.crosstab(ev_l["number"], ev_l["label"]).reindex(columns=top, fill_value=0)
    flags = (flags.reindex(numbers.to_numpy(), fill_value=0) > 0).astype(int)
    flags.columns = [f"lab_{c}" for c in flags.columns]
    return n_lab, flags.reset_index(drop=True)


def build_table(root: Path, slug: str) -> pd.DataFrame:
    """One row per split issue: meta (split, flags, T, event) and raw creation-time features."""
    sp = pd.concat([
        pd.read_parquet(root / "splits" / slug / f"{s}.parquet",
                        columns=["number", "split", "classifier_eligible", "leak_texthash", "leak_dup_pair"])  # fmt: skip
        for s in ("train", "validation", "test")
    ])  # fmt: skip
    iss = pd.read_parquet(root / "derived_v2" / slug / "issues.parquet")
    for c in ("created_at", "closed_at"):
        iss[c] = iss[c].dt.tz_localize(None)
    iss = iss.merge(sp, on="number").sort_values("created_at").reset_index(drop=True)
    cutoff = iss["scraped_at_max"].iloc[0].tz_localize(None)
    t, event, follow = make_targets(iss["created_at"], iss["closed_at"], iss["resolution_hours"], cutoff)  # fmt: skip
    meta = pd.DataFrame({
        "number": iss["number"], "split": iss["split"], "eligible": iss["classifier_eligible"],
        "leak": iss["leak_texthash"] | iss["leak_dup_pair"], "created_at": iss["created_at"],
        "T": t, "event": event, "follow_h": follow,
    })  # fmt: skip
    le = pd.read_parquet(root / "derived_v2" / slug / "label_events.parquet")
    le["ts"] = le["ts"].dt.tz_localize(None)
    pool = ((iss["split"] == "train") & iss["classifier_eligible"]).to_numpy()
    n_lab, flags = creation_labels(le, iss["created_at"], iss["number"], pool)
    return pd.concat([
        meta, text_cols(iss), feat.author_history(iss),
        load_features(iss["created_at"].to_numpy(), iss["closed_at"].to_numpy(), iss["resolution_hours"].to_numpy()),  # fmt: skip
        pd.DataFrame({"n_lab_t0": n_lab}), flags,
    ], axis=1)  # fmt: skip


def fit_prep(train: pd.DataFrame) -> dict[str, Any]:
    """Train-only imputation, log1p, standardisation; drops zero-variance columns."""
    x = _raw(train)
    med = x.median()
    x = _prep_apply(x, med)
    sd = x.std(ddof=0)
    keep = sd.index[sd > 0].tolist()
    return {"med": med, "mean": x[keep].mean(), "std": sd[keep], "cols": keep}


def _raw(df: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=META).astype(float)
    x["a_med_missing"] = x["a_med_res_h"].isna().astype(float)
    x["rec_med_missing"] = x["rec_med_res_h"].isna().astype(float)
    return x


def _prep_apply(x: pd.DataFrame, med: pd.Series) -> pd.DataFrame:
    x = x.fillna(med)
    x[LOG_COLS] = np.log1p(x[LOG_COLS].clip(lower=0))
    return x


def apply_prep(df: pd.DataFrame, prep: dict[str, Any]) -> np.ndarray:
    """Design matrix (float64) of df under a train-fitted prep."""
    x = _prep_apply(_raw(df), prep["med"])[prep["cols"]]
    return ((x - prep["mean"]) / prep["std"]).to_numpy(dtype=np.float64)


# ---- survival objects -----------------------------------------------------------------------
def breslow_h0(lp: np.ndarray, t: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, float]:
    """Breslow cumulative baseline hazard on GRID_H for risk scores lp, and the shift c used."""
    c = float(lp.max())
    w = np.exp(lp - c)
    o = np.argsort(t, kind="stable")
    ts_all, suffix = t[o], np.cumsum(w[o][::-1])[::-1]
    ts, d = np.unique(t[e], return_counts=True)
    h = d / suffix[np.searchsorted(ts_all, ts, "left")]
    h0 = np.concatenate([[0.0], np.cumsum(h)])
    return h0[np.searchsorted(ts, GRID_H, "right")], c


def grid_quantile(s: np.ndarray, q: float) -> np.ndarray:
    """Hours at which P(T <= t) = q: log-time interpolation on GRID_H, CAP_H if never reached."""
    target = 1 - q
    below = s <= target
    k = below.argmax(1)
    k0, r = np.maximum(k - 1, 0), np.arange(len(s))
    s0, s1 = s[r, k0], s[r, k]
    frac = np.clip((s0 - target) / np.maximum(s0 - s1, 1e-12), 0, 1)
    out = np.exp(np.log(GRID_H[k0]) + frac * (np.log(GRID_H[k]) - np.log(GRID_H[k0])))
    return np.where(below.any(1), out, CAP_H)


def pred_from_grid(s: np.ndarray, risk: np.ndarray) -> dict[str, np.ndarray]:
    """Prediction dict from survival on GRID_H (n x G): point = median, 10% / 90% quantiles."""
    edge_cols = np.searchsorted(GRID_H, q4.EDGES[1 : q4.NB])
    return {"point": grid_quantile(s, 0.5), "q10": grid_quantile(s, 0.1),
            "q90": grid_quantile(s, 0.9), "risk": risk, "surv": s[:, edge_cols]}  # fmt: skip


def cox_pred(lp_fn: Callable, lp_tr: np.ndarray, t: np.ndarray, e: np.ndarray) -> Callable:
    """Predictor from a linear-predictor function: Breslow baseline from the training scores."""
    h0, c = breslow_h0(lp_tr, t, e)

    def predict(x: np.ndarray) -> dict[str, np.ndarray]:
        lp = np.asarray(lp_fn(x), dtype=float)
        return pred_from_grid(np.exp(-h0[None, :] * np.exp(lp - c)[:, None]), lp)

    return predict


def surv_y(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    return np.array(list(zip(e, t, strict=True)), dtype=[("e", bool), ("t", float)])


def fit_family(
    fam: str, hp: dict[str, Any], x: np.ndarray, t: np.ndarray, e: np.ndarray
) -> Callable:
    """Fit one family at one grid point on the training matrix; returns predict(x) -> dict."""
    warnings.filterwarnings("ignore")
    if fam == "F1_coxnet":
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        m = CoxnetSurvivalAnalysis(l1_ratio=hp["l1_ratio"], alphas=[hp["alpha"]], max_iter=100000)
        m.fit(x, surv_y(t, e))
        return cox_pred(m.predict, m.predict(x), t, e)
    if fam == "F3_gb_cox":
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        m = GradientBoostingSurvivalAnalysis(
            loss="coxph", n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            learning_rate=0.1, subsample=0.8, min_samples_leaf=50, random_state=SEED,
        )  # fmt: skip
        m.fit(x, surv_y(t, e))
        return cox_pred(m.predict, m.predict(x), t, e)
    if fam == "F2_aft":
        from lifelines import LogNormalAFTFitter, WeibullAFTFitter

        cls = LogNormalAFTFitter if hp["dist"] == "lognormal" else WeibullAFTFitter
        cols = [f"x{j}" for j in range(x.shape[1])]
        m = cls(penalizer=hp["penalizer"]).fit(
            pd.DataFrame(x, columns=cols).assign(T=t, E=e.astype(int)), "T", "E"
        )

        def predict(xn: np.ndarray) -> dict[str, np.ndarray]:
            d = pd.DataFrame(xn, columns=cols)
            s = m.predict_survival_function(d, times=GRID_H).T.to_numpy()
            return pred_from_grid(s, -np.log(m.predict_median(d).to_numpy()))

        return predict
    hz = q4.fit_m1(x.astype(np.float32), t, e, hp)

    def predict_f4(xn: np.ndarray) -> dict[str, np.ndarray]:
        s = q4.surv_from_hazard(hz(xn.astype(np.float32)))
        return {"point": q4.quantile_h(s, 0.5), "q10": q4.quantile_h(s, 0.1),
                "q90": q4.quantile_h(s, 0.9), "risk": -s[:, : q4.NB - 1].sum(1),
                "surv": s[:, : q4.NB - 1]}  # fmt: skip

    return predict_f4


def const_pred(
    point_h: float, n: int, aux: dict[str, Any], interval: bool
) -> dict[str, np.ndarray]:
    """Constant predictor: point, optional 10/90 interval, KM survival at the edges (for IBS)."""
    lo, hi = aux["q"] if interval else (np.nan, np.nan)
    return {"point": np.full(n, point_h), "q10": np.full(n, lo), "q90": np.full(n, hi),
            "risk": np.zeros(n), "surv": np.tile(q4.km_at(*aux["km"], q4.EDGES[1 : q4.NB]), (n, 1))}  # fmt: skip


# ---- metrics --------------------------------------------------------------------------------
def c_indexes(t: np.ndarray, e: np.ndarray, risk: np.ndarray, w: np.ndarray, tau: float = TAU_H):
    """(Harrell C, Uno C): pairs i event, T_i < T_j; ties 0.5; Uno weights w_i^2 for T_i < tau."""
    num_h = den_h = num_u = den_u = 0.0
    ev_idx = np.flatnonzero(e)
    for s in range(0, len(ev_idx), 512):
        i = ev_idx[s : s + 512]
        comp = t[i, None] < t[None, :]
        r = risk[i, None] - risk[None, :]
        conc = comp * ((r > 0) + 0.5 * (r == 0))
        wi = w[i] ** 2 * (t[i] < tau)
        num_h, den_h = num_h + conc.sum(), den_h + comp.sum()
        num_u, den_u = num_u + (wi * conc.sum(1)).sum(), den_u + (wi * comp.sum(1)).sum()
    return num_h / max(den_h, 1), num_u / max(den_u, 1e-12)


def ipcw_mae(err: np.ndarray, e: np.ndarray, w: np.ndarray) -> float:
    """Hajek IPCW mean absolute error: closed issues weighted 1/G(T_i); open issues weigh 0."""
    ew = e * w
    return float((ew * err).sum() / max(ew.sum(), 1e-12))


class Split:
    """Arrays of one evaluated split: T/E in hours, and split-specific IPCW weights."""

    def __init__(self, d: pd.DataFrame) -> None:
        self.t, self.e = d["T"].to_numpy(), d["event"].to_numpy().astype(bool)
        self.ci = np.flatnonzero(self.e)
        self.w = q4.ipcw_weights(self.t, self.e, self.t)  # G of THIS split (prereg)
        self.n = len(self.t)


def abs_err_days(pred: dict[str, np.ndarray], s: Split) -> np.ndarray:
    return np.abs(pred["point"] - s.t) / 24


def boot_pair(stat: Callable[[np.ndarray], tuple[float, float]], n: int, n_boot: int) -> list[dict]:
    """Like ev.bootstrap (same seed and index stream) for a statistic returning two values."""
    rng = np.random.default_rng(ev.SEED)
    draws = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    pt = stat(np.arange(n))
    return [
        {"point": float(pt[k]), "ci95": [float(v) for v in np.percentile(draws[:, k], [2.5, 97.5])], "n": n}
        for k in range(2)
    ]  # fmt: skip


def evaluate(pred: dict, s: Split, g_tr: tuple, w_tr: np.ndarray, qhat: float | None, n_boot_c: int = N_BOOT_C) -> dict:  # fmt: skip
    """Metrics with 95% bootstrap CIs for one predictor on one split."""
    err = abs_err_days(pred, s)
    ec, n, nc = err[s.ci], s.n, len(s.ci)
    out: dict[str, Any] = {"n": n, "n_closed": nc, "censored_share": float(1 - s.e.mean())}
    if pred["risk"].std() > 0:
        pool, uno = boot_pair(lambda i: c_indexes(s.t[i], s.e[i], pred["risk"][i], w_tr[i]), n, n_boot_c)  # fmt: skip
    else:
        pool = uno = {"point": 0.5, "ci95": [0.5, 0.5], "n": n}
    out |= {"harrell_c": pool, "uno_c_tau365d": uno}
    out["median_ae_days"] = ev.bootstrap(lambda i: float(np.median(ec[i])), nc, N_BOOT)
    out["mean_ae_days"] = ev.bootstrap(lambda i: float(ec[i].mean()), nc, N_BOOT)
    out["ipcw_mae_days"] = ev.bootstrap(lambda i: ipcw_mae(err[i], s.e[i], s.w[i]), n, N_BOOT)
    out["ibs_ipcw_train_G"] = ev.bootstrap(lambda i: q4.brier_ipcw(pred["surv"][i], s.t[i], s.e[i], g_tr), n, N_BOOT)  # fmt: skip
    out["ibs_ipcw_split_G"] = ev.bootstrap(lambda i: q4.brier_ipcw(pred["surv"][i], s.t[i], s.e[i], (s.t[i], s.e[i])), n, N_BOOT)  # fmt: skip
    if qhat is not None and not np.isnan(pred["q10"]).any():
        out["interval_EXPLORATORY"] = interval_report(pred, s, qhat)
    return out


def interval_report(pred: dict, s: Split, qhat: float) -> dict:
    """EXPLORATORY: CQR-adjusted 10/90 interval coverage and width on closed issues."""
    lg = lambda v: np.log1p(np.maximum(v, 0))  # noqa: E731
    lo, hi = lg(pred["q10"][s.ci]) - qhat, lg(pred["q90"][s.ci]) + qhat
    y = np.log1p(s.t[s.ci])
    cov, wid = (y >= lo) & (y <= hi), (np.expm1(hi) - np.expm1(np.maximum(lo, 0))) / 24
    nc = len(s.ci)
    return {"qhat_log": qhat, "target": 0.8,
            "coverage80": ev.bootstrap(lambda i: float(cov[i].mean()), nc, N_BOOT),
            "mean_width_days": ev.bootstrap(lambda i: float(wid[i].mean()), nc, N_BOOT),
            "median_width_days": float(np.median(wid))}  # fmt: skip


def paired_gain(base: dict, model: dict, s: Split) -> dict:
    """Baseline error minus model error (days; positive = model better), paired bootstrap."""
    eb, em = abs_err_days(base, s), abs_err_days(model, s)
    cb, cm, n, nc = eb[s.ci], em[s.ci], s.n, len(s.ci)
    return {
        "median_ae_gain_days": ev.bootstrap(
            lambda i: float(np.median(cb[i]) - np.median(cm[i])), nc, N_BOOT
        ),  # fmt: skip
        "mean_ae_gain_days": ev.bootstrap(
            lambda i: float(cb[i].mean() - cm[i].mean()), nc, N_BOOT
        ),  # fmt: skip
        "ipcw_mae_gain_days": ev.bootstrap(
            lambda i: ipcw_mae(eb[i], s.e[i], s.w[i]) - ipcw_mae(em[i], s.e[i], s.w[i]), n, N_BOOT
        ),  # fmt: skip
    }


def calibration(pred: dict, s: Split) -> dict:
    """Mean predicted P(T <= h) vs KM on the split at 1/7/30/90 d, and by quintile of F(7d)."""
    out: dict[str, Any] = {}
    for name, k in HORIZONS_D.items():
        h = q4.EDGES[1 + k]
        obs = ev.bootstrap(lambda i, h=h: float(1 - q4.km_at(*q4.km(s.t[i], s.e[i]), np.array([h]))[0]), s.n, N_BOOT)  # fmt: skip
        out[name] = {"pred_mean": float(1 - pred["surv"][:, k].mean()), "km_observed": obs}
    f7 = 1 - pred["surv"][:, HORIZONS_D["7d"]]
    q = np.unique(np.quantile(f7, np.linspace(0, 1, 6)))
    bins = np.clip(np.searchsorted(q, f7, "right") - 1, 0, len(q) - 2)
    h7 = q4.EDGES[1 + HORIZONS_D["7d"]]
    out["quintiles_F7d"] = [
        {"n": int((bins == b).sum()), "pred_mean": float(f7[bins == b].mean()),
         "km_observed": float(1 - q4.km_at(*q4.km(s.t[bins == b], s.e[bins == b]), np.array([h7]))[0])}  # fmt: skip
        for b in range(len(q) - 1) if (bins == b).sum() > 20
    ]  # fmt: skip
    return out


def criteria(gains: dict[str, dict], champ: dict) -> dict:
    """C1 (Harrell C CI > 0.5), C2 (median-AE gain CI > 0), C3 (IPCW-MAE gain CI > 0), per baseline."""
    c1 = champ["harrell_c"]["ci95"][0] > 0.5
    per = {b: {"C2": g["median_ae_gain_days"]["ci95"][0] > 0, "C3": g["ipcw_mae_gain_days"]["ci95"][0] > 0} for b, g in gains.items()}  # fmt: skip
    return {"C1": bool(c1), "per_baseline": per,
            "all_met": bool(c1 and all(v["C2"] and v["C3"] for v in per.values()))}  # fmt: skip


# ---- stages ---------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pop_rows(tab: pd.DataFrame, split: str, population: str, final: bool) -> pd.DataFrame:
    """Rows of one split; test rows only with final=True. population: eligible | nonleaky."""
    ev.guard(split, final)
    d = tab[tab["split"] == split]
    d = d[d["eligible"]] if population == "eligible" else d[~d["leak"]]
    return d.reset_index(drop=True)


def baselines(tr: pd.DataFrame, n: int) -> tuple[dict, dict]:
    """(auxes, predictors) for served constant / naive / KM on n rows; train-closed statistics."""
    t, e = tr["T"].to_numpy(), tr["event"].to_numpy().astype(bool)
    aux = q4.naive_aux("M0a", t, e)
    served_d = float(pd.read_parquet(SERVED_TRAIN, columns=["resolution_hours"])["resolution_hours"].median() / 24)  # fmt: skip
    assert abs(served_d - 3.8374) < 1e-3, f"served constant drifted: {served_d}"
    auxk = q4.naive_aux("M0b", t, e)
    preds = {"served_const_3.84d": const_pred(served_d * 24, n, aux, False),
             "naive_train_closed_median": const_pred(aux["median_h"], n, aux, True),
             "km_train_median": const_pred(auxk["median_h"], n, auxk, True)}  # fmt: skip
    return {"M0a": aux, "M0b": auxk, "served_d": served_d}, preds


def versions() -> dict[str, str]:
    import importlib.metadata as md

    return {p: md.version(p) for p in ("numpy", "pandas", "scikit-learn", "scikit-survival", "lifelines", "lightgbm")}  # fmt: skip


def run(stage: str, tab: pd.DataFrame, report: Path, out_dir: Path) -> dict:
    """Validation (stage 'validate') or the single test (stage 'final')."""
    final = stage == "final"
    saved = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    if final and ("test" in saved or saved.get("freeze", {}).get("test_read", True)):
        raise FinalTestError("test already read (or no freeze record) in " + report.name)
    tr = pop_rows(tab, "train", "eligible", False)
    prep = fit_prep(tr)
    xtr, ttr, etr = apply_prep(tr, prep), tr["T"].to_numpy(), tr["event"].to_numpy().astype(bool)
    g_tr = (ttr, etr)
    va = pop_rows(tab, "validation", "eligible", False)
    sva, xva = Split(va), apply_prep(va, prep)
    w_va_tr = q4.ipcw_weights(ttr, etr, sva.t)
    auxes, bva = baselines(tr, len(va))
    result: dict[str, Any] = saved if final else {}
    if not final:
        grid, sel, fitted = [], {}, {}
        for fam in FAMILIES:
            best = None
            for hp in GRIDS[fam]:
                t0 = time.time()
                try:
                    pr = fit_family(fam, hp, xtr, ttr, etr)(xva)
                except Exception as exc:  # noqa: BLE001 - a failed fit is a reported result
                    grid.append(
                        {"family": fam, **hp, "error": f"{type(exc).__name__}: {exc}"[:200]}
                    )
                    continue
                c = c_indexes(sva.t, sva.e, pr["risk"], w_va_tr)[0]
                med = float(np.median(abs_err_days(pr, sva)[sva.ci]))
                ipw = ipcw_mae(abs_err_days(pr, sva), sva.e, sva.w)
                grid.append({"family": fam, **hp, "val_harrell_c": c, "val_median_ae_days": med,
                             "val_ipcw_mae_days": ipw, "fit_seconds": round(time.time() - t0, 1)})  # fmt: skip
                print(grid[-1], file=sys.stderr, flush=True)
                if best is None or (-c, med) < best[:2]:
                    best = (-c, med, hp, pr, ipw)
            if best is not None:
                sel[fam] = {"hp": best[2], "val_harrell_c": -best[0], "val_ipcw_mae_days": best[4]}
                fitted[fam] = best[3]
        champ = min(sel, key=lambda f: sel[f]["val_ipcw_mae_days"])
        lg = lambda v: np.log1p(np.maximum(v, 0))  # noqa: E731
        ysel = np.log1p(sva.t)[sva.ci]
        qh = {"champion": q4.cqr_q(lg(fitted[champ]["q10"])[sva.ci], lg(fitted[champ]["q90"])[sva.ci], ysel),  # fmt: skip
              "naive": q4.cqr_q(lg(bva["naive_train_closed_median"]["q10"])[sva.ci], lg(bva["naive_train_closed_median"]["q90"])[sva.ci], ysel)}  # fmt: skip
        out_dir.mkdir(parents=True, exist_ok=True)
        pf = out_dir / "pred_validation_champion.parquet"
        pd.DataFrame({"number": va["number"], "point": fitted[champ]["point"], "risk": fitted[champ]["risk"]}).to_parquet(pf, index=False)  # fmt: skip
        val = {"champion": evaluate(fitted[champ], sva, g_tr, w_va_tr, qh["champion"]),
               "families_selected": {f: evaluate(fitted[f], sva, g_tr, w_va_tr, None) for f in sel},
               "baselines": {k: evaluate(p, sva, g_tr, w_va_tr, qh["naive"] if k.startswith("naive") else None) for k, p in bva.items()}}  # fmt: skip
        result = {
            "script": "scripts/r10_vscode_survival.py",
            "seed": SEED,
            "validation_grid": grid,
            "selection": sel,
            "champion": champ,
            "cqr_qhat_log": qh,
            "validation": val,
            "features": prep["cols"],
            "n_train": len(tr),
            "train_censored_share": float(1 - etr.mean()),  # fmt: skip
            "served_constant_days": auxes["served_d"],
            "versions": versions(),
            "freeze": {
                "champion": champ,
                "hp": sel[champ]["hp"],
                "test_read": False,
                "note": "test not yet read",
                "prereg_sha256": sha256(PREREG),
                "split_manifest_sha256": sha256(MANIFEST),
                "pred_validation_sha256": sha256(pf),
            },  # fmt: skip
        }
    else:
        champ, hp = result["freeze"]["champion"], result["freeze"]["hp"]
        assert result["freeze"]["prereg_sha256"] == sha256(PREREG), "pre-registration changed"
        model = fit_family(champ, hp, xtr, ttr, etr)
        pr = model(xva)
        pf = out_dir / "pred_validation_champion.parquet"
        assert sha256(pf) == result["freeze"]["pred_validation_sha256"], "validation preds file changed"  # fmt: skip
        assert np.allclose(pd.read_parquet(pf)["point"], pr["point"], rtol=1e-6), "refit drift"
        result["freeze"]["test_read"] = True  # flip BEFORE any test target is read; no re-run
        result["freeze"]["note"] = "test read once"
        report.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        result["test"] = {}
        for popn in ("eligible", "nonleaky"):
            te = pop_rows(tab, "test", popn, True)
            ste, xte = Split(te), apply_prep(te, prep)
            w_te_tr = q4.ipcw_weights(ttr, etr, ste.t)
            _, bte = baselines(tr, len(te))
            pc = model(xte)
            qh = result["cqr_qhat_log"]
            nbc = N_BOOT_C if popn == "eligible" else 100  # secondary: cheaper C bootstrap
            ent = {"champion": evaluate(pc, ste, g_tr, w_te_tr, qh["champion"] if popn == "eligible" else None, nbc),  # fmt: skip
                   "baselines": {k: evaluate(p, ste, g_tr, w_te_tr, qh["naive"] if k.startswith("naive") and popn == "eligible" else None, nbc) for k, p in bte.items()},  # fmt: skip
                   "paired_gain_vs": {k: paired_gain(bte[k], pc, ste) for k in ("served_const_3.84d", "naive_train_closed_median", "km_train_median")},  # fmt: skip
                   "calibration": calibration(pc, ste), "calibration_km_baseline": calibration(bte["km_train_median"], ste),  # fmt: skip
                   "test_period": [str(te["created_at"].min()), str(te["created_at"].max())]}  # fmt: skip
            ent["criteria"] = criteria({k: ent["paired_gain_vs"][k] for k in ("served_const_3.84d", "naive_train_closed_median")}, ent["champion"])  # fmt: skip
            result["test"][popn] = ent
            pd.DataFrame({"number": te["number"], "point": pc["point"], "risk": pc["risk"]}).to_parquet(out_dir / f"pred_test_{popn}.parquet", index=False)  # fmt: skip
        result["freeze"]["pred_test_eligible_sha256"] = sha256(
            out_dir / "pred_test_eligible.parquet"
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R10 vscode resolution survival study")
    ap.add_argument("--final-test", action="store_true", help="evaluate TEST once (explicit)")
    ap.add_argument("--report", type=Path, default=REPORT)
    a = ap.parse_args(argv)
    out_dir = ROOT / "r10_surv"
    out_dir.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":  # BelowNormal: laptop jobs must not starve the desktop
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    p = out_dir / "features.parquet"
    if p.exists():
        tab = pd.read_parquet(p)
    else:
        tab = build_table(ROOT, SLUG)
        tab.to_parquet(p, index=False)
    with threadpool_limits(limits=4):
        r = run("final" if a.final_test else "validate", tab, a.report, out_dir)
    brief = r["test"] if a.final_test else r["validation"]
    print(json.dumps(r["selection"] if not a.final_test else {k: v["criteria"] for k, v in brief.items()}, indent=1, default=str))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
