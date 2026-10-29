"""R10 survival study library: features, models, metrics and bootstrap helpers.

Moved verbatim out of scripts/r10_vscode_survival.py (which keeps the evaluation stages, the CLI
and the pre-registration docstring); see that module for the study protocol.
"""

from __future__ import annotations

import re
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
