"""Phase 4: resolution time on kubernetes/kubernetes as a survival problem (spec section 9).

PRE-REGISTERED DESIGN (2026-10-23; do not change after seeing test results).
Target: hours from created_at to closed_at; an open issue is right-censored at the scrape cutoff
(scraped_at_max of the derived issue table, 2026-10-22 21:21:18 UTC). Population: classifier_eligible
issues of the frozen splits (train / validation / test; the test split is ~43% censored).
Features (qp4_features.py), creation time only: author_association; author history (earlier issues
by the same author, how many were closed by t_q and their median resolution - an earlier issue is
resolved only if closed_at <= t_q); title/body length, code block, stack trace, URL count; hour and
day of week; template type (first matching body header: bug form 1, feature form 2, failing-jobs 3,
flaking-tests 4, else 0); candidate-1 (TF-IDF+LR, C=64 balanced) probabilities, out-of-fold for train
rows (5 folds, seed 42), saved/refit for validation and test, plus top-1 and max probability;
open-issue load (issues created before t_q and not closed by t_q) in the predicted top-1 component
and overall; kNN: of the 20 nearest TRAIN issues (BGE cosine) that are closed at or before t_q, the
median log1p resolution and the share closed within 7 days; BGE PCA to 16 dims (fit on train).
Models: M0a naive = train closed median (production, ADR-0064); M0b = train Kaplan-Meier median.
M1 = discrete-time hazard, bins [0,1h) [1h,1d) [1d,3d) [3d,7d) [7d,14d) [14d,30d) [30d,90d)
[90d,180d) [180d,365d) [365d,inf); person-period rows: an event in bin k gives rows j<=k (y=1 only
at k); an issue censored in bin k gives rows j<k (treated as censored at the start of bin k);
LightGBM binary on features + bin index. Survival S_k = prod_{j<=k}(1-h_j); predicted median =
first bin with S <= 0.5 (else the last), POINT = geometric midpoint of that bin (bin 0: 0.5 h, last
bin: 547 d); quantiles for the interval use exponential-in-bin interpolation (last bin capped at
730 d). M1 ranking score for the C-index = -sum of S over the 9 finite bins. M2 = LightGBM
regression on log1p(hours) over CLOSED train issues only (censoring-naive; quantile objectives at
0.1/0.9 for its interval; ranking score = -prediction). Grid on validation per family:
num_leaves {15,31} x learning_rate {0.05} x n_estimators {200,400} x min_child_samples {20,50},
seed 42. Selection: validation C-index, tie-break validation median AE (days, closed issues).
Metrics (validation for selection, test once per model id), 95% percentile bootstrap, 2000
resamples, seed 42: Harrell C (comparable pairs i event, T_i < T_j; tied scores 0.5); AE of the
point on closed issues (log1p scale and days; median and mean; closed-only AE favours fast issues,
the censored share is reported); IBS = mean over the 9 finite bin boundaries of the IPCW Brier
score, censoring KM G from TRAIN (clipped at 0.01); the same with G from the evaluated split is
reported as a sensitivity; M0 survival = train KM, M2 survival = log-normal around its prediction
with sigma from validation residuals; bucket accuracy (1/7/30/180 d) vs the train-majority bucket;
80% interval on log1p hours: CQR adjustment on VALIDATION closed issues of the model's 10%/90%
quantiles (M0a: empirical train closed quantiles, M0b: KM quantiles), coverage on closed issues
(target 80% +/- 3pp) and width in days, plus coverage on test issues with >= 180 d follow-up.
Serving rule: the learned point is served only if the paired-bootstrap CI (2000, seed 42) of
median-AE(naive M0a) - median-AE(model) on test closed issues excludes zero in the model's favour.
Nothing in production changes here.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_eval as ev  # noqa: E402
import qp4_features as feat  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
EDGES = np.array([0, 1, 24, 72, 168, 336, 720, 2160, 4320, 8760, np.inf])  # bin edges, hours
NB, GFLOOR, SEED, ALPHA, LAST_CAP_H = 10, 0.01, 42, 0.2, 730 * 24.0
MID = np.array([0.5, *np.sqrt(EDGES[1:9] * EDGES[2:10]), 547 * 24.0])  # geometric bin midpoints
BREAKS_H = np.array([24, 168, 720, 4320.0])  # bucket breaks of src/triage_iq/models/resolution.py
GRID = [dict(num_leaves=nl, learning_rate=0.05, n_estimators=ne, min_child_samples=mc)
        for nl in (15, 31) for ne in (200, 400) for mc in (20, 50)]  # fmt: skip
META = ["number", "split", "classifier_eligible", "created_at", "T", "event", "follow_h"]
IDS = {"M0a": "qp4_M0a_train_median", "M0b": "qp4_M0b_km_median", "M1": "qp4_M1_hazard", "M2": "qp4_M2_closed_only_regression"}  # fmt: skip


# ---- survival primitives -------------------------------------------------------------------
def km(t: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier (event times, S just after each): d_t events over n_t still at risk."""
    ts, d = np.unique(t[e], return_counts=True)
    at_risk = len(t) - np.searchsorted(np.sort(t), ts, "left")
    return ts, np.cumprod(1 - d / at_risk)


def km_at(ts: np.ndarray, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Right-continuous step value of a KM curve at t (1 before the first event time)."""
    k = np.searchsorted(ts, t, "right")
    return np.where(k > 0, s[np.maximum(k - 1, 0)], 1.0)


def km_quantile(ts: np.ndarray, s: np.ndarray, q: float) -> float:
    """Smallest event time with S <= 1 - q (last event time if the curve never gets there)."""
    hit = np.flatnonzero(s <= 1 - q)
    return float(ts[hit[0]] if len(hit) else ts[-1])


def ipcw_weights(t_tr: np.ndarray, e_tr: np.ndarray, at: np.ndarray) -> np.ndarray:
    """1 / G(at), G = KM of the censoring distribution (censorings play the events), floored."""
    ts, g = km(t_tr, ~e_tr)
    return 1.0 / np.maximum(km_at(ts, g, at), GFLOOR)


def bin_index(t: np.ndarray) -> np.ndarray:
    return np.searchsorted(EDGES[1:], t, "right").clip(0, NB - 1)


def expand(
    x: np.ndarray, t: np.ndarray, e: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Person-period expansion: (rows of x, bin index, y). Censored in bin k -> rows j < k only."""
    k = bin_index(t)
    last = np.where(e, k, k - 1)  # last bin with a row
    rows = np.repeat(np.arange(len(t)), last + 1)
    j = np.concatenate([np.arange(n + 1) for n in last]) if len(t) else np.array([], int)
    y = (e[rows] & (j == k[rows])).astype(int)
    return x[rows], j, y


def surv_from_hazard(h: np.ndarray) -> np.ndarray:
    return np.cumprod(1 - h, axis=1)


def median_point_h(s: np.ndarray) -> np.ndarray:
    """Geometric midpoint (hours) of the first bin with S <= 0.5, else of the last bin."""
    hit = s <= 0.5
    return MID[np.where(hit.any(1), hit.argmax(1), NB - 1)]


def quantile_h(s: np.ndarray, q: float) -> np.ndarray:
    """Time (hours) with F = q: exponential-in-bin interpolation, last bin capped at 730 d."""
    prev = np.hstack([np.ones((len(s), 1)), s[:, :-1]])
    hit = s <= 1 - q
    k = np.where(hit.any(1), hit.argmax(1), NB - 1)
    r = np.arange(len(s))
    a, b = prev[r, k], np.maximum(s[r, k], 1e-12)
    frac = np.clip(np.log(a / (1 - q)) / np.maximum(np.log(a / b), 1e-12), 0, 1)
    hi = np.where(np.isinf(EDGES[k + 1]), LAST_CAP_H, EDGES[k + 1])
    return EDGES[k] + frac * (hi - EDGES[k])


# ---- metrics -------------------------------------------------------------------------------
def harrell_c(t: np.ndarray, e: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C: pairs (i event, T_i < T_j); concordant if risk_i > risk_j, tie counts 0.5."""
    comp = e[:, None] & (t[:, None] < t[None, :])
    r = risk[:, None] - risk[None, :]
    return float(((comp & (r > 0)).sum() + 0.5 * (comp & (r == 0)).sum()) / max(comp.sum(), 1))


def brier_ipcw(s: np.ndarray, t: np.ndarray, e: np.ndarray, g_tr: tuple) -> float:
    """Mean over the 9 finite bin boundaries of the IPCW Brier score; s[n, 9] = S at the edges."""
    at = EDGES[1:NB]
    w_t = ipcw_weights(*g_tr, t)  # weight of an event seen before the boundary
    w_b = ipcw_weights(*g_tr, at)  # weight of a subject still at risk at the boundary
    dead = e[:, None] & (t[:, None] <= at[None, :])
    alive = t[:, None] > at[None, :]
    bs = (dead * s**2 * w_t[:, None] + alive * (1 - s) ** 2 * w_b[None, :]).mean(0)
    return float(bs.mean())


def cqr_q(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> float:
    """Conformalised quantile regression adjustment from calibration closed issues."""
    sc = np.sort(np.maximum(lo - y, y - hi))
    k = int(np.ceil((len(sc) + 1) * (1 - alpha)))
    return float(sc[min(k, len(sc)) - 1])


def bucket(h: np.ndarray) -> np.ndarray:
    return np.searchsorted(BREAKS_H, h, "right")


def survival_table(model: str, x: np.ndarray, aux: dict) -> np.ndarray:
    """S at the 9 finite edges for any model id (n x 9)."""
    if model == "M1":
        return surv_from_hazard(aux["hazard"](x))[:, : NB - 1]
    if model == "M2":
        mu = aux["pred"](x)[:, None]
        return 1 - norm.cdf((np.log1p(EDGES[None, 1:NB]) - mu) / aux["sigma"])
    return np.tile(km_at(*aux["km"], EDGES[1:NB]), (len(x), 1))


# ---- models --------------------------------------------------------------------------------
def lgb_params(p: dict, **kw) -> dict:  # noqa: ANN003
    return {**p, "random_state": SEED, "n_jobs": 4, "verbose": -1, "deterministic": True, "force_row_wise": True, **kw}  # fmt: skip


def fit_m1(x: np.ndarray, t: np.ndarray, e: np.ndarray, p: dict):  # noqa: ANN201
    xp, j, y = expand(x, t, e)
    m = lgb.LGBMClassifier(**lgb_params(p)).fit(np.column_stack([xp, j]), y)

    def hazard(xn: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict_proba(np.column_stack([xn, np.full(len(xn), k)]))[:, 1] for k in range(NB)])  # fmt: skip

    return hazard


def predict_all(model: str, x: np.ndarray, aux: dict) -> dict:
    """Point (hours), 10%/90% quantiles (hours), ranking score (higher = faster), S table."""
    n = len(x)
    if model == "M1":
        s = surv_from_hazard(aux["hazard"](x))
        return dict(point=median_point_h(s), q10=quantile_h(s, 0.1), q90=quantile_h(s, 0.9), risk=-s[:, : NB - 1].sum(1), surv=s[:, : NB - 1])  # fmt: skip
    if model == "M2":
        mu = aux["pred"](x)
        return dict(point=np.expm1(mu), q10=np.expm1(aux["q10"](x)), q90=np.expm1(aux["q90"](x)), risk=-mu, surv=survival_table(model, x, aux))  # fmt: skip
    ts, s = aux["km"]
    pt = aux["median_h"]
    lo, hi = aux["q"]
    return dict(point=np.full(n, pt), q10=np.full(n, lo), q90=np.full(n, hi), risk=np.zeros(n), surv=survival_table(model, x, aux))  # fmt: skip


def fit_family(model: str, xtr, ttr, etr, p) -> dict:  # noqa: ANN001
    """Fitted callables for M1 / M2 with hyperparameters p."""
    if model == "M1":
        return {"hazard": fit_m1(xtr, ttr, etr, p)}
    yl, xc = np.log1p(ttr[etr]), xtr[etr]
    fits = {k: lgb.LGBMRegressor(**lgb_params(p, **kw)).fit(xc, yl) for k, kw in
            [("pred", {}), ("q10", dict(objective="quantile", alpha=0.1)), ("q90", dict(objective="quantile", alpha=0.9))]}  # fmt: skip
    return {k: m.predict for k, m in fits.items()} | {"sigma": float(np.std(yl - fits["pred"].predict(xc)))}  # fmt: skip


def naive_aux(kind: str, ttr, etr) -> dict:  # noqa: ANN001
    """M0a: train closed median; M0b: train KM median. Interval = closed empirical / KM quantiles."""
    ts, s = km(ttr, etr)
    closed = ttr[etr]
    if kind == "M0a":
        q = np.quantile(closed, [0.1, 0.9])
        return {"km": (ts, s), "median_h": float(np.median(closed)), "q": (q[0], q[1])}
    return {"km": (ts, s), "median_h": km_quantile(ts, s, 0.5), "q": (km_quantile(ts, s, 0.1), km_quantile(ts, s, 0.9))}  # fmt: skip


def evaluate_model(
    model: str, pr: dict, d: pd.DataFrame, g_tr: tuple, qhat: float, maj: int, n_boot: int
) -> dict:  # noqa: PLR0913
    """All point metrics with bootstrap CIs for one model on one split table d."""
    t, e = d["T"].to_numpy(), d["event"].to_numpy().astype(bool)
    ci = np.flatnonzero(e)
    tc, pt = t[ci], pr["point"][ci]
    lo, hi = (
        np.log1p(np.maximum(pr["q10"][ci], 0)) - qhat,
        np.log1p(np.maximum(pr["q90"][ci], 0)) + qhat,
    )
    ae_log, ae_d = np.abs(np.log1p(pt) - np.log1p(tc)), np.abs(pt - tc) / 24
    cov, wid = (np.log1p(tc) >= lo) & (np.log1p(tc) <= hi), (np.expm1(hi) - np.expm1(lo)) / 24
    long_fu = d["follow_h"].to_numpy()[ci] >= 180 * 24
    hit, hit_maj = bucket(pt) == bucket(tc), bucket(tc) == maj
    b = lambda f, n: ev.bootstrap(f, n, n_boot)  # noqa: E731
    out = {
        "n": len(t),
        "n_closed": len(ci),
        "censored_share": float(1 - e.mean()),
        "c_index": b(lambda i: harrell_c(t[i], e[i], pr["risk"][i]), len(t))
        if pr["risk"].std() > 0
        else {"point": 0.5, "ci95": [0.5, 0.5], "n": len(t)},  # fmt: skip
        "ae_log_median": b(lambda i: float(np.median(ae_log[i])), len(ci)),
        "ae_log_mean": b(lambda i: float(ae_log[i].mean()), len(ci)),  # fmt: skip
        "ae_days_median": b(lambda i: float(np.median(ae_d[i])), len(ci)),
        "ae_days_mean": b(lambda i: float(ae_d[i].mean()), len(ci)),  # fmt: skip
        "ibs_ipcw_train_G": b(lambda i: brier_ipcw(pr["surv"][i], t[i], e[i], g_tr), len(t)),
        "ibs_ipcw_split_G": b(
            lambda i: brier_ipcw(pr["surv"][i], t[i], e[i], (t[i], e[i])), len(t)
        ),  # fmt: skip
        "bucket_acc": b(lambda i: float(hit[i].mean()), len(ci)),
        "bucket_acc_majority": b(lambda i: float(hit_maj[i].mean()), len(ci)),  # fmt: skip
        "bucket_acc_minus_majority": b(
            lambda i: float(hit[i].mean() - hit_maj[i].mean()), len(ci)
        ),  # fmt: skip
        "coverage80": b(lambda i: float(cov[i].mean()), len(ci)),
        "mean_width_days": b(lambda i: float(wid[i].mean()), len(ci)),  # fmt: skip
        "median_width_days": float(np.median(wid)),
        "cqr_qhat_log": qhat,
        "coverage80_followup_ge180d": ev.bootstrap(
            lambda i: float(cov[long_fu][i].mean()), int(long_fu.sum()), n_boot
        )
        if long_fu.any()
        else None,  # fmt: skip
    }
    return out


def paired_verdict(pn: dict, pm: dict, d: pd.DataFrame, n_boot: int) -> dict:
    """Median/mean AE (days) naive minus model on closed issues; positive = model better."""
    ci = np.flatnonzero(d["event"].to_numpy().astype(bool))
    tc = d["T"].to_numpy()[ci]
    an, am = np.abs(pn["point"][ci] - tc) / 24, np.abs(pm["point"][ci] - tc) / 24
    med = ev.bootstrap(lambda i: float(np.median(an[i]) - np.median(am[i])), len(ci), n_boot)
    mean = ev.bootstrap(lambda i: float(an[i].mean() - am[i].mean()), len(ci), n_boot)
    return {
        "median_ae_gain_days": med,
        "mean_ae_gain_days": mean,
        "served": bool(med["ci95"][0] > 0),
    }


# ---- stages --------------------------------------------------------------------------------
def rows(tab: pd.DataFrame, split: str, final: bool) -> pd.DataFrame:
    """Eligible rows of one split; test targets are reachable only with final=True."""
    ev.guard(split, final)
    return tab[(tab["split"] == split) & tab["classifier_eligible"]].reset_index(drop=True)


def xmat(d: pd.DataFrame) -> np.ndarray:
    return d.drop(columns=META).to_numpy(dtype=np.float32)


def run(
    stage: str, root: Path, slug: str, report: Path, manifest: Path, n_boot: int = ev.N_BOOT
) -> dict:
    out = root / "models" / slug / "qp4"
    out.mkdir(parents=True, exist_ok=True)
    final = stage == "final-test"
    p = out / "features.parquet"
    if p.exists():
        tab = pd.read_parquet(p)
    else:
        tab = feat.build(root, slug, out / "oof_probs.npz")
        tab.to_parquet(p, index=False)
    writer = ev.ResultsWriter(report, manifest)
    for m in IDS.values():  # fail before touching any test target
        if final and "test" in writer.data["models"].get(m, {}):
            raise ev.TestGuardError(f"test already evaluated for {m} in {report.name}")
    tr, va = rows(tab, "train", False), rows(tab, "validation", False)
    xtr, ttr, etr = xmat(tr), tr["T"].to_numpy(), tr["event"].to_numpy().astype(bool)
    xva, tva, eva = xmat(va), va["T"].to_numpy(), va["event"].to_numpy().astype(bool)
    g_tr, maj, grid = (ttr, etr), int(np.bincount(bucket(ttr[etr])).argmax()), []
    if final:
        sel = json.loads(report.read_text(encoding="utf-8"))["selection"]
    else:  # stage 1: grid on validation, select by C-index then median AE
        sel = {}
        for m in ("M1", "M2"):
            best = None
            for hp in GRID:
                pr = predict_all(m, xva, fit_family(m, xtr, ttr, etr, hp))
                c = harrell_c(tva, eva, pr["risk"])
                mae = float(np.median(np.abs(pr["point"][eva] - tva[eva]) / 24))
                grid.append({"model": m, **hp, "val_c_index": c, "val_median_ae_days": mae})
                print(m, hp, round(c, 4), round(mae, 3), file=sys.stderr)
                if best is None or (-c, mae) < best[:2]:
                    best = (-c, mae, hp)
            sel[m] = best[2]
    auxs = {k: naive_aux(k, ttr, etr) for k in ("M0a", "M0b")} | {m: fit_family(m, xtr, ttr, etr, sel[m]) for m in ("M1", "M2")}  # fmt: skip
    pva = {m: predict_all(m, xva, a) for m, a in auxs.items()}
    lg = lambda v: np.log1p(np.maximum(v, 0))  # noqa: E731
    cal = {m: cqr_q(lg(pr["q10"])[eva], lg(pr["q90"])[eva], np.log1p(tva)[eva]) for m, pr in pva.items()}  # fmt: skip
    result = {"design": DESIGN, "selection": sel, "cqr_qhat_log": cal, "bucket_majority": maj}
    if final:
        pf_val = pd.read_parquet(out / "pred_validation.parquet")  # refit must equal stage 1
        for m in ("M1", "M2"):
            assert np.allclose(pf_val[f"{m}_point"], pva[m]["point"], rtol=1e-6), f"{m} refit drift"
        d = rows(tab, "test", True)  # the first and only place test targets are read
        sname, prs = "test", {m: predict_all(m, xmat(d), a) for m, a in auxs.items()}
        result["serving_verdict"] = {m: paired_verdict(prs["M0a"], prs[m], d, n_boot) for m in ("M0b", "M1", "M2")}  # fmt: skip
    else:
        sname, d, prs = "validation", va, pva
    result[sname] = {
        m: evaluate_model(m, pr, d, g_tr, cal[m], maj, n_boot) for m, pr in prs.items()
    }
    pf = out / f"pred_{sname}.parquet"
    pd.DataFrame({"number": d["number"]} | {f"{m}_{k}": v for m, pr in prs.items() for k, v in pr.items() if k != "surv"}).to_parquet(pf, index=False)  # fmt: skip
    for m in prs:
        writer.add(IDS[m], sname, result[sname][m], pf, final=final)
    writer.data |= {k: v for k, v in result.items() if k not in ("validation", "test")}
    if grid:
        writer.data["validation_grid"] = grid
    writer.save()
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 4 resolution-time survival study")
    ap.add_argument("--repo", default="kubernetes/kubernetes")
    ap.add_argument("--final-test", action="store_true", help="evaluate TEST once (explicit)")
    ap.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    slug, root = a.repo.replace("/", "__"), Path(r"D:\triageiq-data")
    rep = Path("reports")
    if sys.platform == "win32":  # BelowNormal: laptop jobs must not starve the desktop
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    with threadpool_limits(limits=4):
        r = run("final-test" if a.final_test else "validate", root, slug, a.report or rep / f"qp4_resolution_{slug}_2026-10.json", rep / f"qp0_split_manifest_{slug}_2026-10.json")  # fmt: skip
    print(json.dumps({s: {m: {k: v["point"] for k, v in x.items() if isinstance(v, dict) and "point" in v} for m, x in r[s].items()} for s in ("validation", "test") if s in r}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
