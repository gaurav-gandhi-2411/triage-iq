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
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_eval as ev  # noqa: E402
import qp4_resolution as q4  # noqa: E402
from r10_survival_lib import (  # noqa: E402
    FAMILIES,
    GRIDS,
    HORIZONS_D,
    MANIFEST,
    N_BOOT,
    N_BOOT_C,
    PREREG,
    REPORT,
    ROOT,
    SEED,
    SERVED_TRAIN,
    SLUG,
    FinalTestError,
    Split,
    abs_err_days,
    apply_prep,
    boot_pair,
    build_table,
    c_indexes,
    const_pred,
    fit_family,
    fit_prep,
    ipcw_mae,
)


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
