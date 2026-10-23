"""Phase 3b: kNN resolution estimate on kubernetes/kubernetes (spec sections 8-9, runbook Q7).

PRE-REGISTERED DESIGN (2026-10-23; do not change after seeing test results).
Context: the test split was already used once by the Phase 4 models M0-M2 (qp4_resolution.py,
naive = train closed median 35.1 d). The new model ids here (K1, C1) are each tested once; the
test split is therefore not fresh for the study as a whole, only for these two ids.
Reference set: TRAIN-split issues only (all train rows of qp4's feature table, as in qp4's kNN
feature). Embeddings: plain BGE document embeddings for query and reference (the same files and
symmetric issue-to-issue use as qp4; the `issues_query_instr` variant is NOT used).
Censoring at query time: for a query q created at t_q, a candidate neighbour n must be created
before t_q (else nothing about it was knowable) and n != q. n contributes an observed duration
(n.resolution) iff closed_at_n <= t_q; otherwise it is right-censored at t_q - created_n (this
includes issues that are still open at the scrape cutoff). The k nearest candidates by cosine are
taken AFTER this eligibility filter (the k most similar issues that already existed).
K1: Kaplan-Meier median over the k neighbours' (duration, event) pairs (smallest event time with
S <= 0.5); if the curve never reaches 0.5 the naive train closed median (35.1 d) is used and the
fallback is counted. k is selected on VALIDATION from {5, 10, 20, 50} by median AE (days) on the
closed eligible validation issues (qp4's M2 closed-only population), tie-break mean AE. The
interval for K1 is the KM 10%/90% quantiles of the neighbours (naive empirical quantile if
undefined), CQR-adjusted on validation closed issues exactly like qp4. No similarity-weighted
variant (K2): it would add a temperature hyper-parameter, not trivially cheap.
C1 = geometric mean of K1's estimate and the saved Phase 4 M2 point (pred_*.parquet from qp4; M2
is not retrained); interval ends are combined the same way; no IBS for C1 (no survival curve).
Evaluation = qp4's code paths (evaluate_model, paired_verdict): median/mean AE (days) vs naive M0a
on closed eligible test issues, paired bootstrap 2000 draws seed 42, Harrell C on ALL eligible
test issues with censoring (risk = -estimate), IBS (IPCW, censoring KM from train) as the
censoring-aware full-population view for K1, coverage/width of the 80% interval. Secondary
paired comparison K1 and C1 against M2. Serving rule (spec section 9): a point is served only if
the median-AE gain vs naive has a 95% CI lower bound > 0. Nothing in production changes here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_eval as ev  # noqa: E402
import qp4_resolution as q4  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
KS, KMAX = (5, 10, 20, 50), 50
IDS = {"K1": "qp3b_K1_knn_km_median", "C1": "qp3b_C1_geomean_K1_M2"}
SLUG = "kubernetes__kubernetes"


# ---- censoring-at-query-time kNN -----------------------------------------------------------
def neighbour_pairs(
    t_q: float, created: np.ndarray, dur: np.ndarray, event: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(duration, observed) of reference issues as knowable at t_q (all in hours).

    Closed by t_q (event and created + dur <= t_q) -> observed dur; otherwise censored at the
    age t_q - created (also for issues that are still open at the scrape cutoff).
    """
    closed = event & (created + dur <= t_q)
    return np.where(closed, dur, t_q - created), closed


def km_q(t: np.ndarray, e: np.ndarray, q: float) -> float:
    """Kaplan-Meier time with F >= q (smallest event time with S <= 1 - q); NaN if undefined."""
    ts, s = q4.km(t, e)
    hit = np.flatnonzero(s <= 1 - q)
    return float(ts[hit[0]]) if len(hit) else float("nan")


def km_surv(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    """S at the 9 finite qp4 bin edges (all 1 if there is no event)."""
    if not e.any():
        return np.ones(q4.NB - 1)
    return q4.km_at(*q4.km(t, e), q4.EDGES[1 : q4.NB])


def neighbour_ids(
    sims: np.ndarray, t_q: float, q_num: int, created: np.ndarray, nums: np.ndarray, kmax: int
) -> np.ndarray:
    """Indices of the kmax most similar eligible reference issues (created < t_q, not q itself)."""
    ok = np.flatnonzero((created < t_q) & (nums != q_num))
    return ok[np.argsort(-sims[ok], kind="stable")[:kmax]]


def knn_k1(d: pd.DataFrame, ref: pd.DataFrame, e_d: np.ndarray, e_ref: np.ndarray, k: int,
           naive: dict) -> dict:  # fmt: skip
    """K1 predictions for query table d (rows aligned with embeddings e_d) with k neighbours."""
    sims_all = e_d @ e_ref.T
    cr_d, cr_r = _hours(d["created_at"]), _hours(ref["created_at"])
    dur_r, ev_r, nums = ref["T"].to_numpy(), ref["event"].to_numpy().astype(bool), ref["number"].to_numpy()  # fmt: skip
    n = len(d)
    out = {x: np.zeros(n) for x in ("point", "q10", "q90")} | {"fell_back": np.zeros(n, bool), "surv": np.zeros((n, q4.NB - 1))}  # fmt: skip
    for i in range(n):
        idx = neighbour_ids(sims_all[i], cr_d[i], int(d["number"].iloc[i]), cr_r, nums, k)
        t, e = neighbour_pairs(cr_d[i], cr_r[idx], dur_r[idx], ev_r[idx])
        med = km_q(t, e, 0.5)
        out["fell_back"][i] = np.isnan(med)
        out["point"][i] = naive["median_h"] if np.isnan(med) else med
        lo, hi = km_q(t, e, 0.1), km_q(t, e, 0.9)
        out["q10"][i], out["q90"][i] = (naive["q"][0] if np.isnan(lo) else lo), (naive["q"][1] if np.isnan(hi) else hi)  # fmt: skip
        out["surv"][i] = km_surv(t, e)
    out["risk"] = -out["point"]
    return out


def combine_c1(k1: dict, m2_point: np.ndarray, m2_q10: np.ndarray, m2_q90: np.ndarray) -> dict:
    """C1: geometric mean of K1 and M2 (point and interval ends); no survival curve."""
    gm = lambda a, b: np.sqrt(np.maximum(a, 1e-6) * np.maximum(b, 1e-6))  # noqa: E731
    pt = gm(k1["point"], m2_point)
    return dict(point=pt, q10=gm(k1["q10"], m2_q10), q90=gm(k1["q90"], m2_q90), risk=-pt, surv=k1["surv"])  # fmt: skip


def _hours(s: pd.Series) -> np.ndarray:
    return s.to_numpy().astype("datetime64[s]").astype("int64") / 3600.0


# ---- stages --------------------------------------------------------------------------------
def load(root: Path, final: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """qp4 feature table (targets T / event / created_at only) and row-aligned unit embeddings."""
    tab = pd.read_parquet(root / "models" / SLUG / "qp4" / "features.parquet",
                          columns=["number", "split", "classifier_eligible", "created_at", "T", "event", "follow_h"])  # fmt: skip
    if not final:  # structurally unable to see test targets before the final stage
        tab = tab[tab["split"] != "test"].reset_index(drop=True)
    emb_dir = root / "embeddings" / SLUG
    emb = np.load(emb_dir / f"{SLUG}__issues_emb.npy").astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = pd.read_parquet(emb_dir / f"{SLUG}__issues_ids.parquet")["number"]
    pos = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    return tab, emb[pos[tab["number"]].to_numpy()]


def sha_info(root: Path) -> dict:
    files = {"issues": root / "derived_v2" / SLUG / "issues.parquet",
             "emb": root / "embeddings" / SLUG / f"{SLUG}__issues_emb.npy",
             "qp4_features": root / "models" / SLUG / "qp4" / "features.parquet"}  # fmt: skip
    files |= {s: root / "splits" / SLUG / f"{s}.parquet" for s in ("train", "validation", "test")}
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()  # noqa: S607
    return {"git_sha": sha, "sha256": {k: ev.file_sha256(p) for k, p in files.items()}}


def run(stage: str, root: Path, report: Path, manifest: Path, n_boot: int = ev.N_BOOT) -> dict:
    """validate: pick k on validation; final-test: evaluate K1 and C1 once on test."""
    final = stage == "final-test"
    out = root / "models" / SLUG / "knn_resolution"
    out.mkdir(parents=True, exist_ok=True)
    tab, emb = load(root, final)
    writer = ev.ResultsWriter(report, manifest)
    for m in IDS.values():  # fail before touching any test target
        if final and "test" in writer.data["models"].get(m, {}):
            raise ev.TestGuardError(f"test already evaluated for {m} in {report.name}")
    tr_all = (tab["split"] == "train").to_numpy()
    ref, e_ref = tab[tr_all].reset_index(drop=True), emb[tr_all]
    el = ref[ref["classifier_eligible"]]  # naive and CQR conventions of qp4: eligible train rows
    ttr, etr = el["T"].to_numpy(), el["event"].to_numpy().astype(bool)
    naive, maj = q4.naive_aux("M0a", ttr, etr), int(np.bincount(q4.bucket(ttr[etr])).argmax())
    g_tr = (ttr, etr)

    def split_rows(name: str) -> tuple[pd.DataFrame, np.ndarray]:
        ev.guard(name, final)
        m = ((tab["split"] == name) & tab["classifier_eligible"]).to_numpy()
        return tab[m].reset_index(drop=True), emb[m]

    def m2_preds(name: str, d: pd.DataFrame) -> pd.DataFrame:
        p = pd.read_parquet(root / "models" / SLUG / "qp4" / f"pred_{name}.parquet")
        assert (p["number"].to_numpy() == d["number"].to_numpy()).all(), "qp4 row order drift"
        return p

    va, e_va = split_rows("validation")
    tva, eva = va["T"].to_numpy(), va["event"].to_numpy().astype(bool)
    result: dict = {"design": DESIGN, "provenance": sha_info(root), "naive_median_h": naive["median_h"]}  # fmt: skip
    k_pred = {k: knn_k1(va, ref, e_va, e_ref, k, naive) for k in KS}
    if final:
        k = json.loads(report.read_text(encoding="utf-8"))["selected_k"]
    else:
        table = []
        for kk, pr in k_pred.items():
            ae = np.abs(pr["point"][eva] - tva[eva]) / 24
            table.append({"k": kk, "val_median_ae_days": float(np.median(ae)), "val_mean_ae_days": float(ae.mean()), "val_fallback_rate": float(pr["fell_back"].mean())})  # fmt: skip
        k = min(table, key=lambda r: (r["val_median_ae_days"], r["val_mean_ae_days"]))["k"]
        result |= {"selected_k": k, "validation_k_table": table}
    lg = lambda v: np.log1p(np.maximum(v, 0))  # noqa: E731

    def stage_preds(d: pd.DataFrame, e_d: np.ndarray, name: str) -> dict:
        k1 = k_pred[k] if name == "validation" else knn_k1(d, ref, e_d, e_ref, k, naive)
        p = m2_preds(name, d)
        m2 = dict(point=p["M2_point"].to_numpy(), q10=p["M2_q10"].to_numpy(), q90=p["M2_q90"].to_numpy())  # fmt: skip
        n0 = p["M0a_point"].to_numpy()
        assert np.allclose(n0, naive["median_h"]), "naive median differs from qp4's M0a"
        return {"K1": k1, "C1": combine_c1(k1, m2["point"], m2["q10"], m2["q90"]), "M0a": dict(point=n0), "M2": m2}  # fmt: skip

    pva = stage_preds(va, e_va, "validation")
    cal = {m: q4.cqr_q(lg(pva[m]["q10"])[eva], lg(pva[m]["q90"])[eva], np.log1p(tva)[eva]) for m in IDS}  # fmt: skip
    if final:
        d, e_d, sname = *split_rows("test"), "test"  # the first and only place test targets load
        prs = stage_preds(d, e_d, sname)
    else:
        d, sname, prs = va, "validation", pva
    res = {m: q4.evaluate_model(m, prs[m], d, g_tr, cal[m], maj, n_boot) for m in IDS}
    res["C1"] = {k_: v for k_, v in res["C1"].items() if not k_.startswith("ibs")}  # no S(t) for C1
    for m in IDS:
        res[m]["fallback_rate"] = float(prs["K1"]["fell_back"].mean())
        res[m]["vs_naive"] = q4.paired_verdict(prs["M0a"], prs[m], d, n_boot)
        res[m]["vs_m2"] = q4.paired_verdict(prs["M2"], prs[m], d, n_boot)
    result[sname] = res
    pf = out / f"pred_{sname}.parquet"
    pd.DataFrame({"number": d["number"], "k": k, "K1_fell_back": prs["K1"]["fell_back"]} | {f"{m}_{x}": prs[m][x] for m in IDS for x in ("point", "q10", "q90")}).to_parquet(pf, index=False)  # fmt: skip
    for m, mid in IDS.items():
        writer.add(mid, sname, res[m], pf, final=final)
    writer.data |= {x: v for x, v in result.items() if x not in ("validation", "test")}
    writer.save()
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3b kNN resolution-time study (k8s)")
    ap.add_argument("--final-test", action="store_true", help="evaluate TEST once (explicit)")
    ap.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    rep = Path("reports")
    r = run("final-test" if a.final_test else "validate", Path(r"D:\triageiq-data"),
            a.report or rep / "qp3b_knn_resolution_kubernetes__kubernetes_2026-10.json",
            rep / "qp0_split_manifest_kubernetes__kubernetes_2026-10.json")  # fmt: skip
    for s in ("validation", "test"):
        if s in r:
            print(s, json.dumps({m: {k: v["point"] for k, v in x.items() if isinstance(v, dict) and "point" in v} for m, x in r[s].items()}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
