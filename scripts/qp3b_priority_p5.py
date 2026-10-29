"""Phase 3b rework: P5, a two-stage abstaining priority model (decision G2; D88).

The pre-registered design is docs/QP3B_PRIORITY_P5_PREREG_2026-10.md (committed before any P5
training; this file implements it unchanged). Summary: stage 1 = text TF-IDF LR "will this issue
ever carry a priority level" over ALL issues of a split, gate threshold on validation (F0.5);
stage 2 = P4's features/LR on the prioritised issues with a validation-chosen class re-weighting
(w_low, w_medium, w_high) = (1, m, h). Serve-time inputs only (title + body). P5 abstains (the LLM
priority is kept) when the gate says no. Stage 2 is compared with P4 on the same rows. The freeze
record is written by `validate`; `--final-test` refuses to run without it and evaluates TEST once.
Offline only: no serving, flag or artifact-upload change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp3b_priority as pr  # noqa: E402

ev, t1, tf = pr.ev, pr.t1, pr.tf
CLASSES = pr.CLASSES
GRID_C = pr.GRID_C
GRID_M = (1.0, 1.1, 1.2, 1.3, 1.5)
GRID_H = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
BETA = 0.5
P5_ID = "p5_two_stage_abstain"
HIGH = CLASSES.index("high")
SEED = 42


def load_all(derived: Path, splits_dir: Path, name: str) -> pd.DataFrame:
    """Every issue of ONE split (no filter): text, gold3 (None if unprioritised), positive flag."""
    sp = pd.read_parquet(splits_dir / f"{name}.parquet", columns=["number"])
    iss = pd.read_parquet(derived / "issues.parquet")
    iss = iss[iss["number"].isin(sp["number"])].sort_values("number").reset_index(drop=True)
    d = pd.DataFrame({"number": iss["number"]})
    d["text"] = iss["title"].fillna("") + "\n" + iss["body"].fillna("").str.slice(0, 4000)
    d["gold3"] = iss["labels_final"].map(pr.gold3)
    d["positive"] = d["gold3"].notna()
    d["prio_t72h"] = iss["labels_t72h"].map(lambda ls: any(x.startswith("priority/") for x in ls))
    return d


class Stage1:
    """Text features + balanced LR for 'will be prioritised' (probability of the positive class)."""

    def __init__(self, c: float) -> None:
        self.c, self.feat = c, tf.Featurizer(True, False)

    def fit(self, d: pd.DataFrame) -> Stage1:
        y = ["pos" if v else "neg" for v in d["positive"]]
        self.lr = t1.fit_lr(self.feat.fit_transform(d["text"]), y, self.c, "balanced")
        return self

    def proba(self, d: pd.DataFrame) -> np.ndarray:
        p = self.lr.predict_proba(self.feat.transform(d["text"]))
        return p[:, list(self.lr.classes_).index("pos")]


def best_threshold(y: np.ndarray, p: np.ndarray, beta: float = BETA) -> tuple[float, float]:
    """Probability threshold maximising F-beta of the positive class; ties -> the higher threshold."""
    best_t, best_f = 1.0, -1.0
    for t in sorted(set(p.tolist()), reverse=True):
        pred = p >= t
        tp = float((pred & y).sum())
        prec, rec = tp / max(pred.sum(), 1), tp / max(y.sum(), 1)
        f = 0.0 if tp == 0 else (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        if f > best_f + 1e-12:
            best_t, best_f = t, f
    return float(best_t), float(best_f)


def weights(m: float, h: float) -> np.ndarray:
    return np.array([1.0, m, h])


def predict_weighted(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    """argmax of class probabilities times the multiplicative class weights."""
    return (p * w).argmax(axis=1)


def sel_key2(r: dict[str, Any]) -> tuple:
    """Lower is better: macro-F1, accuracy, closeness of the weights to 1, smaller C."""
    return (-r["macro_f1"], -r["accuracy"], abs(np.log(r["m"])) + abs(np.log(r["h"])), r["C"])


def stage1_stats(y: np.ndarray, p: np.ndarray, t: float) -> dict[str, Any]:
    gate = p >= t
    tp = int((gate & y).sum())
    return {"n": len(y), "prevalence": float(y.mean()), "average_precision": float(average_precision_score(y, p)),  # fmt: skip
            "roc_auc": float(roc_auc_score(y, p)), "threshold": t, "gate_pass_rate": float(gate.mean()),  # fmt: skip
            "precision_at_t": tp / max(int(gate.sum()), 1), "recall_at_t": tp / max(int(y.sum()), 1)}  # fmt: skip


def stage2_stats(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    cm = np.array(pr.confusion(y, pred))
    return {**pr.acc_f1(y, pred), "confusion_matrix_gold_rows_pred_cols": cm.tolist(),
            "per_class_recall": {k: float(cm[i, i] / max(cm[i].sum(), 1)) for i, k in enumerate(CLASSES)},  # fmt: skip
            "predicted_distribution": {k: int((pred == i).sum()) for i, k in enumerate(CLASSES)},
            "gold_distribution": {k: int((y == i).sum()) for i, k in enumerate(CLASSES)}}  # fmt: skip


def sha_text(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def validate(derived: Path, splits_dir: Path, models_dir: Path, report: Path) -> dict[str, Any]:
    """Fit on train, select on VALIDATION only, write the freeze record. Test is never loaded."""
    tr, va = (load_all(derived, splits_dir, n) for n in ("train", "validation"))
    # stage 2 rows must be exactly P4's population
    assert list(tr[tr["positive"]]["number"]) == list(pr.load_split(derived, splits_dir, "train")[0]["number"])  # fmt: skip
    yv1 = va["positive"].to_numpy()
    s1 = []
    for c in GRID_C:
        p = Stage1(c).fit(tr).proba(va)
        s1.append({"C": c, "average_precision": float(average_precision_score(yv1, p)), "p": p})
        print(f"stage1 C={c}: AP={s1[-1]['average_precision']:.4f} prev={yv1.mean():.3f}", file=sys.stderr)  # fmt: skip
    b1 = min(s1, key=lambda r: (-r["average_precision"], r["C"]))
    t, f05 = best_threshold(yv1, b1["p"])
    st1 = stage1_stats(yv1, b1["p"], t)
    trp, vap = tr[tr["positive"]].reset_index(drop=True), va[va["positive"]].reset_index(drop=True)
    yv2 = ev.to_idx(vap["gold3"], CLASSES)
    rows, probas = [], {}
    for c in GRID_C:
        pv = pr.Model(False, c).fit(trp).proba(vap)
        probas[c] = pv
        for m in GRID_M:
            for h in GRID_H:
                rows.append({"C": c, "m": m, "h": h, **pr.acc_f1(yv2, predict_weighted(pv, weights(m, h)))})  # fmt: skip
    b2 = min(rows, key=sel_key2)
    p4sel = json.loads((models_dir / "priority" / "p4" / "selection.json").read_text(encoding="utf-8"))  # fmt: skip
    p4_val = pd.read_parquet(models_dir / "priority" / "p4" / "validation.parquet")
    assert (p4_val["number"].to_numpy() == vap["number"].to_numpy()).all(), "P4 validation rows differ"  # fmt: skip
    pred2 = predict_weighted(probas[b2["C"]], weights(b2["m"], b2["h"]))
    pred4 = p4_val[CLASSES].to_numpy().argmax(axis=1)
    frozen = {"model_id": P5_ID, "stage1_C": b1["C"], "gate_threshold": t, "stage2_C": b2["C"],
              "m_medium": b2["m"], "h_high": b2["h"], "train_majority": pr.majority(list(trp["gold3"])),  # fmt: skip
              "p4_C": p4sel["C"], "beta": BETA, "seed": SEED}  # fmt: skip
    out = models_dir / "priority" / "p5"
    out.mkdir(parents=True, exist_ok=True)
    t1.save_proba(out / "validation_stage2.parquet", vap["number"], probas[b2["C"]], CLASSES)
    pd.DataFrame({"number": va["number"], "p_prioritised": b1["p"]}).to_parquet(out / "validation_stage1.parquet", index=False)  # fmt: skip
    (out / "frozen.json").write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    data = {"design_doc": "docs/QP3B_PRIORITY_P5_PREREG_2026-10.md",
            "population": {"train": {"n": len(tr), "n_prioritised": int(tr["positive"].sum())},
                           "validation": {"n": len(va), "n_prioritised": int(vap.shape[0])}},  # fmt: skip
            "validation": {"stage1_grid": [{k: v for k, v in r.items() if k != "p"} for r in s1],
                           "stage1_f05_at_threshold": f05, "stage1": st1,
                           "stage2_grid_size": len(rows), "stage2_selected": b2,
                           "stage2_p5": stage2_stats(yv2, pred2), "stage2_p4_same_rows": stage2_stats(yv2, pred4)},  # fmt: skip
            "freeze": {"frozen": frozen, "frozen_sha256": sha_text(frozen),
                       "p4_validation_predictions_sha256": ev.file_sha256(models_dir / "priority" / "p4" / "validation.parquet")}}  # fmt: skip
    report.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    return data["freeze"]


def final_test(derived: Path, splits_dir: Path, models_dir: Path, report: Path, n_boot: int = ev.N_BOOT) -> dict[str, Any]:  # fmt: skip
    """Evaluate the FROZEN P5 on TEST once (refuses without the freeze record / on a repeat)."""
    data = json.loads(report.read_text(encoding="utf-8"))
    if "test" in data:
        raise ev.TestGuardError(f"test already evaluated for {P5_ID} in {report.name}")
    out = models_dir / "priority" / "p5"
    frozen = json.loads((out / "frozen.json").read_text(encoding="utf-8"))
    assert sha_text(frozen) == data["freeze"]["frozen_sha256"], "frozen config changed after freeze"
    tr, va = (load_all(derived, splits_dir, n) for n in ("train", "validation"))
    s1 = Stage1(frozen["stage1_C"]).fit(tr)
    saved1 = pd.read_parquet(out / "validation_stage1.parquet")["p_prioritised"].to_numpy()
    assert np.allclose(s1.proba(va), saved1, atol=1e-6), "stage 1 refit differs from frozen run"
    trp = tr[tr["positive"]].reset_index(drop=True)
    vap = va[va["positive"]].reset_index(drop=True)
    m2 = pr.Model(False, frozen["stage2_C"]).fit(trp)
    saved2 = pd.read_parquet(out / "validation_stage2.parquet")[CLASSES].to_numpy()
    assert np.allclose(m2.proba(vap), saved2, atol=1e-6), "stage 2 refit differs from frozen run"
    m4 = pr.Model(False, frozen["p4_C"]).fit(trp)
    maj = pr.majority(list(trp["gold3"]))
    assert maj == frozen["train_majority"], "train majority drifted since freeze"
    te = load_all(derived, splits_dir, "test")  # first and only place test labels are read
    w = weights(frozen["m_medium"], frozen["h_high"])
    gate_p = s1.proba(te)
    gate = gate_p >= frozen["gate_threshold"]
    pop = te["positive"].to_numpy()
    tep = te[pop].reset_index(drop=True)
    y = ev.to_idx(tep["gold3"], CLASSES)
    p2, p4 = m2.proba(tep), m4.proba(tep)
    pred5, pred4 = predict_weighted(p2, w), p4.argmax(axis=1)
    p4_saved = pd.read_parquet(models_dir / "priority" / "p4" / "test.parquet")
    assert (p4_saved["number"].to_numpy() == tep["number"].to_numpy()).all(), "P4 test rows differ"
    assert np.allclose(p4_saved[CLASSES].to_numpy(), p4, atol=1e-6), "P4 refit differs from saved"
    n, k = len(tep), len(CLASSES)
    h5, h0 = (pred5 == y).astype(float), (np.full(n, CLASSES.index(maj)) == y).astype(float)  # fmt: skip

    def pdiff(a: Any, b: Any) -> dict[str, Any]:
        return ev.paired_diff(a, b, n, n_boot)

    f1 = lambda pred: lambda i: ev.macro_f1(y[i], pred[i], k)  # noqa: E731
    gold_high_rows = float((y == HIGH).mean())
    served = gate & (predict_weighted(m2.proba(te), w) == HIGH)  # all issues, gate AND level high
    gold_high_all = float((te["gold3"] == "high").mean())
    p4_high_all = float((m4.proba(te).argmax(axis=1) == HIGH).mean())
    auc_ci = ev.bootstrap(lambda i: float(roc_auc_score(pop[i], gate_p[i])) if 0 < pop[i].sum() < len(i) else 0.5, len(te), n_boot)  # fmt: skip
    st1 = stage1_stats(pop, gate_p, frozen["gate_threshold"])
    c = {
        "C1_macro_f1_minus_p4": pdiff(f1(pred5), f1(pred4)),
        "C2_accuracy_minus_majority": pdiff(
            lambda i: float(h5[i].mean()), lambda i: float(h0[i].mean())
        ),  # fmt: skip
        "C3_high_share_gap_pp": 100 * abs(float((pred5 == HIGH).mean()) - gold_high_rows),
        "C4_served_high_share_all_issues": float(served.mean()),
        "C4_gold_high_share_all_issues": gold_high_all,  # fmt: skip
        "C4_gap_pp": 100 * (float(served.mean()) - gold_high_all),
        "C4_precision_at_t": st1["precision_at_t"],  # fmt: skip
        "C4_prevalence": st1["prevalence"],
        "C4_auc_bootstrap": auc_ci,
    }
    met = {"C1": c["C1_macro_f1_minus_p4"]["ci95"][0] > 0, "C2": c["C2_accuracy_minus_majority"]["ci95"][0] > 0,  # fmt: skip
           "C3": c["C3_high_share_gap_pp"] <= 3.0,
           "C4": c["C4_gap_pp"] <= 5.0 and st1["precision_at_t"] > st1["prevalence"] and auc_ci["ci95"][0] > 0.5}  # fmt: skip
    res = {"n_test_issues": len(te), "n_prioritised": n, "stage1": st1, "stage1_auc_bootstrap": auc_ci,  # fmt: skip
           "stage2_p5": stage2_stats(y, pred5), "stage2_p4_same_rows": stage2_stats(y, pred4),
           "stage2_majority": {"class": maj, "accuracy": float(h0.mean())},
           "end_to_end": {"gate_passed": int(gate.sum()), "gate_passed_truly_prioritised": int((gate & pop).sum()),  # fmt: skip
                          "served_high_all_issues_p5": float(served.mean()), "served_high_all_issues_p4": p4_high_all,  # fmt: skip
                          "gold_high_share_all_issues": gold_high_all},
           "criteria": c, "criteria_met": met, "all_met": all(met.values())}  # fmt: skip
    data["test"] = res
    report.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3b P5 (two-stage abstaining priority), offline")
    ap.add_argument("--final-test", action="store_true", help="evaluate TEST once (explicit)")
    a = ap.parse_args(argv)
    root = Path(r"D:\triageiq-data")
    args = (root / "derived_v2" / pr.SLUG, root / "splits" / pr.SLUG, root / "models" / pr.SLUG,
            Path("reports") / "qp3b_priority_p5_2026-10.json")  # fmt: skip
    t1.lower_priority()
    with threadpool_limits(limits=4):
        r = final_test(*args) if a.final_test else validate(*args)
    print(json.dumps(r, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
