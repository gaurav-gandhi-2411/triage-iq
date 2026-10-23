"""Phase 3b follow-up: P4, the priority candidate whose inputs exist at serve time (spec section 8).

PRE-REGISTERED DESIGN P4 (2026-10-24; committed with its validation table BEFORE its test
evaluation; do not change after seeing test results). Context: the production TriageRequest carries
only repo, title, body, issue_number, created_at: no labels, no author association. P2 (t72h labels)
and P3 (creation-time labels) therefore cannot be served as built. The test split was already used
by P0, P2 and P3; P4 is evaluated on test exactly once (model id `p4_tfidf_text_lr`) because it is
the only candidate whose inputs exist at serve time.
 * Features = exactly P1's: qp1_tfidf_features.Featurizer(title_sep on, paths off) over
   title + "\\n" + body[:4000]; NO metadata. Model = multinomial LogisticRegression, class_weight
   balanced, max_iter 300, seed 42; C in {0.25, 1, 4} chosen on VALIDATION (accuracy, tie macro-F1,
   tie smaller C). P1's validation table (0.4526 / 0.4599 / 0.4964 -> C=4) is re-run and must match.
 * Same population, 3-level gold, train-majority baseline P0, paired bootstrap (2000 draws, seed 42)
   and acceptance (lower bound of the paired accuracy difference P4 - P0 > 0) as qp3b_priority.py.
 * On test also: P4 vs P3 paired (does creation-time metadata matter?) and the subset with no
   priority/* label at t72h. The fitted featuriser and model are saved with joblib under
   <models-dir>/priority/p4/ for later packaging.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp3b_priority as pr  # noqa: E402
import qp3b_priority_p3 as p3  # noqa: E402

ev, t1 = pr.ev, pr.t1
DESIGN_P4 = __doc__.split("PRE-REGISTERED DESIGN P4", 1)[1].strip()
P4_ID = "p4_tfidf_text_lr"


def validate_p4(derived: Path, splits_dir: Path, models_dir: Path, report: Path, manifest: Path) -> dict:  # fmt: skip
    """Stage 1: P4 grid on VALIDATION; checks the P1 table is reproduced. No test labels loaded."""
    (tr, _), (va, _) = (pr.load_split(derived, splits_dir, n) for n in ("train", "validation"))
    yva = ev.to_idx(va["gold3"], pr.CLASSES)
    rows, probas = [], {}
    for c in pr.GRID_C:
        pv = pr.Model(False, c).fit(tr).proba(va)
        rows.append({"model": P4_ID, "C": c, **pr.acc_f1(yva, pv.argmax(axis=1))})
        probas[c] = pv
        print(f"{P4_ID} C={c}: acc={rows[-1]['accuracy']:.4f} f1={rows[-1]['macro_f1']:.4f}", file=sys.stderr)  # fmt: skip
    w = ev.ResultsWriter(report, manifest)
    p1 = {r["C"]: r for r in w.data["validation_table"] if r["model"] == pr.P1_ID}
    assert all(abs(p1[r["C"]]["accuracy"] - r["accuracy"]) < 1e-9 for r in rows), "P1 not reproduced"  # fmt: skip
    best = min(rows, key=lambda r: (-r["accuracy"], -r["macro_f1"], r["C"]))
    out = models_dir / "priority" / "p4"
    out.mkdir(parents=True, exist_ok=True)
    t1.save_proba(out / "validation.parquet", va["number"], probas[best["C"]], pr.CLASSES)
    sel = {"model_id": P4_ID, "C": best["C"], "majority": pr.majority(list(tr["gold3"])), "val_accuracy": best["accuracy"]}  # fmt: skip
    (out / "selection.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
    w.data.update({"p4_design": DESIGN_P4, "p4_validation_table": rows, "p4_selection": sel, "p4_matches_p1_validation_table": True})  # fmt: skip
    w.save()
    return sel


def final_test_p4(derived: Path, splits_dir: Path, models_dir: Path, report: Path, manifest: Path, n_boot: int = ev.N_BOOT) -> dict:  # fmt: skip
    """Stage 2: refit P4, save artifacts, evaluate on TEST once; paired vs P0 and vs P3."""
    base = models_dir / "priority"
    out = base / "p4"
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    w = ev.ResultsWriter(report, manifest)
    if "test" in w.data["models"].get(P4_ID, {}):  # fail before any test label is read
        raise ev.TestGuardError(f"test already evaluated for {P4_ID} in {Path(report).name}")
    (tr, _), (va, _) = (pr.load_split(derived, splits_dir, n) for n in ("train", "validation"))
    m = pr.Model(False, sel["C"]).fit(tr)
    assert np.allclose(m.proba(va), pd.read_parquet(out / "validation.parquet")[pr.CLASSES].to_numpy(), atol=1e-6), "refit drifted"  # fmt: skip
    maj = pr.majority(list(tr["gold3"]))
    assert maj == sel["majority"], "train majority drifted since selection"
    te, _ = pr.load_split(
        derived, splits_dir, "test"
    )  # the first and only place P4 reads test labels
    y, pt = ev.to_idx(te["gold3"], pr.CLASSES), m.proba(te)
    pred, pred0 = pt.argmax(axis=1), np.full(len(te), pr.CLASSES.index(maj))
    q3 = pd.read_parquet(base / "test_p3.parquet")
    assert (q3["number"].to_numpy() == te["number"].to_numpy()).all(), "P3 test predictions misaligned"  # fmt: skip
    hit, hit0, hit3 = ((p == y).astype(float) for p in (pred, pred0, q3[pr.CLASSES].to_numpy().argmax(axis=1)))  # fmt: skip
    allm, pre = np.ones(len(te), dtype=bool), ~te["prio_t72h"].to_numpy()
    joblib.dump(m.feat, out / "featurizer.joblib")
    joblib.dump(m.lr, out / "model.joblib")
    pf = t1.save_proba(out / "test.parquet", te["number"], pt, pr.CLASSES)
    entry = {"n": len(te), **pr.acc_f1(y, pred), "accuracy_wilson95": list(ev.wilson(int(hit.sum()), len(te))),
             "confusion_matrix_gold_rows_pred_cols": pr.confusion(y, pred), "classes": pr.CLASSES}  # fmt: skip
    w.add(P4_ID, "test", entry, pf, final=True)
    res = {"model_id": P4_ID, "C": sel["C"], "note": "P2/P3 results in the sibling keys are unchanged",
           "accuracy": entry["accuracy"], "accuracy_wilson95": entry["accuracy_wilson95"],
           "vs_P0_majority": p3.paired_stats(hit, hit0, allm, n_boot), "vs_P3": p3.paired_stats(hit, hit3, allm, n_boot),  # fmt: skip
           "no_priority_label_at_t72h": {"vs_P0_majority": p3.paired_stats(hit, hit0, pre, n_boot), "vs_P3": p3.paired_stats(hit, hit3, pre, n_boot)},  # fmt: skip
           "artifact_sha256": {f: ev.file_sha256(out / f) for f in ("featurizer.joblib", "model.joblib")}}  # fmt: skip
    res["acceptance_ci_lower_gt_0"] = res["vs_P0_majority"]["difference_paired"]["ci95"][0] > 0
    w.data["test"]["p4"] = res
    w.save()
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3b P4 (serve-time text-only), offline")
    ap.add_argument("--final-test", action="store_true", help="evaluate TEST once (explicit)")
    a = ap.parse_args(argv)
    root, rep = Path(r"D:\triageiq-data"), Path("reports")
    args = (root / "derived_v2" / pr.SLUG, root / "splits" / pr.SLUG, root / "models" / pr.SLUG,
            rep / "qp3b_priority_kubernetes__kubernetes_2026-10.json", rep / f"qp0_split_manifest_{pr.SLUG}_2026-10.json")  # fmt: skip
    t1.lower_priority()
    with threadpool_limits(limits=4):
        r = final_test_p4(*args) if a.final_test else validate_p4(*args)
    print(json.dumps(r, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
