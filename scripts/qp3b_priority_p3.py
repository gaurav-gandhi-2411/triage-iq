"""Phase 3b follow-up: P3, the serve-time-feasible priority candidate (spec section 8).

PRE-REGISTERED DESIGN P3 (2026-10-23; committed with its validation table BEFORE its test
evaluation; do not change after seeing test results). Context: P0 and P2 of qp3b_priority.py
already used the test split once. A verifier found that P2's labels_t72h kind/* and sig/* inputs
are not creation-time (44/244 test issues had an input label applied within 60 s of or after the
first priority label), while /triage runs on a just-created issue. P3 is the serve-time-feasible
variant and is evaluated on test exactly once (model id `p3_tfidf_creationmeta_lr`).
 * Same population, gold (3-level), train-majority baseline P0, bootstrap (2000 draws, seed 42),
   acceptance (lower bound of paired accuracy difference vs P0 > 0) as in qp3b_priority.py.
 * Features = P1 features + one-hot CREATION-TIME kind/* and sig/* labels + one-hot author_assoc.
   Creation-time rule, identical for train / validation / test: take labels_t72h restricted to
   kind/* and sig/*; keep a label iff it has no `labeled` event in label_events for that issue, or
   its EARLIEST `labeled` event is <= W = 60 s after created_at. W is fixed at 60 s now.
 * LR as P1 (class_weight balanced, max_iter 300, seed 42); C in {0.25, 1, 4} on validation;
   selection: accuracy, tie macro-F1, tie smaller C (more regularisation).
 * Exploratory only, validation only, not used for selection: W in {0 s, 300 s} sensitivity.
 * On test: report P3 vs P0 and vs P2 (paired), and the subset with no priority/* label at t72h.
 * The /priority <level> command lines in issue bodies were stripped by the verifier on validation
   with no accuracy change; text is therefore left unedited.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp3b_priority as pr  # noqa: E402

ev, t1 = pr.ev, pr.t1
DESIGN_P3 = __doc__.split("PRE-REGISTERED DESIGN P3", 1)[1].strip()
P3_ID, W_SECONDS, W_SENS = "p3_tfidf_creationmeta_lr", 60, (0, 300)


def creation_labels(labels: Any, created: pd.Timestamp, first_labeled: dict[str, pd.Timestamp], w: float) -> list[str]:  # fmt: skip
    """Input labels present at creation: no `labeled` event, or earliest one <= w seconds after created."""
    limit = created + pd.Timedelta(seconds=w)
    return [x for x in pr.input_labels(labels) if x not in first_labeled or first_labeled[x] <= limit]  # fmt: skip


def load_p3(derived: Path, splits_dir: Path, name: str, w: float = W_SECONDS) -> tuple[pd.DataFrame, int]:  # fmt: skip
    """qp3b_priority.load_split with `meta` rebuilt from creation-time labels (one rule, any split)."""
    d, n = pr.load_split(derived, splits_dir, name)
    iss = pd.read_parquet(derived / "issues.parquet").set_index("number").loc[d["number"]]
    ev_ = pd.read_parquet(derived / "label_events.parquet")
    ev_ = ev_[(ev_["action"] == "labeled") & ev_["number"].isin(d["number"])]
    first = ev_.groupby(["number", "label"])["ts"].min().reset_index()
    by_issue = {k: dict(zip(g["label"], g["ts"], strict=True)) for k, g in first.groupby("number")}
    d["meta"] = [
        {
            **{
                f"l:{x}": 1
                for x in creation_labels(r.labels_t72h, r.created_at, by_issue.get(num, {}), w)
            },
            f"a:{r.author_assoc}": 1,
        }  # fmt: skip
        for num, r in zip(d["number"], iss.itertuples(), strict=True)
    ]
    return d, n


def validate_p3(derived: Path, splits_dir: Path, models_dir: Path, report: Path, manifest: Path) -> dict:  # fmt: skip
    """Stage 1: P3 grid on VALIDATION (+ exploratory W sensitivity). Test labels are never loaded."""
    tr, _ = load_p3(derived, splits_dir, "train")
    va, _ = load_p3(derived, splits_dir, "validation")
    yva = ev.to_idx(va["gold3"], pr.CLASSES)
    rows, probas = [], {}
    for c in pr.GRID_C:
        pv = pr.Model(True, c).fit(tr).proba(va)
        rows.append({"model": P3_ID, "C": c, "W_seconds": W_SECONDS, **pr.acc_f1(yva, pv.argmax(axis=1))})  # fmt: skip
        probas[c] = pv
        print(f"{P3_ID} C={c}: acc={rows[-1]['accuracy']:.3f} f1={rows[-1]['macro_f1']:.3f}", file=sys.stderr)  # fmt: skip
    best = min(rows, key=lambda r: (-r["accuracy"], -r["macro_f1"], r["C"]))
    sens = []
    for w in W_SENS:  # exploratory: same pipeline, only the creation-time window changes
        trw, vaw = load_p3(derived, splits_dir, "train", w)[0], load_p3(derived, splits_dir, "validation", w)[0]  # fmt: skip
        sens += [{"W_seconds": w, "C": c, **pr.acc_f1(yva, pr.Model(True, c).fit(trw).proba(vaw).argmax(axis=1))} for c in pr.GRID_C]  # fmt: skip
    out = models_dir / "priority"
    t1.save_proba(out / "validation_p3.parquet", va["number"], probas[best["C"]], pr.CLASSES)
    sel = {"model_id": P3_ID, "C": best["C"], "majority": pr.majority(list(tr["gold3"])), "val_accuracy": best["accuracy"]}  # fmt: skip
    (out / "selection_p3.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
    w = ev.ResultsWriter(report, manifest)
    w.data.update({"p3_design": DESIGN_P3, "p3_validation_table": rows, "p3_selection": sel, "p3_w_sensitivity_exploratory_validation_only": sens})  # fmt: skip
    w.save()
    return sel


def paired_stats(hit_a: np.ndarray, hit_b: np.ndarray, mask: np.ndarray, n_boot: int) -> dict[str, Any]:  # fmt: skip
    """Accuracy of a and b on the masked issues plus the paired a - b bootstrap difference."""
    a, b, n = hit_a[mask], hit_b[mask], int(mask.sum())
    return {"n": n, "a": ev.bootstrap(lambda i: float(a[i].mean()), n, n_boot),
            "b": ev.bootstrap(lambda i: float(b[i].mean()), n, n_boot),
            "difference_paired": ev.paired_diff(lambda i: float(a[i].mean()), lambda i: float(b[i].mean()), n, n_boot)}  # fmt: skip


def final_test_p3(derived: Path, splits_dir: Path, models_dir: Path, report: Path, manifest: Path, n_boot: int = ev.N_BOOT) -> dict:  # fmt: skip
    """Stage 2: refit P3, evaluate it on TEST once; paired vs P0 and vs P2 (kept untouched)."""
    out = models_dir / "priority"
    sel = json.loads((out / "selection_p3.json").read_text(encoding="utf-8"))
    w = ev.ResultsWriter(report, manifest)
    if "test" in w.data["models"].get(P3_ID, {}):  # fail before any test label is read
        raise ev.TestGuardError(f"test already evaluated for {P3_ID} in {Path(report).name}")
    tr, _ = load_p3(derived, splits_dir, "train")
    va, _ = load_p3(derived, splits_dir, "validation")
    m = pr.Model(True, sel["C"]).fit(tr)
    assert np.allclose(m.proba(va), pd.read_parquet(out / "validation_p3.parquet")[pr.CLASSES].to_numpy(), atol=1e-6), "refit drifted"  # fmt: skip
    maj = pr.majority(list(tr["gold3"]))
    assert maj == sel["majority"], "train majority drifted since selection"
    te, _ = load_p3(derived, splits_dir, "test")  # the first and only place P3 reads test labels
    y, pt = ev.to_idx(te["gold3"], pr.CLASSES), m.proba(te)
    pred, pred0 = pt.argmax(axis=1), np.full(len(te), pr.CLASSES.index(maj))
    p2 = pd.read_parquet(out / "test.parquet")
    assert (p2["number"].to_numpy() == te["number"].to_numpy()).all(), (
        "P2 test predictions misaligned"
    )
    pred2 = p2[pr.CLASSES].to_numpy().argmax(axis=1)
    hit, hit0, hit2 = ((p == y).astype(float) for p in (pred, pred0, pred2))
    allm, pre = np.ones(len(te), dtype=bool), ~te["prio_t72h"].to_numpy()
    pf = t1.save_proba(out / "test_p3.parquet", te["number"], pt, pr.CLASSES)
    entry = {"n": len(te), **pr.acc_f1(y, pred), "accuracy_wilson95": list(ev.wilson(int(hit.sum()), len(te))),
             "confusion_matrix_gold_rows_pred_cols": pr.confusion(y, pred), "classes": pr.CLASSES}  # fmt: skip
    w.add(P3_ID, "test", entry, pf, final=True)
    res = {"model_id": P3_ID, "C": sel["C"], "W_seconds": W_SECONDS, "note": "P2 results in the sibling keys are unchanged",  # fmt: skip
           "vs_P0_majority": paired_stats(hit, hit0, allm, n_boot), "vs_P2": paired_stats(hit, hit2, allm, n_boot),  # fmt: skip
           "no_priority_label_at_t72h": {"vs_P0_majority": paired_stats(hit, hit0, pre, n_boot), "vs_P2": paired_stats(hit, hit2, pre, n_boot)}}  # fmt: skip
    res["acceptance_ci_lower_gt_0"] = res["vs_P0_majority"]["difference_paired"]["ci95"][0] > 0
    w.data["test"]["p3"] = res
    w.save()
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3b P3 (creation-time labels), offline")
    ap.add_argument("--final-test", action="store_true", help="evaluate TEST once (explicit)")
    a = ap.parse_args(argv)
    root, rep = Path(r"D:\triageiq-data"), Path("reports")
    args = (root / "derived_v2" / pr.SLUG, root / "splits" / pr.SLUG, root / "models" / pr.SLUG,
            rep / "qp3b_priority_kubernetes__kubernetes_2026-10.json", rep / f"qp0_split_manifest_{pr.SLUG}_2026-10.json")  # fmt: skip
    t1.lower_priority()
    with threadpool_limits(limits=4):
        r = final_test_p3(*args) if a.final_test else validate_p3(*args)
    print(json.dumps(r, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
