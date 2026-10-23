"""Phase 3b offline study: priority model trained on real k8s priority/* labels (spec section 8).

PRE-REGISTERED DESIGN (2026-10-23; committed with validation results BEFORE any test evaluation;
do not change after seeing test results). OFFLINE STUDY ONLY: no serving, prompt or flag change.
 * Population: kubernetes issues of each split (train / validation / test, all issues of the split,
   no eligibility filter) whose FINAL labels (labels_final) carry >= 1 of priority/critical-urgent,
   priority/important-soon, priority/important-longterm, priority/backlog. priority/awaiting-more-
   evidence alone is excluded (not a level). Several levels present -> gold = the most urgent
   (critical-urgent > important-soon > important-longterm > backlog). Population counts per split
   and the share of all split issues are reported. LIMITATION: the model is evaluated only on issues
   maintainers prioritised (selection effect); it says nothing about issues nobody prioritised.
 * Served schema gold (priority_guess): critical-urgent, important-soon -> high; important-longterm
   -> medium; backlog -> low. PRIMARY metric = accuracy on this 3-level gold. Secondary (reported,
   not claimed): macro-F1, the 4-gold-level x 3-predicted-level table (a 3-level model has no
   4-level accuracy; training a second model would breach "test once per candidate"), confusion
   matrix.
 * Baseline P0 = majority class of the TRAIN population (3-level). ACCEPTANCE: test accuracy of the
   selected model minus P0 accuracy, paired bootstrap 95% CI (2000 draws, seed 42, same resampled
   issues), lower bound > 0.
 * Inputs (creation-time only). Phase 1 used text only (title + "\\n" + body[:4000]) because sig/*
   was its target; here the target is priority, so labels_t72h MINUS every priority/* label may
   enter, restricted to kind/* and sig/* (lifecycle/, triage/, needs-* etc. are never used), plus
   author_assoc. Priority labels never reach any input; text is not edited.
 * Candidates (model ids; every one fitted on train, scored on VALIDATION; seed 42):
   P0 majority. P1 `p1_tfidf_lr` = qp1_tfidf_features.Featurizer(title_sep on, paths off) (word 1-2
   grams of title [50k] and of body [200k], char_wb 3-5 grams of title [100k]; min_df=2, sublinear)
   + multinomial LogisticRegression, class_weight balanced, max_iter 300; C in {0.25, 1, 4} chosen on
   validation. P2 `p2_tfidf_meta_lr` = P1 features + one-hot t72h kind/* and sig/* labels + one-hot
   author_assoc, same LR grid. Selection on VALIDATION accuracy; ties -> macro-F1 -> simpler (P1
   before P2, then smaller C). Only the selected candidate and P0 are evaluated on test, once each
   per report file (qp1_eval.ResultsWriter guard); test labels are read only by --final-test.
 * Also on test (secondary): selected model vs majority restricted to issues with NO priority/*
   label at t72h (the model predicting before a human prioritised), with paired CI.
 * Wilson 95% for proportions, bootstrap/paired CIs from qp1_eval (2000 draws, seed 42).
Models and probabilities go to D:\\triageiq-data\\models\\<slug>\\priority\\, never the repo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_eval as ev  # noqa: E402
import qp1_tfidf_features as tf  # noqa: E402
import qp1_tfidf_lr as t1  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
LEVELS = ["priority/critical-urgent", "priority/important-soon", "priority/important-longterm", "priority/backlog"]  # fmt: skip
TO3 = dict(zip(LEVELS, ["high", "high", "medium", "low"], strict=True))
CLASSES = ["low", "medium", "high"]  # fixed order: also the majority tie-break order
GRID_C = (0.25, 1, 4)
P0_ID, P1_ID, P2_ID = "p0_majority", "p1_tfidf_lr", "p2_tfidf_meta_lr"
SLUG = "kubernetes__kubernetes"


def gold4(labels: Any) -> str | None:
    """Most urgent priority level among the labels; None when no level (awaiting-more-evidence only)."""
    return next((lv for lv in LEVELS if lv in set(labels)), None)


def gold3(labels: Any) -> str | None:
    g = gold4(labels)
    return None if g is None else TO3[g]


def input_labels(labels: Any) -> list[str]:
    """t72h labels allowed as model input: kind/* and sig/*; priority/* can never pass."""
    return sorted(x for x in labels if x.startswith(("kind/", "sig/")))


def meta_dict(labels: Any, assoc: Any) -> dict[str, int]:
    return {**{f"l:{x}": 1 for x in input_labels(labels)}, f"a:{assoc}": 1}


def majority(train_gold: list[str]) -> str:
    """Most frequent class of the TRAIN population; ties resolved by CLASSES order."""
    c = Counter(train_gold)
    return max(CLASSES, key=lambda k: (c[k], -CLASSES.index(k)))


def load_split(derived: Path, splits_dir: Path, name: str) -> tuple[pd.DataFrame, int]:
    """(priority population of one split, number of issues in the split). Other splits never read."""
    sp = pd.read_parquet(splits_dir / f"{name}.parquet", columns=["number"])
    iss = pd.read_parquet(derived / "issues.parquet")
    iss = iss[iss["number"].isin(sp["number"])].sort_values("number").reset_index(drop=True)
    d = pd.DataFrame({"number": iss["number"], "split": name})
    d["text"] = iss["title"].fillna("") + "\n" + iss["body"].fillna("").str.slice(0, 4000)
    d["gold4"], d["gold3"] = iss["labels_final"].map(gold4), iss["labels_final"].map(gold3)
    d["meta"] = [meta_dict(a, b) for a, b in zip(iss["labels_t72h"], iss["author_assoc"], strict=True)]  # fmt: skip
    d["prio_t72h"] = iss["labels_t72h"].map(lambda ls: any(x.startswith("priority/") for x in ls))
    return d[d["gold3"].notna()].reset_index(drop=True), len(iss)


class Model:
    """Fitted featuriser (+ optional meta block) and LR for one candidate."""

    def __init__(self, with_meta: bool, c: float) -> None:
        self.with_meta, self.c = with_meta, c
        self.feat, self.dv = tf.Featurizer(True, False), DictVectorizer()

    def x(self, d: pd.DataFrame, fit: bool = False) -> csr_matrix:
        f = self.feat.fit_transform if fit else self.feat.transform
        blocks = [f(d["text"])]
        if self.with_meta:
            m = self.dv.fit_transform if fit else self.dv.transform
            blocks.append(m(list(d["meta"])))
        return hstack(blocks).tocsr()

    def fit(self, tr: pd.DataFrame) -> Model:
        self.lr = t1.fit_lr(self.x(tr, fit=True), list(tr["gold3"]), self.c, "balanced")
        return self

    def proba(self, d: pd.DataFrame) -> np.ndarray:
        return t1.proba_full(self.lr, self.x(d), CLASSES)


def sel_key(r: dict[str, Any]) -> tuple:
    """Lower is better: accuracy, macro-F1, simpler candidate (P1 < P2), smaller C."""
    return (-r["accuracy"], -r["macro_f1"], r["with_meta"], r["C"])


def acc_f1(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float((y == pred).mean()), "macro_f1": ev.macro_f1(y, pred, len(CLASSES))}


def confusion(y: np.ndarray, pred: np.ndarray) -> list[list[int]]:
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    np.add.at(m, (y, pred), 1)
    return m.tolist()


def population_stats(d: pd.DataFrame, n_split: int) -> dict[str, Any]:
    return {"n_split_issues": n_split, "n_population": len(d), "share_of_split": len(d) / n_split,
            "dist_3level": {k: int((d["gold3"] == k).sum()) for k in CLASSES},
            "dist_4level": {k: int((d["gold4"] == k).sum()) for k in LEVELS},
            "n_priority_label_at_t72h": int(d["prio_t72h"].sum())}  # fmt: skip


def provenance(derived: Path, splits_dir: Path, names: tuple[str, ...]) -> dict[str, Any]:
    files = [derived / "issues.parquet", derived / "label_events.parquet"]
    files += [splits_dir / f"{n}.parquet" for n in names]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain", "scripts/qp3b_priority.py"], capture_output=True, text=True, check=False)  # fmt: skip
    return {"data_sha256": {f.name if f.parent == derived else f"splits/{f.name}": ev.file_sha256(f) for f in files},  # fmt: skip
            "git_sha": sha.stdout.strip(), "script_dirty": bool(dirty.stdout.strip())}  # fmt: skip


def validate(derived: Path, splits_dir: Path, models_dir: Path, report: Path, manifest: Path) -> dict:  # fmt: skip
    """Stage 1: fit the grid on train, score on VALIDATION, select. Test labels are never loaded."""
    (tr, ntr), (va, nva) = (load_split(derived, splits_dir, n) for n in ("train", "validation"))
    yva = ev.to_idx(va["gold3"], CLASSES)
    maj = majority(list(tr["gold3"]))
    rows = [{"model": P0_ID, "with_meta": False, "C": None, **acc_f1(yva, np.full(len(yva), CLASSES.index(maj)))}]  # fmt: skip
    rows[0]["fit_seconds"] = 0.0
    cand, probas = [], {}
    for with_meta in (False, True):
        for c in GRID_C:
            t0 = time.perf_counter()
            m = Model(with_meta, c).fit(tr)
            pv = m.proba(va)
            row = {"model": P2_ID if with_meta else P1_ID, "with_meta": with_meta, "C": c,
                   "fit_seconds": time.perf_counter() - t0, **acc_f1(yva, pv.argmax(axis=1))}  # fmt: skip
            cand.append(row)
            probas[(with_meta, c)] = pv
            print(f"{row['model']} C={c}: acc={row['accuracy']:.3f} f1={row['macro_f1']:.3f}", file=sys.stderr)  # fmt: skip
    best = min(cand, key=sel_key)
    out = models_dir / "priority"
    out.mkdir(parents=True, exist_ok=True)
    t1.save_proba(out / "validation.parquet", va["number"], probas[(best["with_meta"], best["C"])], CLASSES)  # fmt: skip
    sel = {"model_id": best["model"], "with_meta": best["with_meta"], "C": best["C"], "majority": maj,
           "val_accuracy": best["accuracy"]}  # fmt: skip
    (out / "selection.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
    w = ev.ResultsWriter(report, manifest)
    w.data.update({"design": DESIGN, "repo": "kubernetes/kubernetes", "classes": CLASSES,
                   "population": {"train": population_stats(tr, ntr), "validation": population_stats(va, nva)},
                   "validation_table": [*rows, *cand], "selection": sel,
                   "validation_majority_accuracy": rows[0]["accuracy"],
                   **provenance(derived, splits_dir, ("train", "validation", "test"))})  # fmt: skip
    w.save()
    return sel


def final_test(derived: Path, splits_dir: Path, models_dir: Path, report: Path, manifest: Path, n_boot: int = ev.N_BOOT) -> dict:  # fmt: skip
    """Stage 2: refit the selected candidate, evaluate it and P0 on TEST once each."""
    out = models_dir / "priority"
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    w = ev.ResultsWriter(report, manifest)
    for mid in (P0_ID, sel["model_id"]):  # fail before any test label is read
        if "test" in w.data["models"].get(mid, {}):
            raise ev.TestGuardError(f"test already evaluated for {mid} in {Path(report).name}")
    (tr, _), (va, _) = (load_split(derived, splits_dir, n) for n in ("train", "validation"))
    m = Model(sel["with_meta"], sel["C"]).fit(tr)
    saved = pd.read_parquet(out / "validation.parquet")[CLASSES].to_numpy()
    assert np.allclose(m.proba(va), saved, atol=1e-6), "refit differs from stage-1 probabilities"
    maj = majority(list(tr["gold3"]))
    assert maj == sel["majority"], "train majority drifted since selection"
    te, nte = load_split(
        derived, splits_dir, "test"
    )  # the first and only place test labels are read
    y, pt = ev.to_idx(te["gold3"], CLASSES), m.proba(te)
    pred, pred0 = pt.argmax(axis=1), np.full(len(te), CLASSES.index(maj))
    n = len(te)
    hit, hit0 = (pred == y).astype(float), (pred0 == y).astype(float)

    def paired(mask: np.ndarray) -> dict[str, Any]:
        idx = np.flatnonzero(mask)
        return {"model": ev.bootstrap(lambda i: float(hit[idx][i].mean()), len(idx), n_boot),
                "majority": ev.bootstrap(lambda i: float(hit0[idx][i].mean()), len(idx), n_boot),
                "difference_paired": ev.paired_diff(lambda i: float(hit[idx][i].mean()), lambda i: float(hit0[idx][i].mean()), len(idx), n_boot),  # fmt: skip
                "model_wilson95": list(ev.wilson(int(hit[idx].sum()), len(idx)))}  # fmt: skip

    def entry(h: np.ndarray, p: np.ndarray) -> dict[str, Any]:
        return {"n": n, **acc_f1(y, p), "accuracy_wilson95": list(ev.wilson(int(h.sum()), n)),
                "confusion_matrix_gold_rows_pred_cols": confusion(y, p), "classes": CLASSES}  # fmt: skip

    pf0, pf = (
        out / "test_p0.parquet",
        t1.save_proba(out / "test.parquet", te["number"], pt, CLASSES),
    )
    pd.DataFrame({"number": te["number"], "pred": [CLASSES[i] for i in pred0]}).to_parquet(pf0, index=False)  # fmt: skip
    four = {g: {k: int(((te["gold4"] == g) & (pred == i)).sum()) for i, k in enumerate(CLASSES)} for g in LEVELS}  # fmt: skip
    full = paired(np.ones(n, dtype=bool))
    pre = paired(~te["prio_t72h"].to_numpy())
    w.add(P0_ID, "test", entry(hit0, pred0) | {"majority_class": maj}, pf0, final=True)
    w.add(sel["model_id"], "test", entry(hit, pred) | {"gold4_by_predicted_3level": four}, pf, final=True)  # fmt: skip
    ci = full["difference_paired"]["ci95"]
    w.data["test"] = {"population": population_stats(te, nte), "primary": full,
                      "acceptance_ci_lower_gt_0": ci[0] > 0, "no_priority_label_at_t72h": {"n": int((~te["prio_t72h"]).sum()), **pre},  # fmt: skip
                      "model_id": sel["model_id"]}  # fmt: skip
    w.save()
    return w.data["test"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3b priority study (k8s, offline)")
    ap.add_argument("--final-test", action="store_true", help="evaluate TEST once (explicit)")
    a = ap.parse_args(argv)
    root, rep = Path(r"D:\triageiq-data"), Path("reports")
    args = (root / "derived_v2" / SLUG, root / "splits" / SLUG, root / "models" / SLUG,
            rep / "qp3b_priority_kubernetes__kubernetes_2026-10.json", rep / f"qp0_split_manifest_{SLUG}_2026-10.json")  # fmt: skip
    t1.lower_priority()
    with threadpool_limits(limits=4):
        r = final_test(*args) if a.final_test else validate(*args)
    print(json.dumps(r, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
