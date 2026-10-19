"""Phase 1 candidate 1: TF-IDF + multinomial logistic regression on the kubernetes sig/* task.

PRE-REGISTERED DESIGN (2026-10-20; do not change after seeing test results). Gold per ADR-0067:
the set of sig/* labels at FINAL; primary = earliest `labeled` event among labels still present
(qp0_ceiling_labels.per_issue, imported, not reimplemented); modelled classes = the sig/* labels
carried by >= 30 TRAIN issues + `other` (all remaining sig labels). Population = issues with >= 1
sig/* label at final. Train on train-split issues with classifier_eligible; select on validation
(classifier_eligible); the TEST split's labels are loaded only by the explicit `--final-test`
stage, which evaluates test (classifier_eligible issues) exactly once per report file.
Input text = title + "\n" + body[:4000], lowercased by the vectoriser, nothing else.
Model: TfidfVectorizer(word 1-2 grams, min_df=2, sublinear_tf, max_features=200000) fitted on the
train texts + LogisticRegression (multinomial, lbfgs, max_iter 300, seed 42). Grid on
VALIDATION: C in {1, 4, 16, 64} x class_weight in {None, 'balanced'}; selection = validation
top-3 accuracy, tie-break validation macro-F1, then smaller C. CPU only, threads capped at 4,
BelowNormal priority. Metrics, conformal sets, temperature scaling and CIs: qp1_eval.py.
Probabilities for train / validation / test go to <models-dir>/tfidf_lr/, never the repo.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp0_ceiling_labels as cl  # noqa: E402
import qp1_eval as ev  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
MODEL_ID = "tfidf_lr_v1"
GRID = [(c, cw) for cw in (None, "balanced") for c in (1, 4, 16, 64)]
SEED = 42


def sel_key(row: dict[str, Any]) -> tuple[float, float, float]:
    """Lower is better: validation top-3, then macro-F1, then the smaller C."""
    return (-row["top3"], -row["macro_f1"], row["C"])


def lower_priority() -> None:
    """BelowNormal process priority on Windows (laptop jobs must not starve the desktop)."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def family(label: str) -> bool:
    return label.startswith("sig/")


def load_split(derived: Path, splits_dir: Path, names: tuple[str, ...]) -> pd.DataFrame:
    """Labelled issues of the named splits only; other splits' issues never reach per_issue."""
    cols = ["number", "split", "classifier_eligible"]
    sp = pd.concat([pd.read_parquet(splits_dir / f"{n}.parquet", columns=cols) for n in names])
    issues = pd.read_parquet(derived / "issues.parquet")
    issues = (
        issues[issues["number"].isin(sp["number"])].sort_values("number").reset_index(drop=True)
    )
    events = pd.read_parquet(derived / "label_events.parquet")
    d = cl.per_issue(issues, events[events["number"].isin(issues["number"])], family)
    text = issues["title"].fillna("") + "\n" + issues["body"].fillna("").str.slice(0, 4000)
    d = d.merge(sp, on="number").merge(
        pd.DataFrame({"number": issues["number"], "text": text}), on="number"
    )
    return d[d["final"].map(bool).astype(bool)].reset_index(drop=True)


def define_classes(train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(modelled sig/* labels with >= 30 train issues, class list incl. `other`)."""
    carried = Counter(lbl for s in train["final"] for lbl in s)
    modelled = sorted(k for k, v in carried.items() if v >= 30)
    return modelled, [*modelled, "other"]


def gold(d: pd.DataFrame, modelled: list[str]) -> tuple[list[str], list[frozenset[str]]]:
    m = lambda x: x if x in modelled else "other"  # noqa: E731
    return [m(p) for p in d["primary"]], [frozenset(m(x) for x in s) for s in d["final"]]


def fit_lr(x, y: list[str], c: float, cw: str | None) -> LogisticRegression:  # noqa: ANN001
    return LogisticRegression(C=c, class_weight=cw, max_iter=300, random_state=SEED).fit(x, y)


def proba_full(model: LogisticRegression, x, classes: list[str]) -> np.ndarray:  # noqa: ANN001
    """Probabilities aligned to `classes` (a class unseen in training gets 0)."""
    out = np.zeros((x.shape[0], len(classes)))
    out[:, [classes.index(c) for c in model.classes_]] = model.predict_proba(x)
    return out


def save_proba(path: Path, numbers: pd.Series, p: np.ndarray, classes: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(p, columns=classes).assign(number=numbers.to_numpy())
    df[["number", *classes]].to_parquet(path, index=False)
    return path


def point_metrics(p: np.ndarray, y: list[str], sets: list[frozenset[str]], classes: list[str]) -> dict:  # fmt: skip
    h = ev.hit_arrays(y, sets, p, classes)
    yi = ev.to_idx(y, classes)
    return {
        **{k: float(v.mean()) for k, v in h.items()},
        "macro_f1": ev.macro_f1(yi, p.argmax(axis=1), len(classes)), "nll": ev.nll(p, yi),
    }  # fmt: skip


def prepare(derived: Path, splits_dir: Path):  # noqa: ANN201
    """Train/validation data (eligible only), classes and the vectorised text; no test access."""
    d = load_split(derived, splits_dir, ("train", "validation"))
    modelled, classes = define_classes(d[d["split"] == "train"])
    tr, va = (
        d[(d["split"] == s) & d["classifier_eligible"]].reset_index(drop=True)
        for s in ("train", "validation")
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    return modelled, classes, tr, va, vec, vec.fit_transform(tr["text"]), vec.transform(va["text"])


def select(derived: Path, splits_dir: Path, models_dir: Path) -> dict[str, Any]:
    """Stage 1: grid on validation; test labels are never loaded."""
    modelled, classes, tr, va, vec, xtr, xva = prepare(derived, splits_dir)
    ytr, _ = gold(tr, modelled)
    yva, sva = gold(va, modelled)
    rows, best = [], None
    for c, cw in GRID:
        t0 = time.perf_counter()
        model = fit_lr(xtr, ytr, c, cw)
        secs = time.perf_counter() - t0
        pv = proba_full(model, xva, classes)
        row = {"C": c, "class_weight": cw, "fit_seconds": secs, **point_metrics(pv, yva, sva, classes)}  # fmt: skip
        rows.append(row)
        key = sel_key(row)
        if best is None or key < best[0]:
            best = (key, row, model, pv)
        print(f"C={c} cw={cw}: top3={row['top3']:.3f} f1={row['macro_f1']:.3f} {secs:.0f}s", file=sys.stderr)  # fmt: skip
    _, chosen, model, pv = best
    out = models_dir / "tfidf_lr"
    save_proba(out / "validation.parquet", va["number"], pv, classes)
    save_proba(out / "train.parquet", tr["number"], proba_full(model, xtr, classes), classes)
    info = {"grid": rows, "selected": {"C": chosen["C"], "class_weight": chosen["class_weight"]},
            "modelled_classes": modelled, "classes": classes, "n_train": len(tr), "n_validation": len(va)}  # fmt: skip
    (out / "grid.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def final_test(
    derived, splits_dir, models_dir, report, manifest, ceiling, n_boot=ev.N_BOOT
) -> dict:  # noqa: ANN001
    """Stage 2: refit the selected config, evaluate TEST once, write the report."""
    out = models_dir / "tfidf_lr"
    info = json.loads((out / "grid.json").read_text(encoding="utf-8"))
    writer = ev.ResultsWriter(report, manifest)
    if "test" in writer.data["models"].get(MODEL_ID, {}):  # fail before touching any test label
        raise ev.TestGuardError(f"test already evaluated for {MODEL_ID} in {Path(report).name}")
    modelled, classes, tr, va, vec, xtr, xva = prepare(derived, splits_dir)
    assert classes == info["classes"], "class definition drifted since model selection"
    sel = info["selected"]
    t0 = time.perf_counter()
    model = fit_lr(xtr, gold(tr, modelled)[0], sel["C"], sel["class_weight"])
    train_s = time.perf_counter() - t0
    pv = proba_full(model, xva, classes)
    saved = pd.read_parquet(out / "validation.parquet")[classes].to_numpy()
    assert np.allclose(pv, saved, atol=1e-6), "refit differs from stage-1 validation probabilities"
    yva, sva = gold(va, modelled)
    te = load_split(derived, splits_dir, ("test",))  # the first and only place test labels are read
    te = te[te["classifier_eligible"]].reset_index(drop=True)
    yte, ste = gold(te, modelled)
    t1 = time.perf_counter()
    pt = proba_full(model, vec.transform(te["text"]), classes)
    infer_ms = (time.perf_counter() - t1) * 1000 / len(te)
    pred_file = save_proba(out / "test.parquet", te["number"], pt, classes)
    ytei, yvai = ev.to_idx(yte, classes), ev.to_idx(yva, classes)
    temp = ev.fit_temperature(pv, yvai)
    pvt, ptt = ev.apply_temperature(pv, temp), ev.apply_temperature(pt, temp)
    qraw, qtemp = (ev.conformal_qhat(ev.aps_scores(p, yvai)) for p in (pv, pvt))
    f1 = ev.f1_per_class(ytei, pt.argmax(axis=1), len(classes))
    ceil = json.loads(Path(ceiling).read_text(encoding="utf-8"))
    test = ev.evaluate("test", True, yte, ste, pt, classes, n_boot)
    test |= {
        "calibration": {"temperature": temp, "ece_before": ev.ece(pt, ytei), "ece_after": ev.ece(ptt, ytei),
                        "nll_after": ev.nll(ptt, ytei), "validation_ece_before": ev.ece(pv, yvai),
                        "validation_ece_after": ev.ece(pvt, yvai)},
        "conformal_alpha_0.05_raw_proba": ev.conformal_report(pt, yte, ste, classes, qraw),
        "conformal_alpha_0.05_temperature_scaled": ev.conformal_report(ptt, yte, ste, classes, qtemp),
        "per_class_f1": {c: (None if np.isnan(v) else float(v)) for c, v in zip(classes, f1, strict=True)},
        "per_class_support": {c: int((np.array(yte) == c).sum()) for c in classes},
        "timing": {"train_seconds": train_s, "inference_ms_per_issue": infer_ms},
        "n_test_eligible_labelled": len(te),
        "context_baselines_from_ceiling_report": {
            "majority_top1_test_all_labelled": ceil["baselines"]["test"]["majority_top1"],
            "majority_any_valid_test": ceil["baselines"]["test"]["majority_any_valid"],
            "ceiling_proxies": ceil["ceiling_proxies"],
        },
    }  # fmt: skip
    val = ev.evaluate("validation", False, yva, sva, pv, classes, n_boot)
    writer.add(MODEL_ID, "validation", val, out / "validation.parquet")
    writer.add(MODEL_ID, "test", test, pred_file, final=True)
    writer.data.update({"design": DESIGN, "harness_design": ev.DESIGN, "validation_grid": info["grid"],
                        "selected_config": sel, "classes": classes, "repo": "kubernetes/kubernetes"})  # fmt: skip
    writer.save()
    return test


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TF-IDF + LR candidate for the sig/* task")
    ap.add_argument("--repo", default="kubernetes/kubernetes")
    ap.add_argument("--derived-dir", type=Path)
    ap.add_argument("--splits-dir", type=Path)
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--final-test", action="store_true", help="evaluate TEST once (explicit)")
    a = ap.parse_args(argv)
    slug, root = a.repo.replace("/", "__"), Path(r"D:\triageiq-data")
    derived = a.derived_dir or root / "derived" / slug
    splits_dir = a.splits_dir or root / "splits" / slug
    models_dir = a.models_dir or root / "models" / slug
    reports = Path("reports")
    report = a.report or reports / f"qp1_tfidf_lr_{slug}_2026-10.json"
    lower_priority()
    with threadpool_limits(limits=4):
        if a.final_test:
            r = final_test(derived, splits_dir, models_dir, report, reports / f"qp0_split_manifest_{slug}_2026-10.json",
                           reports / f"qp0_ceiling_labels_{slug}_2026-10.json")  # fmt: skip
            print({k: round(r[k]["point"], 4) for k in ("top1", "top3", "anyvalid_top1", "macro_f1", "ece")})  # fmt: skip
        else:
            info = select(derived, splits_dir, models_dir)
            print(json.dumps(info["selected"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
