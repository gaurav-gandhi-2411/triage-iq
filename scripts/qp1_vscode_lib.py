"""Reusable pieces of the vscode Phase 1 runner (qp1_vscode_phase1.py): taxonomy gold mapping,
strict / lenient / in-family scoring, abstention threshold, V2 views, model selection key and the
conformal selection loop. Pure functions; the design is pre-registered in qp1_vscode_phase1.py.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp0_ceiling_labels as cl  # noqa: E402
import qp1_conformal as qc  # noqa: E402
import qp1_eval as ev  # noqa: E402
import qp2_vscode_taxonomy as tax  # noqa: E402


    IDS,
    OTHER,
    classes_sha256,
    conformal_select,
    define_classes,
    fit_grid,
    gold,
    hit_arrays,
    labelled_eligible,
    load_split,
    majority_proba,
    paired,
    production_proba,
    prop,
    rank_proba,
    save_pred,

)

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
REPO = "microsoft/vscode"
SLUG = "microsoft__vscode"
ADR_PATH = Path("docs/architecture/adr/0071-vscode-conformal-method.md")


# ------------------------------------------------------------------------------------- stages


def report_paths(a: argparse.Namespace) -> tuple[Path, Path]:
    reports = Path("reports")
    return (
        a.report or reports / "qp1_vscode_phase1_2026-10.json",
        reports / f"qp0_split_manifest_{SLUG}_2026-10.json",
    )


def prepare_train(derived: Path, splits_dir: Path):  # noqa: ANN201
    """Train data, class list (asserted), vectoriser fitted on eligible labelled train text."""
    train_all = load_split(derived, splits_dir, "train")
    classes = define_classes(train_all)
    tr = labelled_eligible(train_all)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    return classes, tr, vec, vec.fit_transform(tr["text"])


def stage_validation(a: argparse.Namespace, derived, splits_dir, models_dir) -> dict[str, Any]:  # noqa: ANN001
    """Fit V0 / V1 / V2, score production, select on validation. Test labels never loaded."""
    out = models_dir / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    classes, tr, vec, xtr = prepare_train(derived, splits_dir)
    va = labelled_eligible(load_split(derived, splits_dir, "validation"))
    ytr, _ = gold(tr, classes)
    yva, sva = gold(va, classes)
    xva = vec.transform(va["text"])
    report, manifest = report_paths(a)
    writer = ev.ResultsWriter(report, manifest)
    scores: dict[str, dict[str, Any]] = {}

    pv0, prior = majority_proba(ytr, classes, len(va))
    p0 = out / "majority_prior.parquet"
    save_pred(p0, pd.Series([0]), prior[None, :], classes)
    scores[IDS["v0"]] = score(
        "validation", False, yva, sva, rank_proba(pv0, len(classes) - 1), classes
    )
    writer.add(IDS["v0"], "validation", scores[IDS["v0"]], p0)

    grids = {}
    grids["v1"] = fit_grid("v1", xtr, ytr, xva, yva, sva, classes, out, None)
    is_oth = np.array([c == OTHER for c in yva])
    keep = np.array([c != OTHER for c in ytr])
    grids["v2"] = fit_grid(
        "v2", xtr[keep], [c for c in ytr if c != OTHER], xva, yva, sva, classes, out,
        lambda p: best_abstention_tau(p.max(axis=1), is_oth),
    )  # fmt: skip
    for tag in ("v1", "v2"):
        best = joblib.load(out / f"best_{tag}.joblib")
        pv = t1.proba_full(best["model"], xva, classes)
        rk = (
            v2_views(pv, best["tau"], classes)[0]
            if tag == "v2"
            else rank_proba(pv, len(classes) - 1)
        )
        pf = save_pred(out / f"validation_{tag}.parquet", va["number"], pv, classes)
        scores[IDS[tag]] = score("validation", False, yva, sva, rk, classes)
        writer.add(IDS[tag], "validation", scores[IDS[tag]], pf)
    pprod = production_proba(va, classes, Path("data") / "models")
    pf = save_pred(out / "validation_prod.parquet", va["number"], pprod, classes)
    scores[IDS["prod"]] = score(
        "validation", False, yva, sva, rank_proba(pprod, len(classes) - 1), classes
    )
    writer.add(IDS["prod"], "validation", scores[IDS["prod"]], pf)

    cand = {k: v for k, v in scores.items() if k in COMPLEXITY}
    selected = min(cand, key=lambda k: sel_key(cand[k], k))
    writer.data.update({
        "design": DESIGN, "harness_design": ev.DESIGN, "repo": REPO, "classes": classes,
        "classes_sha256_excluding_other": classes_sha256(classes[:-1]), "adr_0070_sha256": FROZEN_SHA,
        "validation_grids": grids, "selected_model": selected,
        "populations": {"train_labelled_eligible": len(tr), "validation_labelled_eligible": len(va)},
        "not_run": "embedding / fine-tuned candidates blocked by Kaggle GPU quota",
    })  # fmt: skip
    writer.save()
    return {
        "selected": selected,
        **{k: round(v["strict_top3"]["point"], 4) for k, v in scores.items()},
    }


def supersede_alpha_0_1(writer: ev.ResultsWriter) -> None:
    """Move the alpha-0.1 conformal results (brief error, ADR-0071 amendment) under one key."""
    old = writer.data.get("superseded_alpha_0_1", {})
    for k in ("conformal_selection", "conformal_test"):
        if k in writer.data and writer.data[k].get("alpha") == 0.1:
            old[k] = writer.data.pop(k)
    if old:
        writer.data["superseded_alpha_0_1"] = old


def stage_conformal(a: argparse.Namespace, derived, splits_dir, models_dir) -> dict[str, Any]:  # noqa: ANN001
    """Validation-only conformal selection for the selected model; writes into the report."""
    report, manifest = report_paths(a)
    writer = ev.ResultsWriter(report, manifest)
    selected, classes = writer.data["selected_model"], writer.data["classes"]
    tag = {IDS["v1"]: "v1", IDS["v2"]: "v2"}[selected]  # V0 has no per-issue scores to calibrate
    va = labelled_eligible(load_split(derived, splits_dir, "validation"))
    d = pd.read_parquet(models_dir / "phase1" / f"validation_{tag}.parquet")
    assert (d["number"].to_numpy() == va["number"].to_numpy()).all()
    p, y = d[classes].to_numpy(), ev.to_idx(gold(va, classes)[0], classes)
    keep = np.ones(len(y), bool) if tag == "v1" else y != classes.index(OTHER)
    supersede_alpha_0_1(writer)
    res = conformal_select(p[keep], y[keep])
    res["population"] = "all labelled eligible" if keep.all() else "in-family only (V2)"
    res["n"] = int(keep.sum())
    writer.data["conformal_selection"] = res
    writer.save()
    return res


def stage_test(a: argparse.Namespace, derived, splits_dir, models_dir) -> dict[str, Any]:  # noqa: ANN001
    """Evaluate TEST once for the selected model, production and V0; write the report."""
    out = models_dir / "phase1"
    report, manifest = report_paths(a)
    writer = ev.ResultsWriter(report, manifest)
    sel_id = writer.data["selected_model"]
    ids = [sel_id, IDS["prod"], IDS["v0"]] if sel_id != IDS["v0"] else [sel_id, IDS["prod"]]
    for m in ids:  # fail before touching any test label
        if "test" in writer.data["models"].get(m, {}):
            raise ev.TestGuardError(f"test already evaluated for {m} in {report.name}")
    classes = writer.data["classes"]
    other = classes.index(OTHER)
    classes_now, tr, vec, _ = prepare_train(derived, splits_dir)
    assert classes_now == classes, "class definition drifted since model selection"
    ytr, _ = gold(tr, classes)
    va = labelled_eligible(load_split(derived, splits_dir, "validation"))
    yva, _ = gold(va, classes)
    if sel_id != IDS["v0"]:  # the refit vectoriser must reproduce the saved validation scores
        m = joblib.load(out / f"best_{'v2' if sel_id == IDS['v2'] else 'v1'}.joblib")["model"]
        saved = pd.read_parquet(out / f"validation_{'v2' if sel_id == IDS['v2'] else 'v1'}.parquet")
        assert np.allclose(t1.proba_full(m, vec.transform(va["text"]), classes), saved[classes].to_numpy(), atol=1e-6)  # fmt: skip
    te_all = load_split(derived, splits_dir, "test")  # first and only read of test labels
    te = labelled_eligible(te_all)
    yte, ste = gold(te, classes)
    n_boot, tag = ev.N_BOOT, "v2" if sel_id == IDS["v2"] else "v1"
    views: dict[str, Ranked] = {}
    preds: dict[str, np.ndarray] = {}
    lat: dict[str, Any] = {}
    # selected model (or V0)
    if sel_id == IDS["v0"]:
        preds[sel_id], _ = majority_proba(ytr, classes, len(te))
        views[sel_id] = rank_proba(preds[sel_id], other)
    else:
        best = joblib.load(out / f"best_{tag}.joblib")
        xte = vec.transform(te["text"])
            "batch_ms_per_issue_excl_vectorise": (time.perf_counter() - t0) * 1000 / len(te)
        }
        views[sel_id] = (
            v2_views(preds[sel_id], best["tau"], classes)[0]
            if tag == "v2"
            else rank_proba(preds[sel_id], other)
        )
        one = []
        for t in te["text"].iloc[:200]:
            t0 = time.perf_counter()
            best["model"].predict_proba(vec.transform([t]))
            one.append((time.perf_counter() - t0) * 1000)
        lat[sel_id] |= {"single_issue_ms_p50": float(np.percentile(one, 50)), "single_issue_ms_p95": float(np.percentile(one, 95))}  # fmt: skip
    from triage_iq.models.component_classifier import _build_text, load_classifier

    t0 = time.perf_counter()
    preds[IDS["prod"]] = production_proba(te, classes, Path("data") / "models")
    lat[IDS["prod"]] = {"batch_ms_per_issue_incl_load_and_vectorise": (time.perf_counter() - t0) * 1000 / len(te)}  # fmt: skip
    clf, one = load_classifier(Path("data") / "models", "microsoft_vscode"), []
    for _, r in te.iloc[:200].iterrows():
        t0 = time.perf_counter()
    writer.data["latency_cpu"] = lat
    ceil = json.loads(
        Path("reports/qp0_ceiling_labels_microsoft__vscode_2026-10.json").read_text(
            encoding="utf-8"
        )
    )
    mask = qc.predict_mask(pt, q, spec, qc.uniforms(len(y), 1))
    fam = y != other
    res = {
        "exploratory": "ADR-0069: unconfirmed until the post-test confirmation set",
        "alpha": CONF_ALPHA, "spec": spec, "qhat": q, "model": sel_id,
        "all_labelled": qc.mask_report(mask, y, sm),
        "in_family_only": qc.mask_report(mask[fam], y[fam], sm[fam]),
    }  # fmt: skip
    supersede_alpha_0_1(writer)
    writer.data["conformal_test"] = res
    writer.save()
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vscode Phase 1: CPU component candidates")
    ap.add_argument(
        "--stage", choices=["validation", "conformal", "test", "conformal-test"], required=True
    )
    ap.add_argument("--derived-dir", type=Path)
    ap.add_argument("--splits-dir", type=Path)
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args(argv)
    root = Path(r"D:\triageiq-data")
    derived = a.derived_dir or root / "derived_v2" / SLUG
    splits_dir = a.splits_dir or root / "splits" / SLUG
    models_dir = a.models_dir or root / "models" / SLUG
    t1.lower_priority()
    with threadpool_limits(limits=4):
        if a.stage == "validation":
            r = stage_validation(a, derived, splits_dir, models_dir)
        elif a.stage == "conformal-test":
            r = stage_conformal_test(a, derived, splits_dir, models_dir)
        elif a.stage == "conformal":
            r = stage_conformal(a, derived, splits_dir, models_dir)
        else:
            r = stage_test(a, derived, splits_dir, models_dir)
    print(json.dumps(r, indent=1, default=str))
