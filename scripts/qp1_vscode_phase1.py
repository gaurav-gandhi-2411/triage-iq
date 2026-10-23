"""Phase 1 for vscode (spec-autonomous-run.md Q4): CPU component-classifier candidates on the ADR-0070 taxonomy.

PRE-REGISTERED DESIGN (2026-10-23; do not change after seeing test results). Reuses the k8s
Phase 1 pieces unchanged (qp1_tfidf_lr fit_lr / proba_full / GRID / vectoriser settings,
qp1_eval bootstrap / Wilson / ResultsWriter test-once guard, qp1_conformal score functions); no
k8s file is modified.
 * Taxonomy (ADR-0070): population = issues with >= 1 non-excluded label at final; primary =
   qp0_ceiling_labels.per_issue with family = qp2_vscode_taxonomy.not_excluded; gold class =
   parent_of(primary) if in the 85-class list else `other` (last class). The class list is
   regenerated from the TRAIN split only (parents carried by >= 30 train issues, labels_final) and
   its sha256 is asserted equal to ADR-0070's. Train on train-split classifier_eligible labelled
   issues; validation / test the same filter. Text = title + "\\n" + body[:4000] (as k8s).
 * Candidates (model ids): V0 majority_prior_v0 = train class frequencies as probabilities.
   V1 tfidf_lr_v1 = k8s candidate 1 exactly (TF-IDF word 1-2 grams, min_df 2, sublinear, 200k;
   multinomial LR; 86 classes incl. `other`; grid C {1,4,16,64} x class_weight {None,balanced},
   best on validation by the key below, then smaller C). V2 tfidf_lr_family_only_v2 = same
   features / grid, trained on issues whose gold is NOT `other` (85 classes, `other` column = 0);
   it abstains (top-1 becomes `other`) when its max probability < tau, tau = the value on the
   validation max-probability grid (200 quantiles) maximising abstention F1 against gold `other`
   (ties: higher precision). Production baseline production_vscode_ovr = the served
   data/models classifier (calibrated sigmoid scores, load_classifier), each class mapped through
   parent_of -> 85 classes else `other`, score of a parent = max over its source classes; a fixed
   model, scored on validation and test without selection. Embedding / fine-tuned candidates are
   NOT run (GPU quota).
 * Scoring. HEADLINE (strict): top-k hit iff gold != other and gold in the model's top-k ranked
   classes; a predicted `other` never earns a hit; V2's strict ranking is its family ranking
   (abstention is reported next to it, not applied to it). k8s-convention variant (lenient):
   `other` is an ordinary class, a gold-other issue is a hit when `other` is predicted (top-1) or
   in the top-3; V2's lenient ranking puts `other` first when it abstains. Also: in-family strict
   top-1/top-3 (issues with gold != other), strict any-valid top-1/top-3 (true set minus `other`),
   macro-F1 over classes present in y_true on the lenient top-1, abstention rate / precision /
   recall (abstain = lenient top-1 is `other`; precision vs gold other; recall of gold other),
   the empirical coverage bound (share of gold != other), Wilson 95% for proportions, 2000-sample
   percentile bootstrap (seed 42), paired differences on shared resamples.
 * Selection (validation): among V0 / V1(best row) / V2(best row), the highest strict top-3, tie
   macro-F1, tie simpler (V0 < V1 < V2). Production is never a selection candidate.
 * Conformal (stage `conformal`, VALIDATION ONLY, exploratory per ADR-0069): alpha 0.1; candidate
   scores aps_nonrandomised, aps_randomised, raps (lambda 0.1, k_reg 1, as k8s), lac (qp1_conformal
   definitions); 50 seeded random 50/50 splits of the labelled eligible validation set (calibrate on
   one half, evaluate coverage w.r.t. the primary label, `other` included, and set size on the
   other); choose by: mean coverage within 0.90 +/- 0.02, then smallest mean set size, then name;
   if none qualifies, the coverage closest to 0.90. For V2 the population is in-family issues
   only (its `other` column is 0, so a gold-other issue cannot be covered). The choice is frozen in
   ADR-0071 before the test stage, which refuses to run unless that ADR names the chosen score.
 * Test (stage `test`, explicit): test labels are read only there; selected model, production and
   V0 are each evaluated once per report file (ResultsWriter guard); paired bootstrap of the
   selected model vs production and vs V0 (strict top-1, strict top-3, in-family top-3, lenient
   top-3); conformal calibrated on FULL validation applied to test; CPU latency (batch ms/issue
   and single-issue p50 / p95 over the first 200 test issues). Probabilities and model files go to
   D:\\triageiq-data\\models\\microsoft__vscode\\phase1\\, never the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # triage_iq (production model)
import qp0_ceiling_labels as cl  # noqa: E402
import qp1_conformal as qc  # noqa: E402
import qp1_eval as ev  # noqa: E402
import qp1_tfidf_lr as t1  # noqa: E402
import qp2_vscode_taxonomy as tax  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
OTHER = "other"
FROZEN_SHA = "95129c95809c21574ed34d41a46358149322680a7ccb0e42e90f4b62e7b761ea"  # ADR-0070
REPO = "microsoft/vscode"
SLUG = "microsoft__vscode"
IDS = {
    "v0": "majority_prior_v0",
    "v1": "tfidf_lr_v1",
    "v2": "tfidf_lr_family_only_v2",
    "prod": "production_vscode_ovr",
}
COMPLEXITY = {IDS["v0"]: 0, IDS["v1"]: 1, IDS["v2"]: 2}
CONF_ALPHA, CONF_TARGET, CONF_TOL = 0.1, 0.90, 0.02
CONF_SPECS = [
    {"name": "aps_nonrandomised"},
    {"name": "aps_randomised"},
    {"name": "raps_l0.1_k1", "lam": 0.1, "k_reg": 1},
    {"name": "lac"},
]
ADR_PATH = Path("docs/architecture/adr/0071-vscode-conformal-method.md")


# ----------------------------------------------------------------------------- taxonomy and data


def classes_sha256(classes: list[str]) -> str:
    """sha256 of the newline-joined sorted class list (ADR-0070 decision 2; `other` excluded)."""
    return hashlib.sha256("\n".join(sorted(classes)).encode()).hexdigest()


def define_classes(train: pd.DataFrame) -> list[str]:
    """Parent-area classes carried by >= 30 TRAIN issues (labels_final), `other` appended last.

    Asserts the list equals ADR-0070's frozen one (by hash): the test run uses it as is.
    """
    fam = tax.family(
        tax.window_counts(pd.DataFrame({"labels_final": train["final"]}), tax.parent_of)
    )
    got = classes_sha256(fam)
    if got != FROZEN_SHA:
        raise AssertionError(f"class list sha256 {got} != ADR-0070 {FROZEN_SHA}")
    return [*sorted(fam), OTHER]


def to_class(label: str, classes: list[str]) -> str:
    """parent_of(label) if it is a modelled class, else `other`."""
    p = tax.parent_of(label)
    return p if p in classes and p != OTHER else OTHER


def gold(d: pd.DataFrame, classes: list[str]) -> tuple[list[str], list[frozenset[str]]]:
    """Primary-label class and the set of classes of all final labels, per issue."""
    prim = [to_class(p, classes) for p in d["primary"]]
    sets = [frozenset(to_class(x, classes) for x in s) for s in d["final"]]
    return prim, sets


def load_split(derived: Path, splits_dir: Path, name: str) -> pd.DataFrame:
    """All issues of ONE split with per_issue labels (family = non-excluded), text and flags."""
    sp = pd.read_parquet(splits_dir / f"{name}.parquet", columns=["number", "classifier_eligible"])
    cols = ["number", "created_at", "title", "body", "labels_final", "labels_t72h", "text_hash"]
    issues = pd.read_parquet(derived / "issues.parquet", columns=cols)
    issues = (
        issues[issues["number"].isin(sp["number"])].sort_values("number").reset_index(drop=True)
    )
    events = pd.read_parquet(derived / "label_events.parquet")
    events = events[events["number"].isin(issues["number"])]
    d = cl.per_issue(issues, events, tax.not_excluded).merge(sp, on="number")
    d = d.merge(issues[["number", "title", "body"]], on="number")
    d["text"] = d["title"].fillna("") + "\n" + d["body"].fillna("").str.slice(0, 4000)
    d["split"] = name
    return d.reset_index(drop=True)


def labelled_eligible(d: pd.DataFrame) -> pd.DataFrame:
    """Evaluation / training population: >= 1 non-excluded label and classifier_eligible."""
    keep = d["final"].map(bool).astype(bool) & d["classifier_eligible"].astype(bool)
    return d[keep].reset_index(drop=True)


# ------------------------------------------------------------------------------------- scoring


class Ranked:
    """Per-issue top-3 class indices for the strict and the lenient (k8s-convention) reading."""

    def __init__(self, strict: np.ndarray, lenient: np.ndarray) -> None:
        self.strict, self.lenient = strict[:, :3], lenient[:, :3]


def rank_proba(p: np.ndarray, other: int, abstain: np.ndarray | None = None) -> Ranked:
    """Rankings from probabilities; with `abstain`, the lenient list leads with `other`."""
    strict = ev.ranked(p)[:, :3]
    if abstain is None:
        return Ranked(strict, strict)
    lenient = np.where(
        abstain[:, None], np.column_stack([np.full(len(p), other), strict[:, :2]]), strict
    )
    return Ranked(strict, lenient)


def hit_arrays(y: np.ndarray, sm: np.ndarray, rk: Ranked, other: int) -> dict[str, np.ndarray]:
    """Per-issue booleans: strict / lenient top-1 and top-3, strict any-valid, abstain, in-family."""
    rows = np.arange(len(y))
    gold_in = y != other
    sm_in = sm.copy()
    sm_in[:, other] = False
    return {
        "strict_top1": gold_in & (rk.strict[:, 0] == y),
        "strict_top3": gold_in & (rk.strict == y[:, None]).any(axis=1),
        "lenient_top1": rk.lenient[:, 0] == y,
        "lenient_top3": (rk.lenient == y[:, None]).any(axis=1),
        "anyvalid_strict_top1": sm_in[rows, rk.strict[:, 0]],
        "anyvalid_strict_top3": sm_in[rows[:, None], rk.strict].any(axis=1),
        "abstain": rk.lenient[:, 0] == other,
        "gold_in_family": gold_in,
    }


def prop(k: int, n: int) -> dict[str, Any]:
    return {"k": int(k), "n": int(n), "rate": k / n if n else None, "wilson95": wilson_list(k, n)}


def wilson_list(k: int, n: int) -> list[float] | None:
    w = ev.wilson(int(k), int(n))
    return None if w is None else list(w)


def score(split, final, y_primary, y_sets, rk, classes, n_boot=ev.N_BOOT) -> dict[str, Any]:  # noqa: ANN001
    """All Phase 1 vscode metrics for one model on one split (test requires final=True)."""
    ev.guard(split, final)
    other = classes.index(OTHER)
    y, sm = ev.to_idx(y_primary, classes), ev.set_mask(y_sets, classes)
    h = hit_arrays(y, sm, rk, other)
    n, fam = len(y), h["gold_in_family"]
    out: dict[str, Any] = {"split": split, "n": n, "n_in_family": int(fam.sum())}
    for k in ("strict_top1", "strict_top3", "lenient_top1", "lenient_top3"):
        out[k] = ev.bootstrap(lambda i, a=h[k]: float(a[i].mean()), n, n_boot)
        out[k]["wilson95"] = wilson_list(h[k].sum(), n)
    for k in ("anyvalid_strict_top1", "anyvalid_strict_top3"):
        out[k] = ev.bootstrap(lambda i, a=h[k]: float(a[i].mean()), n, n_boot)
    for k in ("strict_top1", "strict_top3"):  # accuracy among issues the family can express
        a = h[k][fam]
        out[f"in_family_{k.split('_')[1]}"] = ev.bootstrap(
            lambda i, a=a: float(a[i].mean()), len(a), n_boot
        )
        out[f"in_family_{k.split('_')[1]}"]["wilson95"] = wilson_list(a.sum(), len(a))
    pred = rk.lenient[:, 0]
    out["macro_f1"] = ev.bootstrap(lambda i: ev.macro_f1(y[i], pred[i], len(classes)), n, n_boot)
    ab, g_oth = h["abstain"], ~fam
    out["abstention"] = {
        "rate": prop(ab.sum(), n),
        "precision": prop((ab & g_oth).sum(), ab.sum()),
        "recall": prop((ab & g_oth).sum(), g_oth.sum()),
    }
    out["coverage_bound_strict"] = prop(fam.sum(), n)
    return out


def paired(h_a: dict[str, np.ndarray], h_b: dict[str, np.ndarray], n_boot: int) -> dict[str, Any]:
    """Paired bootstrap CI of (A - B) for strict top-1 / top-3, in-family top-3, lenient top-3."""
    n, fam = len(h_a["strict_top1"]), h_a["gold_in_family"]
    out = {}
    for k in ("strict_top1", "strict_top3", "lenient_top3"):
        out[k] = ev.paired_diff(
            lambda i, a=h_a[k]: float(a[i].mean()),
            lambda i, b=h_b[k]: float(b[i].mean()),
            n,
            n_boot,
        )
    ia = h_a["strict_top3"][fam].astype(float)
    ib = h_b["strict_top3"][fam].astype(float)
    out["in_family_top3"] = ev.paired_diff(
        lambda i: float(ia[i].mean()), lambda i: float(ib[i].mean()), len(ia), n_boot
    )
    return out


# ---------------------------------------------------------------------------------- candidates


def majority_proba(y_train: list[str], classes: list[str], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Train class frequencies as a probability row broadcast to n issues."""
    cnt = np.array([y_train.count(c) for c in classes], dtype=float)
    prior = cnt / cnt.sum()
    return np.tile(prior, (n, 1)), prior


def best_abstention_tau(pmax: np.ndarray, is_other: np.ndarray) -> float:
    """tau (validation max-prob quantile grid) maximising abstention F1; ties -> higher precision."""
    best = (-1.0, -1.0, 0.0)
    for tau in np.unique(np.quantile(pmax, np.linspace(0.0, 1.0, 201))):
        ab = pmax < tau
        tp = float((ab & is_other).sum())
        prec = tp / ab.sum() if ab.sum() else 0.0
        rec = tp / max(is_other.sum(), 1)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if (f1, prec) > best[:2]:
            best = (f1, prec, float(tau))
    return best[2]


def production_proba(d: pd.DataFrame, classes: list[str], models_dir: Path) -> np.ndarray:
    """Served vscode classifier -> n x len(classes): per mapped class, max calibrated score."""
    from triage_iq.models.component_classifier import _build_text, load_classifier

    clf = load_classifier(models_dir, "microsoft_vscode")
    raw = clf.predict_proba_calibrated(_build_text(d["title"], d["body"]))
    out = np.zeros((len(d), len(classes)))
    for j, c in enumerate(clf.classes_()):
        k = classes.index(to_class(str(c), classes))
        out[:, k] = np.maximum(out[:, k], raw[:, j])
    return out


def save_pred(path: Path, numbers: pd.Series, p: np.ndarray, classes: list[str]) -> Path:
    return t1.save_proba(path, numbers, p, classes)


def v2_views(p: np.ndarray, tau: float, classes: list[str]) -> tuple[Ranked, np.ndarray]:
    """Ranked + abstain flags for the family-only model."""
    ab = p.max(axis=1) < tau
    return rank_proba(p, classes.index(OTHER), ab), ab


def fit_grid(
    tag: str,
    x,
    y: list[str],
    xva,
    yva,
    sva,
    classes,
    out: Path,
    tau_fn,  # noqa: ANN001
) -> dict[str, Any]:
    """Resumable grid for V1 / V2 on validation; keeps only the best model's pickle."""
    other = classes.index(OTHER)
    cache = out / f"grid_{tag}.json"
    rows = json.loads(cache.read_text())["rows"] if cache.exists() else []
    done = {(r["C"], r["class_weight"]) for r in rows}
    for c, cw in t1.GRID:
        if (c, cw) in done:
            continue
        t0 = time.perf_counter()
        model = t1.fit_lr(x, y, c, cw)
        secs = time.perf_counter() - t0
        pv = t1.proba_full(model, xva, classes)
        tau = tau_fn(pv) if tau_fn else None
        rk = v2_views(pv, tau, classes)[0] if tau_fn else rank_proba(pv, other)
        sc = score("validation", False, yva, sva, rk, classes, n_boot=200)
        row = {
            "C": c, "class_weight": cw, "fit_seconds": secs, "tau": tau,
            "top3": sc["strict_top3"]["point"], "macro_f1": sc["macro_f1"]["point"],
        }  # fmt: skip
        prev = min(rows, key=t1.sel_key) if rows else None
        rows.append(row)
        if prev is None or t1.sel_key(row) < t1.sel_key(prev):
            joblib.dump(
                {"model": model, "tau": tau, "C": c, "class_weight": cw}, out / f"best_{tag}.joblib"
            )
        cache.write_text(json.dumps({"rows": rows}, indent=1))
        print(f"{tag} C={c} cw={cw}: strict_top3={row['top3']:.4f} f1={row['macro_f1']:.4f} {secs:.0f}s", file=sys.stderr, flush=True)  # fmt: skip
    best = min(rows, key=t1.sel_key)
    return {"rows": rows, "best": best}


def sel_key(sc: dict[str, Any], model_id: str) -> tuple[float, float, int]:
    """Lower is better: validation strict top-3, macro-F1, then the simpler model."""
    return (-sc["strict_top3"]["point"], -sc["macro_f1"]["point"], COMPLEXITY[model_id])


# ---------------------------------------------------------------------------------- conformal


def conformal_select(
    p: np.ndarray, y: np.ndarray, reps: int = qc.REPS, seed: int = qc.SEED
) -> dict[str, Any]:
    """Validation-only selection of the conformal score at alpha 0.1 (see module docstring)."""
    table = []
    for spec in CONF_SPECS:
        cov, size = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed * 1000 + r)
            perm = rng.permutation(len(y))
            a, b = perm[: len(y) // 2], perm[len(y) // 2 :]
            q = qc.calibrate(p[a], y[a], spec, rng.random(len(a)), CONF_ALPHA)
            mask = qc.predict_mask(p[b], q, spec, rng.random(len(b)))
            cov.append(mask[np.arange(len(b)), y[b]].mean())
            size.append(mask.sum(axis=1).mean())
        table.append({
            "spec": spec,
            "coverage": {"mean": float(np.mean(cov)), "sd": float(np.std(cov, ddof=1))},
            "set_size_mean": {"mean": float(np.mean(size)), "sd": float(np.std(size, ddof=1))},
        })  # fmt: skip
    return {"table": table, "selected": conformal_choose(table)["spec"], "alpha": CONF_ALPHA}


def conformal_choose(table: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage within 0.90 +/- 0.02, then smallest mean size, then name; else closest coverage."""
    ok = [r for r in table if abs(r["coverage"]["mean"] - CONF_TARGET) <= CONF_TOL]
    if ok:
        return min(ok, key=lambda r: (r["set_size_mean"]["mean"], r["spec"]["name"]))
    return min(table, key=lambda r: (abs(r["coverage"]["mean"] - CONF_TARGET), r["spec"]["name"]))


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
    conf = writer.data.get("conformal_selection")
    if (
        not conf
        or not ADR_PATH.exists()
        or conf["selected"]["name"] not in ADR_PATH.read_text(encoding="utf-8")
    ):
        raise RuntimeError("conformal choice must be frozen in ADR-0071 (naming the score) before test")  # fmt: skip
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
        t0 = time.perf_counter()
        preds[sel_id] = t1.proba_full(best["model"], xte, classes)
        lat[sel_id] = {
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
        clf.predict_proba_calibrated(_build_text(pd.Series([r["title"]]), pd.Series([r["body"]])))
        one.append((time.perf_counter() - t0) * 1000)
    lat[IDS["prod"]] |= {"single_issue_ms_p50": float(np.percentile(one, 50)), "single_issue_ms_p95": float(np.percentile(one, 95))}  # fmt: skip
    views[IDS["prod"]] = rank_proba(preds[IDS["prod"]], other)
    if IDS["v0"] in ids and sel_id != IDS["v0"]:
        preds[IDS["v0"]], _ = majority_proba(ytr, classes, len(te))
        views[IDS["v0"]] = rank_proba(preds[IDS["v0"]], other)
    y, sm = ev.to_idx(yte, classes), ev.set_mask(ste, classes)
    hits = {m: hit_arrays(y, sm, views[m], other) for m in ids}
    for m in ids:
        pf = save_pred(out / f"test_{m}.parquet", te["number"], preds[m], classes)
        writer.add(
            m, "test", score("test", True, yte, ste, views[m], classes, n_boot), pf, final=True
        )
    # conformal on test (exploratory; calibrated on FULL validation population of the selected model)
    spec = conf["selected"]
    pv = (
        pd.read_parquet(out / f"validation_{tag}.parquet")[classes].to_numpy()
        if sel_id != IDS["v0"]
        else None
    )
    cres: dict[str, Any] = {"exploratory": "ADR-0069: unconfirmed until the post-test confirmation set", "spec": spec, "alpha": CONF_ALPHA}  # fmt: skip
    if pv is not None:
        yv = ev.to_idx(yva, classes)
        kv = np.ones(len(yv), bool) if tag != "v2" else yv != other
        q = qc.calibrate(pv[kv], yv[kv], spec, qc.uniforms(int(kv.sum()), 0), CONF_ALPHA)
        mask = qc.predict_mask(preds[sel_id], q, spec, qc.uniforms(len(y), 1))
        cres |= {"qhat": q, "all_labelled": qc.mask_report(mask, y, sm)}
        fam = y != other
        cres["in_family_only"] = qc.mask_report(mask[fam], y[fam], sm[fam])
    writer.data["conformal_test"] = cres
    writer.data["paired_vs"] = {
        f"{sel_id}_minus_{m}": paired(hits[sel_id], hits[m], n_boot) for m in ids if m != sel_id
    }
    writer.data["latency_cpu"] = lat
    ceil = json.loads(
        Path("reports/qp0_ceiling_labels_microsoft__vscode_2026-10.json").read_text(
            encoding="utf-8"
        )
    )
    writer.data["bounds"] = {
        "spec_strict_test_coverage_bound_all_labelled": ceil["coverage_bounded_ceiling"][
            "labelled"
        ]["test"]["primary_in_family_top1_bound"],  # fmt: skip
        "measured_bound_on_eligible_labelled_test": prop(int((y != other).sum()), len(y)),
    }
    writer.data["populations"] |= {
        "test_eligible_total": int(te_all["classifier_eligible"].sum()),
        "test_labelled_eligible": len(te),
        "test_unlabelled_eligible": int(te_all["classifier_eligible"].sum()) - len(te),
    }
    writer.save()
    s = writer.data["models"][sel_id]["test"]
    return {
        k: round(s[k]["point"], 4)
        for k in ("strict_top1", "strict_top3", "lenient_top3", "macro_f1")
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vscode Phase 1: CPU component candidates")
    ap.add_argument("--stage", choices=["validation", "conformal", "test"], required=True)
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
        elif a.stage == "conformal":
            r = stage_conformal(a, derived, splits_dir, models_dir)
        else:
            r = stage_test(a, derived, splits_dir, models_dir)
    print(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
