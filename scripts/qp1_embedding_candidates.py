"""Phase 1 candidates 2, 3 and 6 on the kubernetes sig/* task, on the BGE document embedding.

PRE-REGISTERED DESIGN (2026-10-23; written before any validation or test number of these models
was computed; do not change after seeing test results). Gold per ADR-0067 from the complete-
timeline tables (derived_v2; identical to the old gold). Classes, population and text as in
qp1_tfidf_lr.py. Input = BAAI/bge-base-en-v1.5 document embedding (title + "\\n" + body[:4000],
no instruction, L2-normalised, from Kaggle). Everything is fitted on TRAIN (classifier_eligible,
labelled), selected on VALIDATION, and TEST labels are loaded only by `--final-test`, which
evaluates each model id once. Candidate 1 (TF-IDF+LR C=64 balanced) is NOT re-evaluated: its saved
validation / test probabilities are reused as the paired baseline. Seed 42 everywhere.
 C3a  multinomial LR on the embedding: C in {0.25, 1, 4, 16} x class_weight {None, balanced}.
 C3b  sklearn MLPClassifier, one hidden layer of 256, early stopping on a train-internal 10%
      split, alpha in {1e-4, 1e-3} (run only if it fits the 10-minute local budget).
 C4   kNN vote over TRAIN issues by BGE cosine: k in {5, 10, 20, 50}; class score = sum of the
      (non-negative) similarities of neighbours carrying the class, either as PRIMARY label or as
      ANY of the neighbour's labels; normalised to probabilities (+1e-6 smoothing so NLL is finite).
 C6   stacked ensemble: multinomial LR over the concatenated log-probabilities (clipped at 1e-4)
      of candidate 1, the selected C3a and the selected C4. The stacker is trained on 5-fold
      (stratified on the primary label, seed 42) OUT-OF-FOLD TRAIN probabilities of the three
      base models (each re-fitted inside every fold, the TF-IDF vocabulary included); its C in
      {0.01, 0.03, 0.1, 0.3, 1, 3} x class_weight {None, balanced} is selected on VALIDATION;
      at inference the base models are fitted on the full train set.
SELECTION RULE (applies within a family and across the new families C3a, C3b, C4, C6): primary =
validation top-3 accuracy; tie-break = validation macro-F1; then the simpler model (kNN < LR <
MLP < stack, then the smaller C / smaller k / larger alpha). Candidate 1 is the baseline, not part
of the selection pool. Reported: every candidate on validation with bootstrap CIs and paired
differences vs candidate 1 (qp1_eval.py: 2000 resamples, seed 42, same resampled indices).
TEST: one evaluation per family best (the selected candidate is one of them) with paired
differences vs candidate 1. Conformal = the FROZEN ADR-0069 method (RAPS lambda 0.1, k_reg 1,
randomised seed 42, alpha 0.05, calibrated on validation only), EXPLORATORY, on the selected model
and on candidate 1. Latency = median single-row CPU ms over 200 test issues with the BGE vector
already available (the BGE encode cost is reported separately, not measured here).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_conformal as cf  # noqa: E402
import qp1_eval as ev  # noqa: E402
import qp1_tfidf_lr as tl  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
SEED, FOLDS, N_LAT = 42, 5, 200
FAMILY_RANK = {"c4_knn_bge": 0, "c3a_lr_bge": 1, "c3b_mlp_bge": 2, "c6_stack": 3}
C1_ID, C1_CFG = "tfidf_lr_v1", {"C": 64, "class_weight": "balanced"}
RAPS = "raps_l0.1_k1"


def sel_key(row: dict[str, Any]) -> tuple:
    """Lower is better: validation top-3, macro-F1, family simplicity, config simplicity."""
    return (-row["top3"], -row["macro_f1"], FAMILY_RANK[row["model"]], row["param"])


def knn_proba(sims: np.ndarray, y: np.ndarray, sets: np.ndarray, k: int, mode: str) -> np.ndarray:
    """Neighbour-label vote. sims = queries x references cosine; y = reference primary class
    index; sets = reference x C label-set mask; mode 'primary' or 'any'. Rows sum to 1."""
    nb = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    w = np.clip(np.take_along_axis(sims, nb, axis=1), 0.0, None)
    if mode == "primary":
        score = np.zeros((sims.shape[0], sets.shape[1]))
        for j in range(k):
            np.add.at(score, (np.arange(len(sims)), y[nb[:, j]]), w[:, j])
    else:
        score = np.einsum("nk,nkc->nc", w, sets[nb].astype(float))
    score = score + 1e-6
    return score / score.sum(axis=1, keepdims=True)


def cross_fit(
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    y: np.ndarray,
    n_classes: int,
    folds: int = FOLDS,
) -> np.ndarray:
    """Out-of-fold probabilities: fit_predict(train_idx, held_idx) never sees the held rows."""
    oof, seen = np.zeros((len(y), n_classes)), np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    for tr_idx, held in skf.split(np.zeros(len(y)), y):
        assert not set(tr_idx) & set(held), "a fold's own rows leaked into its training set"
        oof[held] = fit_predict(tr_idx, held)
        seen[held] += 1
    assert (seen == 1).all(), "every row must be held out exactly once"
    return oof


def stack_features(*probs: np.ndarray) -> np.ndarray:
    return np.hstack([np.log(np.clip(p, 1e-4, None)) for p in probs])


def make_lr(c: float, cw: str | None) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight=cw, max_iter=1000, random_state=SEED)


def make_mlp(alpha: float) -> MLPClassifier:
    return MLPClassifier((256,), alpha=alpha, early_stopping=True, validation_fraction=0.1,
                         max_iter=200, random_state=SEED)  # fmt: skip


def head_proba(model: Any, x: np.ndarray, classes: list[str]) -> np.ndarray:  # noqa: ANN401
    """Probabilities of a head fitted on string (LR) or integer-id (MLP) labels, aligned."""
    out = np.zeros((x.shape[0], len(classes)))
    cols = [c if isinstance(c, int | np.integer) else classes.index(c) for c in model.classes_]
    out[:, cols] = model.predict_proba(x)
    return out


def load_emb(emb_dir: Path, slug: str, numbers: pd.Series) -> np.ndarray:
    ids = pd.read_parquet(emb_dir / f"{slug}__issues_ids.parquet")
    pos = {int(n): i for i, n in enumerate(ids["number"])}
    emb = np.load(emb_dir / f"{slug}__issues_emb.npy").astype(np.float32)
    return emb[[pos[int(n)] for n in numbers]]


def load_saved(path: Path, numbers: pd.Series, classes: list[str]) -> np.ndarray:
    """Saved probabilities re-ordered to `numbers` (KeyError if an issue is missing)."""
    df = pd.read_parquet(path).set_index("number")
    return df.loc[numbers.to_numpy(), classes].to_numpy()


def paired_vs_c1(p: np.ndarray, p1: np.ndarray, y, sets, classes, n_boot: int) -> dict:  # noqa: ANN001
    """Paired bootstrap differences (candidate - candidate 1) on the same resampled issues."""
    h, h1, yi = ev.hit_arrays(y, sets, p, classes), ev.hit_arrays(y, sets, p1, classes), ev.to_idx(y, classes)  # fmt: skip
    pr, pr1, n, out = p.argmax(1), p1.argmax(1), len(y), {}
    for m in ("top1", "top3", "anyvalid_top1", "anyvalid_top3"):
        out[m] = ev.paired_diff(lambda i, a=h[m]: float(a[i].mean()), lambda i, b=h1[m]: float(b[i].mean()), n, n_boot)  # fmt: skip
    out["macro_f1"] = ev.paired_diff(lambda i: ev.macro_f1(yi[i], pr[i], len(classes)),
                                     lambda i: ev.macro_f1(yi[i], pr1[i], len(classes)), n, n_boot)  # fmt: skip
    for v in out.values():
        v["ci_excludes_zero"] = bool(v["ci95"][0] > 0 or v["ci95"][1] < 0)
    return out


def raps_conformal(pv, yv, pt, yt, st, classes) -> dict:  # noqa: ANN001
    """FROZEN ADR-0069 method: calibrate on validation, apply to test. EXPLORATORY."""
    spec = next(s for s in cf.specs() if s["name"] == RAPS)
    q = cf.calibrate(pv, ev.to_idx(yv, classes), spec, cf.uniforms(len(yv), 0))
    mask = cf.predict_mask(pt, q, spec, cf.uniforms(len(yt), 1))
    return {"label": "EXPLORATORY: ADR-0069 frozen method", "spec": spec, "qhat": q} | cf.mask_report(
        mask, ev.to_idx(yt, classes), ev.set_mask(st, classes))  # fmt: skip


def median_ms(fn: Callable[[int], Any], n: int = N_LAT) -> float:
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


class Ctx:
    """Train/validation data, embeddings and the fitting recipes shared by both stages."""

    def __init__(self, a: argparse.Namespace) -> None:
        d = tl.load_split(a.derived, a.splits, ("train", "validation"))
        self.modelled, self.classes = tl.define_classes(d[d["split"] == "train"])
        self.tr, self.va = (d[(d["split"] == s) & d["classifier_eligible"]].reset_index(drop=True)
                            for s in ("train", "validation"))  # fmt: skip
        self.ytr, self.str_ = tl.gold(self.tr, self.modelled)
        self.yva, self.sva = tl.gold(self.va, self.modelled)
        self.yi = ev.to_idx(self.ytr, self.classes)
        self.sm = ev.set_mask(self.str_, self.classes)
        self.xtr = load_emb(a.emb_dir, a.slug, self.tr["number"])
        self.xva = load_emb(a.emb_dir, a.slug, self.va["number"])
        self.c1_dir = a.models / "tfidf_lr"
        self.c1_val = load_saved(self.c1_dir / "validation.parquet", self.va["number"], self.classes)  # fmt: skip

    def knn(self, xq: np.ndarray, k: int, mode: str, ref: np.ndarray | None = None) -> np.ndarray:
        ref = np.arange(len(self.xtr)) if ref is None else ref
        return knn_proba(xq @ self.xtr[ref].T, self.yi[ref], self.sm[ref], k, mode)

    def fit_head(self, cfg: dict, idx: np.ndarray | None = None) -> Any:  # noqa: ANN401
        idx = np.arange(len(self.xtr)) if idx is None else idx
        m = make_lr(cfg["C"], cfg["class_weight"]) if "C" in cfg else make_mlp(cfg["alpha"])
        # the MLP gets integer class ids: sklearn's early-stopping score fails on string labels
        return m.fit(self.xtr[idx], self.yi[idx] if "alpha" in cfg else [self.ytr[i] for i in idx])

    def c1_fit(self, idx: np.ndarray) -> tuple[Any, Any]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        x = vec.fit_transform(self.tr["text"].iloc[idx])
        return vec, tl.fit_lr(x, [self.ytr[i] for i in idx], C1_CFG["C"], C1_CFG["class_weight"])

    def oof(self, kind: str, cfg: dict) -> np.ndarray:
        """5-fold out-of-fold TRAIN probabilities of one base model."""
        nc = len(self.classes)
        if kind == "c1":

            def fp(tr_i, he):  # noqa: ANN001, ANN202
                vec, m = self.c1_fit(tr_i)
                return tl.proba_full(m, vec.transform(self.tr["text"].iloc[he]), self.classes)
        elif kind == "c3a":

            def fp(tr_i, he):  # noqa: ANN001, ANN202
                return head_proba(self.fit_head(cfg, tr_i), self.xtr[he], self.classes)
        else:

            def fp(tr_i, he):  # noqa: ANN001, ANN202
                return self.knn(self.xtr[he], cfg["k"], cfg["mode"], tr_i)

        return cross_fit(fp, self.yi, nc)


def stacker(oofs: list[np.ndarray], ctx: Ctx, cfg: dict) -> LogisticRegression:
    return make_lr(cfg["C"], cfg["class_weight"]).fit(stack_features(*oofs), ctx.ytr)


def val_row(model: str, cfg: dict, param: float, pv: np.ndarray, ctx: Ctx) -> dict:
    return {"model": model, "config": cfg, "param": param, **tl.point_metrics(pv, ctx.yva, ctx.sva, ctx.classes)}  # fmt: skip


def select(a: argparse.Namespace) -> dict[str, Any]:
    """Stage 1: all fitting and selection on VALIDATION; test labels are never loaded."""
    ctx = Ctx(a)
    rows: dict[str, list[dict]] = {k: [] for k in FAMILY_RANK}
    pvs: dict[tuple[str, str], np.ndarray] = {}

    def add(model: str, cfg: dict, param: float, pv: np.ndarray) -> None:
        r = val_row(model, cfg, param, pv, ctx)
        rows[model].append(r)
        pvs[(model, json.dumps(cfg, sort_keys=True))] = pv
        print(f"{model} {cfg}: top3={r['top3']:.3f} f1={r['macro_f1']:.3f}", file=sys.stderr)

    for cw in (None, "balanced"):
        for c in (0.25, 1, 4, 16):
            cfg = {"C": c, "class_weight": cw}
            add("c3a_lr_bge", cfg, c, head_proba(ctx.fit_head(cfg), ctx.xva, ctx.classes))
    for al in (1e-4, 1e-3):
        t0 = time.perf_counter()
        cfg = {"alpha": al}
        add("c3b_mlp_bge", cfg, -al, head_proba(ctx.fit_head(cfg), ctx.xva, ctx.classes))
        print(f"mlp fit+eval {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    for k in (5, 10, 20, 50):
        for mode in ("primary", "any"):
            add("c4_knn_bge", {"k": k, "mode": mode}, k + (0.5 if mode == "any" else 0), ctx.knn(ctx.xva, k, mode))  # fmt: skip
    best = {m: min(rows[m], key=sel_key) for m in rows if rows[m]}
    key = lambda m: (m, json.dumps(best[m]["config"], sort_keys=True))  # noqa: E731
    oofs = [ctx.oof("c1", C1_CFG), ctx.oof("c3a", best["c3a_lr_bge"]["config"]), ctx.oof("c4", best["c4_knn_bge"]["config"])]  # fmt: skip
    vin = [ctx.c1_val, pvs[key("c3a_lr_bge")], pvs[key("c4_knn_bge")]]
    for cw in (None, "balanced"):
        for c in (0.01, 0.03, 0.1, 0.3, 1, 3):
            cfg = {"C": c, "class_weight": cw}
            m = stacker(oofs, ctx, cfg)
            add("c6_stack", cfg, c, tl.proba_full(m, stack_features(*vin), ctx.classes))
    best["c6_stack"] = min(rows["c6_stack"], key=sel_key)
    selected = min(best.values(), key=sel_key)
    out = ctx.c1_dir.parent / "emb_candidates"
    out.mkdir(exist_ok=True)
    np.savez(out / "oof_train.npz", c1=oofs[0], c3a=oofs[1], c4=oofs[2])
    sel = {"best": best, "selected_model": selected["model"], "grid": rows, "classes": ctx.classes}
    (out / "selection.json").write_text(json.dumps(sel, indent=1), encoding="utf-8")
    writer = ev.ResultsWriter(a.report, a.manifest)
    ref = ev.evaluate("validation", False, ctx.yva, ctx.sva, ctx.c1_val, ctx.classes)
    writer.add(C1_ID + "_reference", "validation", ref, ctx.c1_dir / "validation.parquet")
    for m in best:
        pv = pvs[key(m)]
        f = tl.save_proba(out / f"{m}_validation.parquet", ctx.va["number"], pv, ctx.classes)
        metrics = ev.evaluate("validation", False, ctx.yva, ctx.sva, pv, ctx.classes)
        metrics["paired_vs_candidate1"] = paired_vs_c1(pv, ctx.c1_val, ctx.yva, ctx.sva, ctx.classes, ev.N_BOOT)  # fmt: skip
        writer.add(m, "validation", metrics, f)
    writer.data.update({"design": DESIGN, "harness_design": ev.DESIGN, "validation_grid": rows,
                        "selection": {"per_family_best": best, "selected_model": selected["model"],
                        "selected_row": selected}, "classes": ctx.classes, "repo": a.repo})  # fmt: skip
    writer.save()
    return sel


def final_test(a: argparse.Namespace) -> dict[str, Any]:
    """Stage 2: refit the selected configs on train, evaluate TEST once per model id."""
    out = a.models / "emb_candidates"
    sel = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    writer = ev.ResultsWriter(a.report, a.manifest)
    for m in sel["best"]:
        if "test" in writer.data["models"].get(m, {}):  # fail before touching any test label
            raise ev.TestGuardError(f"test already evaluated for {m} in {a.report.name}")
    ctx = Ctx(a)
    assert ctx.classes == sel["classes"], "class definition drifted since selection"
    best = {m: r["config"] for m, r in sel["best"].items()}
    oof = np.load(out / "oof_train.npz")
    head = {m: ctx.fit_head(best[m]) for m in ("c3a_lr_bge", "c3b_mlp_bge") if m in best}
    stk = stacker([oof["c1"], oof["c3a"], oof["c4"]], ctx, best["c6_stack"])
    vec1, m1 = ctx.c1_fit(np.arange(len(ctx.tr)))
    assert np.allclose(tl.proba_full(m1, vec1.transform(ctx.va["text"]), ctx.classes), ctx.c1_val, atol=1e-6)  # fmt: skip
    te = tl.load_split(
        a.derived, a.splits, ("test",)
    )  # the first and only place test labels are read
    te = te[te["classifier_eligible"]].reset_index(drop=True)
    yte, ste = tl.gold(te, ctx.modelled)
    xte = load_emb(a.emb_dir, a.slug, te["number"])
    c1_te = load_saved(ctx.c1_dir / "test.parquet", te["number"], ctx.classes)
    assert np.allclose(tl.proba_full(m1, vec1.transform(te["text"]), ctx.classes), c1_te, atol=1e-6)  # fmt: skip
    pv, pt = {}, {}
    for m, h in head.items():
        pv[m], pt[m] = (head_proba(h, x, ctx.classes) for x in (ctx.xva, xte))
    kc = best["c4_knn_bge"]
    pv["c4_knn_bge"], pt["c4_knn_bge"] = ctx.knn(ctx.xva, kc["k"], kc["mode"]), ctx.knn(xte, kc["k"], kc["mode"])  # fmt: skip
    pv["c6_stack"] = tl.proba_full(stk, stack_features(ctx.c1_val, pv["c3a_lr_bge"], pv["c4_knn_bge"]), ctx.classes)  # fmt: skip
    pt["c6_stack"] = tl.proba_full(stk, stack_features(c1_te, pt["c3a_lr_bge"], pt["c4_knn_bge"]), ctx.classes)  # fmt: skip
    for m in pv:
        saved = load_saved(out / f"{m}_validation.parquet", ctx.va["number"], ctx.classes)
        assert np.allclose(pv[m], saved, atol=1e-4), f"refit of {m} differs from stage-1 validation"
    lat = latencies(ctx, best, head, stk, vec1, m1, te, xte)
    tops = {}
    for m in pt:
        f = tl.save_proba(out / f"{m}_test.parquet", te["number"], pt[m], ctx.classes)
        t = ev.evaluate("test", True, yte, ste, pt[m], ctx.classes)
        t["paired_vs_candidate1"] = paired_vs_c1(pt[m], c1_te, yte, ste, ctx.classes, ev.N_BOOT)
        t["selected_by_validation"] = m == sel["selected_model"]
        t["cpu_latency_ms_per_issue_bge_precomputed"] = lat[m]
        t["conformal_exploratory"] = raps_conformal(pv[m], ctx.yva, pt[m], yte, ste, ctx.classes) if t["selected_by_validation"] else None  # fmt: skip
        writer.add(m, "test", t, f, final=True)
        tops[m] = t
    sm = sel["selected_model"]
    c1_test = {"conformal_exploratory": raps_conformal(ctx.c1_val, ctx.yva, c1_te, yte, ste, ctx.classes),
               "cpu_latency_ms_per_issue": lat["c1"]}  # fmt: skip
    writer.data["candidate1_context"] = c1_test
    writer.data["bge_encode_cost"] = {"measured_here": False, "reference": "docs/DECISION_LOG_2026-10.md:168 single-text encode ~203-236 ms median on a 2-vCPU container; bulk Kaggle T4 encode 7733 issues in 44.9 s (run_info.json)"}  # fmt: skip
    writer.data["selected_model"] = sm
    writer.save()
    return tops


def latencies(ctx: Ctx, best: dict, head: dict, stk, vec1, m1, te, xte) -> dict[str, float]:  # noqa: ANN001
    """Median single-row CPU ms with the BGE vector already available (c1/c6 include TF-IDF)."""
    kc, cl = best["c4_knn_bge"], ctx.classes
    c1 = lambda i: tl.proba_full(m1, vec1.transform([te["text"].iloc[i]]), cl)  # noqa: E731
    c3a = lambda i: head_proba(head["c3a_lr_bge"], xte[i : i + 1], cl)  # noqa: E731
    c4 = lambda i: ctx.knn(xte[i : i + 1], kc["k"], kc["mode"])  # noqa: E731
    c6 = lambda i: stk.predict_proba(stack_features(c1(i), c3a(i), c4(i)))  # noqa: E731
    out = {"c1": median_ms(c1), "c3a_lr_bge": median_ms(c3a), "c4_knn_bge": median_ms(c4), "c6_stack": median_ms(c6)}  # fmt: skip
    if "c3b_mlp_bge" in head:
        out["c3b_mlp_bge"] = median_ms(lambda i: head_proba(head["c3b_mlp_bge"], xte[i : i + 1], cl))  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BGE-embedding candidates for the sig/* task")
    ap.add_argument("--repo", default="kubernetes/kubernetes")
    ap.add_argument("--final-test", action="store_true", help="evaluate TEST once (explicit)")
    a = ap.parse_args(argv)
    a.slug, root = a.repo.replace("/", "__"), Path(r"D:\triageiq-data")
    a.derived, a.splits, a.models = (root / s / a.slug for s in ("derived_v2", "splits", "models"))
    a.emb_dir = root / "embeddings" / a.slug
    a.report = Path("reports") / f"qp1_embedding_candidates_{a.slug}_2026-10.json"
    a.manifest = Path("reports") / f"qp0_split_manifest_{a.slug}_2026-10.json"
    tl.lower_priority()
    with threadpool_limits(limits=4):
        if a.final_test:
            for m, t in final_test(a).items():
                print(m, {k: round(t[k]["point"], 4) for k in ("top1", "top3", "macro_f1")})
        else:
            s = select(a)
            print(json.dumps({"selected_model": s["selected_model"], "best": {k: v["config"] for k, v in s["best"].items()}}))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
