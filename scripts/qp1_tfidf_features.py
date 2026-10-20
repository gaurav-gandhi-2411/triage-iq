"""Phase 1.2 candidate 2: TF-IDF + LR with title / path-token / char features (and candidate 1b).

PRE-REGISTERED DESIGN (2026-10-20; do not change after seeing test results). Same data, gold,
classes, eligibility and model family as qp1_tfidf_lr.py (multinomial LR, seed 42). Feature blocks,
all TF-IDF with sublinear tf and min_df=2:
 * base (title_sep off): word 1-2 grams over title + "\\n" + body[:4000] (= candidate 1);
 * title_sep on: word 1-2 grams of the TITLE (50k) + word 1-2 grams of body[:4000] (200k) +
   char_wb 3-5 grams of the TITLE only (100k), horizontally concatenated;
 * paths on: an extra block of identifier tokens from extra_tokens(): every file / package path
   under pkg/, staging/src/k8s.io/, cmd/, test/, plugin/, cluster/, hack/, build/, vendor/, api/,
   docs/ split into directory components ("p:kubelet") plus 2- and 3-level prefixes ("pp:pkg/kubelet"),
   kubectl / kubeadm / kube-* command names with an allow-listed sub-command ("c:kubectl_apply"),
   and error-message stems (first 5 words from error / failed to / unable to / cannot, digits and
   long hex stripped: "e:error_failed_to_get_pod").
Grid on VALIDATION: C in {64, 256, 1024} x class_weight in {None, balanced} x title_sep x paths
(24 fits; the 6 rows with both off are candidate 1b, extended-C plain TF-IDF+LR). Selection:
validation top-3, tie macro-F1, tie smaller C, tie fewer features. Candidate 2 = best row with
title_sep or paths on; the plain model is tested only if its best row differs from candidate 1
(C=64, balanced). Test labels are loaded only by --final-test, once per model id; candidate 1's
committed numbers are reused (its saved probabilities only feed paired bootstrap differences and
conformal sets). The conformal score is chosen on validation only (qp1_conformal.py).
The grid runs in resumable chunks (--budget-seconds) to keep laptop jobs short.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_conformal as qc  # noqa: E402
import qp1_eval as ev  # noqa: E402
import qp1_tfidf_lr as t1  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
ROOTS = r"pkg|staging/src/k8s\.io|cmd|test|plugin|cluster|hack|build|vendor|api|docs"
PATH_RE = re.compile(rf"(?<![\w/.-])((?:{ROOTS})/[\w./-]*\w)")
CMD_RE = re.compile(r"\b(kubectl|kubeadm|kubelet|kube-[a-z-]+)\s+([a-z][a-z-]+)")
ERR_RE = re.compile(r"\b(?:error|failed to|unable to|cannot|can't|couldn't)\b[^\n.]{0,80}", re.I)
VERBS = set(["get", "apply", "delete", "describe", "create", "logs", "exec", "edit", "patch", "rollout", "scale", "drain", "cordon", "uncordon", "init", "join", "upgrade", "reset", "token", "certs", "config", "version", "run", "expose", "label", "annotate", "top", "port-forward", "cp", "auth", "proxy", "wait", "taint"])  # fmt: skip
GRID = [(c, cw, ts, pt) for ts in (False, True) for pt in (False, True) for cw in (None, "balanced") for c in (64, 256, 1024)]  # fmt: skip
C1 = (64, "balanced", False, False)  # candidate 1 (already evaluated on test)


def extra_tokens(text: str) -> str:
    """Path components, command names and error stems of one issue text (see module docstring)."""
    toks: list[str] = []
    for m in PATH_RE.findall(text):
        parts = [p for p in re.split(r"[/.]", m) if p][:8]
        toks += [f"p:{p.lower()}" for p in parts]
        toks += [f"pp:{'/'.join(parts[:n]).lower()}" for n in (2, 3) if len(parts) >= n]
    for binary, verb in CMD_RE.findall(text):
        toks.append(f"c:{binary}")
        if verb in VERBS:
            toks.append(f"c:{binary}_{verb}")
    for m in ERR_RE.findall(text):
        words = [
            w for w in re.findall(r"[a-z']+", re.sub(r"\b[0-9a-f]{8,}\b|\d+", " ", m.lower()))
        ][:5]
        if len(words) >= 2:
            toks.append("e:" + "_".join(words))
    return " ".join(toks)


def field_title(s: pd.Series) -> pd.Series:
    return s.str.split("\n").str[0]


def field_body(s: pd.Series) -> pd.Series:
    return s.str.split("\n", n=1).str[1].fillna("")


def field_full(s: pd.Series) -> pd.Series:
    return s


def field_extra(s: pd.Series) -> pd.Series:
    return s.map(extra_tokens)


class Featurizer:
    """Fitted block list (vectoriser, field function) for one (title_sep, paths) setting."""

    def __init__(self, title_sep: bool, paths: bool) -> None:
        w = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
        if title_sep:
            ch = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=100_000)  # fmt: skip
            self.blocks = [(TfidfVectorizer(max_features=50_000, **w), field_title),
                           (TfidfVectorizer(max_features=200_000, **w), field_body), (ch, field_title)]  # fmt: skip
        else:
            self.blocks = [(TfidfVectorizer(max_features=200_000, **w), field_full)]
        if paths:
            tok = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=2, sublinear_tf=True, max_features=50_000)  # fmt: skip
            self.blocks.append((tok, field_extra))

    def fit_transform(self, texts: pd.Series):  # noqa: ANN201
        return hstack([v.fit_transform(f(texts)) for v, f in self.blocks]).tocsr()

    def transform(self, texts: pd.Series):  # noqa: ANN201
        return hstack([v.transform(f(texts)) for v, f in self.blocks]).tocsr()


def key(row: dict[str, Any]) -> tuple:
    return (-row["top3"], -row["macro_f1"], row["C"], row["n_features"])


def pick(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return min(rows, key=key)


def cfg(row: dict[str, Any]) -> tuple:
    return (row["C"], row["class_weight"], row["title_sep"], row["paths"])


def load_train_val(derived: Path, splits_dir: Path):  # noqa: ANN201
    d = t1.load_split(derived, splits_dir, ("train", "validation"))
    modelled, classes = t1.define_classes(d[d["split"] == "train"])
    tr, va = (d[(d["split"] == s) & d["classifier_eligible"]].reset_index(drop=True) for s in ("train", "validation"))  # fmt: skip
    return modelled, classes, tr, va


def run_grid(derived: Path, splits_dir: Path, models_dir: Path, budget_s: float) -> dict[str, Any]:
    """Stage 1 (resumable): fit remaining grid rows on validation until the time budget is used."""
    path = models_dir / "tfidf_features" / "grid.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rows": []}
    done = {cfg(r) for r in state["rows"]}
    todo = [g for g in GRID if g not in done]
    modelled, classes, tr, va = load_train_val(derived, splits_dir)
    ytr, (yva, sva) = t1.gold(tr, modelled)[0], t1.gold(va, modelled)
    start, cache = time.monotonic(), {}
    for c, cw, ts, pt in todo:
        if time.monotonic() - start > budget_s:
            break
        if (ts, pt) not in cache:
            f = Featurizer(ts, pt)
            cache = {(ts, pt): (f.fit_transform(tr["text"]), None)}  # one featurisation at a time
            cache[(ts, pt)] = (cache[(ts, pt)][0], f.transform(va["text"]))
        xtr, xva = cache[(ts, pt)]
        t0 = time.perf_counter()
        model = t1.fit_lr(xtr, ytr, c, cw)
        secs = time.perf_counter() - t0
        pv = t1.proba_full(model, xva, classes)
        row = {"C": c, "class_weight": cw, "title_sep": ts, "paths": pt, "n_features": xtr.shape[1], "fit_seconds": secs, **t1.point_metrics(pv, yva, sva, classes)}  # fmt: skip
        state["rows"].append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        print(f"{cfg(row)} feats={xtr.shape[1]} top3={row['top3']:.3f} f1={row['macro_f1']:.3f} {secs:.0f}s", file=sys.stderr)  # fmt: skip
    state["complete"] = len(state["rows"]) == len(GRID)
    if state["complete"]:
        plain = [r for r in state["rows"] if not (r["title_sep"] or r["paths"])]
        feat = [r for r in state["rows"] if r["title_sep"] or r["paths"]]
        state["selection"] = {"candidate_2": cfg(pick(feat)), "plain_extC_best": cfg(pick(plain)), "overall_best": cfg(pick(state["rows"]))}  # fmt: skip
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    return state


def fit_model(
    cfg_: tuple, tr: pd.DataFrame, va: pd.DataFrame, modelled: list[str], classes: list[str]
):  # noqa: ANN201
    c, cw, ts, pt = cfg_
    f = Featurizer(ts, pt)
    xtr = f.fit_transform(tr["text"])
    t0 = time.perf_counter()
    model = t1.fit_lr(xtr, t1.gold(tr, modelled)[0], c, cw)
    train_s = time.perf_counter() - t0
    return f, model, t1.proba_full(model, f.transform(va["text"]), classes), train_s


def final_test(
    derived, splits_dir, models_dir, report, manifest, c1_report, n_boot=ev.N_BOOT
) -> dict:  # noqa: ANN001
    """Stage 2: refit the selected configs, choose conformal scores on validation, test once."""
    grid = json.loads((models_dir / "tfidf_features" / "grid.json").read_text(encoding="utf-8"))
    assert grid["complete"], "grid incomplete: run stage 1 until complete"
    sel = {k: tuple(v) for k, v in grid["selection"].items()}
    todo = {"tfidf_feat_v1": sel["candidate_2"]}
    if sel["plain_extC_best"] != C1:
        todo["tfidf_lr_extC_v1"] = sel["plain_extC_best"]
    writer = ev.ResultsWriter(report, manifest)
    for mid in todo:  # fail before touching any test label
        if "test" in writer.data["models"].get(mid, {}):
            raise ev.TestGuardError(f"test already evaluated for {mid} in {Path(report).name}")
    modelled, classes, tr, va = load_train_val(derived, splits_dir)
    yva, sva = t1.gold(va, modelled)
    yvai = ev.to_idx(yva, classes)
    fitted = {}
    for mid, c in todo.items():
        f, model, pv, secs = fit_model(c, tr, va, modelled, classes)
        t1.save_proba(models_dir / mid / "validation.parquet", va["number"], pv, classes)
        fitted[mid] = (f, model, pv, secs, qc.select_score(pv, yvai))
    base = models_dir / "tfidf_lr"  # candidate 1: reuse its saved probabilities, no re-evaluation
    pv1 = pd.read_parquet(base / "validation.parquet")
    assert (pv1["number"].to_numpy() == va["number"].to_numpy()).all()
    pv1p = pv1[classes].to_numpy()
    sel1 = qc.select_score(pv1p, yvai)
    te = t1.load_split(
        derived, splits_dir, ("test",)
    )  # the first and only place test labels are read
    te = te[te["classifier_eligible"]].reset_index(drop=True)
    yte, ste = t1.gold(te, modelled)
    ytei, sets_m = ev.to_idx(yte, classes), ev.set_mask(ste, classes)
    pt1 = pd.read_parquet(base / "test.parquet")
    assert (pt1["number"].to_numpy() == te["number"].to_numpy()).all()
    pt1p = pt1[classes].to_numpy()
    h1 = ev.hit_arrays(yte, ste, pt1p, classes)
    out: dict[str, Any] = {
        "conformal_selection": {"tfidf_lr_v1": sel1},
        "paired_vs_candidate_1": {},
    }
    out["conformal_test"] = {
        "tfidf_lr_v1": qc.final_sets(pv1p, yvai, pt1p, ytei, sets_m, sel1["selected"], True)
    }
    for mid, (f, model, pv, secs, selc) in fitted.items():
        t0 = time.perf_counter()
        pt = t1.proba_full(model, f.transform(te["text"]), classes)
        infer_ms = (time.perf_counter() - t0) * 1000 / len(te)
        pred = t1.save_proba(models_dir / mid / "test.parquet", te["number"], pt, classes)
        test = ev.evaluate("test", True, yte, ste, pt, classes, n_boot)
        f1 = ev.f1_per_class(ytei, pt.argmax(axis=1), len(classes))
        h = ev.hit_arrays(yte, ste, pt, classes)
        test |= {
            "config": dict(zip(("C", "class_weight", "title_sep", "paths"), todo[mid], strict=True)),
            "per_class_f1": {c: (None if np.isnan(v) else float(v)) for c, v in zip(classes, f1, strict=True)},
            "timing": {"train_seconds": secs, "inference_ms_per_issue": infer_ms,
                       "artifact_mb": len(pickle.dumps((f, model))) / 1e6},
        }  # fmt: skip
        writer.add(mid, "validation", ev.evaluate("validation", False, yva, sva, pv, classes, n_boot), models_dir / mid / "validation.parquet")  # fmt: skip
        writer.add(mid, "test", test, pred, final=True)
        out["conformal_selection"][mid] = selc
        out["conformal_test"][mid] = qc.final_sets(
            pv, yvai, pt, ytei, sets_m, selc["selected"], True
        )
        out["paired_vs_candidate_1"][mid] = {
            k: ev.paired_diff(lambda i, a=h[k]: float(a[i].mean()), lambda i, b=h1[k]: float(b[i].mean()), len(te), n_boot)
            for k in ("top1", "top3", "anyvalid_top1")
        }  # fmt: skip
    c1 = json.loads(Path(c1_report).read_text(encoding="utf-8"))
    out["candidate_1_reused"] = c1["models"]["tfidf_lr_v1"]["test"] | {
        "source": Path(c1_report).name
    }
    out["statement"] = ("Test numbers seen BEFORE the conformal score was chosen: candidate 1's non-randomised APS "
                        "result (set size 12.9 of 22) and a verifier's peek that LAC / randomised APS give much smaller sets; "
                        "the score family was widened after that peek and the choice was then made on validation only. "
                        "C=64 was at the grid edge of candidate 1, hence the extended C grid here.")  # fmt: skip
    writer.data.update({"design": DESIGN, "conformal_design": qc.DESIGN, "grid": grid["rows"], "selection": grid["selection"], **out})  # fmt: skip
    writer.save()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Candidate 2: TF-IDF + LR with richer features")
    ap.add_argument("--repo", default="kubernetes/kubernetes")
    ap.add_argument("--budget-seconds", type=float, default=420)
    ap.add_argument("--final-test", action="store_true")
    a = ap.parse_args(argv)
    slug, root, rep = a.repo.replace("/", "__"), Path(r"D:\triageiq-data"), Path("reports")
    t1.lower_priority()
    with threadpool_limits(limits=4):
        if a.final_test:
            r = final_test(root / "derived" / slug, root / "splits" / slug, root / "models" / slug,
                           rep / f"qp1_conformal_features_{slug}_2026-10.json", rep / f"qp0_split_manifest_{slug}_2026-10.json",
                           rep / f"qp1_tfidf_lr_{slug}_2026-10.json")  # fmt: skip
            print(
                json.dumps({k: v["top3"]["point"] for k, v in r["paired_vs_candidate_1"].items()})
            )
        else:
            s = run_grid(
                root / "derived" / slug,
                root / "splits" / slug,
                root / "models" / slug,
                a.budget_seconds,
            )
            print(
                json.dumps(
                    {
                        "rows": len(s["rows"]),
                        "complete": s["complete"],
                        "selection": s.get("selection"),
                    }
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
