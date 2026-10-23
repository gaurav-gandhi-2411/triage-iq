"""Validation-only re-run of the issue-embedding experiments on PRODUCTION-recipe vectors.

Why: qp1 (#255), qp3b (#265) and qp4 (#260) used Kaggle BGE vectors of the RAW body, whereas
production embeds `title + ". " + clean_text(body)` (PR #288, decision log D86; cosine to the
production vector: mean 0.968, p5 0.898). A negative result is only trusted if measured on a valid
setup, so this swaps the production-recipe vectors (the candidate index
`similar_issue_index_kubernetes_kubernetes_bge_v2`, 7,733 rows) into the existing scripts and
re-runs their VALIDATION stages, everything else identical.

How (injection, no copied logic): a scratch root is built that mirrors the layout the scripts hard-
code (`<root>/embeddings|derived|derived_v2|splits|models/<slug>`): the read-only inputs are directory
junctions to the real data, the embedding files are the production vectors written in the Kaggle file
format, and `models/` is a private copy of the few inputs needed (tfidf_lr/validation.parquet and
the embedding-independent candidate-1 OOF cache). The scripts' own `select` / `run` functions are
then called unchanged, so nothing under the real data root or the committed reports is written.
Test labels/targets are never requested: only the "validate" stages run (qp4's feature table
necessarily contains test rows, exactly as in the committed run; its rows() guard blocks them).
PR embeddings of the #257 prior-PR hybrid have no production recipe and are not touched.

Usage: python scripts/qp_recheck_production_vectors.py {qp1|qp4|qp3b|report} [--full-grid]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

SLUG = "kubernetes__kubernetes"
DATA = Path(r"D:\triageiq-data")
INDEX_DIR = DATA / "models" / "candidates" / "similar_issue_index_kubernetes_kubernetes_bge_v2"
REPORTS = Path("reports")
OUT_REPORT = REPORTS / "qp_recheck_production_vectors_2026-10.json"
FAISS_HEADER_BYTES = 45  # fourcc 4 + d 4 + ntotal 8 + 2 dummy 16 + trained 1 + metric 4 + vec len 8
MIN_FREE_RAM_GB = 1.5
CONTROL_DIR = "control_kaggle"
CONTROL_KAGGLE = False  # set by --control-kaggle


# ---- vector loading / alignment (unit-tested) ---------------------------------------------
def read_flat_ip_vectors(path: Path, d: int, n: int) -> np.ndarray:
    """Vectors of a serialised faiss IndexFlat* file (float32, row order) without importing faiss.

    Layout: 45-byte header, then n*d float32. Validated against the file size and the fourcc, so a
    non-flat or differently-versioned file fails loudly instead of yielding garbage.
    """
    raw = Path(path).read_bytes()
    if raw[:4] not in (b"IxFI", b"IxFL"):
        raise ValueError(f"{path}: not a faiss IndexFlatIP file (fourcc {raw[:4]!r})")
    if len(raw) != FAISS_HEADER_BYTES + n * d * 4:
        raise ValueError(f"{path}: size {len(raw)} != header + {n}x{d} float32")
    return np.frombuffer(raw, dtype="<f4", offset=FAISS_HEADER_BYTES).reshape(n, d).copy()


def load_production_vectors(index_dir: Path = INDEX_DIR) -> tuple[np.ndarray, np.ndarray]:
    """(issue_numbers, vectors) of the production-recipe candidate index, in row order."""
    import joblib

    nums = np.asarray(joblib.load(Path(index_dir) / "meta.pkl")["issue_numbers"], dtype=np.int64)
    d = int(np.frombuffer((Path(index_dir) / "index.faiss").read_bytes()[4:8], dtype="<i4")[0])
    return nums, read_flat_ip_vectors(Path(index_dir) / "index.faiss", d, len(nums))


def align_vectors(
    numbers: np.ndarray,
    prod_numbers: np.ndarray,
    prod_vecs: np.ndarray,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Rows of `prod_vecs` re-ordered to `numbers` BY ISSUE NUMBER (never by position).

    Duplicate production numbers are an error. A requested number absent from the production set
    raises KeyError listing the missing numbers, unless `fallback` (a len(numbers) x d matrix, e.g.
    the Kaggle vectors for the same numbers) is given, in which case those rows come from it.
    """
    if len(set(prod_numbers.tolist())) != len(prod_numbers):
        raise ValueError("duplicate issue numbers in the production index")
    pos = {int(n): i for i, n in enumerate(prod_numbers)}
    want = [int(n) for n in numbers]
    miss = [i for i, n in enumerate(want) if n not in pos]
    if miss and fallback is None:
        raise KeyError(f"{len(miss)} issue numbers missing from production vectors: {[want[i] for i in miss[:10]]}")  # fmt: skip
    out = np.empty((len(want), prod_vecs.shape[1]), dtype=np.float32)
    for i, n in enumerate(want):
        out[i] = prod_vecs[pos[n]] if n in pos else fallback[i]
    return out


# ---- scratch root --------------------------------------------------------------------------
def _junction(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists():
        return
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True
        )  # noqa: S603, S607
    else:
        link.symlink_to(target, target_is_directory=True)


def build_scratch_root(scratch: Path) -> Path:
    """Scratch root with production vectors in the Kaggle file layout (idempotent)."""
    root = scratch / "root"
    for sub in ("derived", "derived_v2", "splits"):
        _junction(root / sub / SLUG, DATA / sub / SLUG)  # read-only inputs, never written
    emb_dir = root / "embeddings" / SLUG
    if (
        CONTROL_KAGGLE
    ):  # harness control: the original Kaggle vectors must reproduce the committed numbers
        _junction(emb_dir, DATA / "embeddings" / SLUG)
    emb_dir.mkdir(parents=True, exist_ok=True)
    if not CONTROL_KAGGLE and not (emb_dir / f"{SLUG}__issues_emb.npy").exists():
        nums, vecs = load_production_vectors()
        kaggle = DATA / "embeddings" / SLUG
        kids = pd.read_parquet(kaggle / f"{SLUG}__issues_ids.parquet")["number"].to_numpy()
        # every number the scripts can request (all Kaggle ids) must be present: fail closed
        vecs = align_vectors(kids, nums, vecs)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        np.save(emb_dir / f"{SLUG}__issues_emb.npy", vecs)
        pd.DataFrame({"number": kids}).to_parquet(emb_dir / f"{SLUG}__issues_ids.parquet", index=False)  # fmt: skip
    m = root / "models" / SLUG
    (m / "tfidf_lr").mkdir(parents=True, exist_ok=True)
    (m / "qp4").mkdir(parents=True, exist_ok=True)
    real = DATA / "models" / SLUG
    for src, dst in ((real / "tfidf_lr" / "validation.parquet", m / "tfidf_lr" / "validation.parquet"),
                     (real / "qp4" / "oof_probs.npz", m / "qp4" / "oof_probs.npz")):  # fmt: skip
        if not dst.exists():
            dst.write_bytes(src.read_bytes())  # candidate-1 OOF probs are embedding-independent
    return root


def free_ram_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / 2**30


def ram_guard() -> None:
    if free_ram_gb() < MIN_FREE_RAM_GB:
        raise SystemExit(f"free RAM {free_ram_gb():.2f} GB < {MIN_FREE_RAM_GB} GB: stopping")


# ---- experiments ---------------------------------------------------------------------------
def committed(name: str) -> dict[str, Any]:
    return json.loads((REPORTS / f"{name}_{SLUG}_2026-10.json").read_text(encoding="utf-8"))


def _pt(x: Any) -> Any:  # noqa: ANN401
    return x["point"] if isinstance(x, dict) and "point" in x else x


def run_qp1(scratch: Path) -> dict[str, Any]:
    """Experiment 1: qp1_embedding_candidates.select on production vectors (validation only)."""
    import qp1_embedding_candidates as ec
    from threadpoolctl import threadpool_limits

    root = build_scratch_root(scratch)
    a = argparse.Namespace(repo="kubernetes/kubernetes", slug=SLUG, derived=root / "derived_v2" / SLUG,
                           splits=root / "splits" / SLUG, models=root / "models" / SLUG,
                           emb_dir=root / "embeddings" / SLUG, report=scratch / "qp1_report.json",
                           manifest=REPORTS / f"qp0_split_manifest_{SLUG}_2026-10.json")  # fmt: skip
    a.report.unlink(missing_ok=True)
    ram_guard()
    with threadpool_limits(limits=1):
        sel = ec.select(a)
    old = committed("qp1_embedding_candidates")
    keep = ("top3", "top1", "macro_f1", "nll")
    rows = lambda g: {  # noqa: E731
        m: [{"config": r["config"]} | {k: r[k] for k in keep} for r in g[m]] for m in g
    }
    c1 = json.loads(a.report.read_text(encoding="utf-8"))["models"]["tfidf_lr_v1_reference"]["validation"]  # fmt: skip
    c1_old = old["models"]["tfidf_lr_v1_reference"]["validation"]
    best_new, best_old = sel["best"], old["selection"]["per_family_best"]
    cfg_same = {m: best_new[m]["config"] == best_old[m]["config"] for m in best_old}
    below = {m: {"committed": best_old[m]["top3"] < _pt(c1_old["top3"]), "production": best_new[m]["top3"] < _pt(c1["top3"])} for m in ("c3a_lr_bge", "c4_knn_bge", "c3b_mlp_bge")}  # fmt: skip
    vs_c1 = {w: {m: float(r["top3"]) - _pt(c["top3"]) for m, r in bst.items()} for w, bst, c in (("committed", best_old, c1_old), ("production", best_new, c1))}  # fmt: skip
    changed = sel["selected_model"] != old["selected_model"]
    flips = [m for m in vs_c1["committed"] if (vs_c1["committed"][m] > 0) != (vs_c1["production"][m] > 0)]  # fmt: skip
    verdict = (
        f"selection {'would change: ' + old['selected_model'] + ' -> ' + sel['selected_model'] if changed else 'unchanged (' + sel['selected_model'] + ' selected in both)'}; "
        f"c3a/c3b/c4 below c1 on validation top-3 in both: {all(v['committed'] and v['production'] for v in below.values())}; "
        f"per-family best configs identical: {cfg_same}; "
        f"families whose validation top-3 sign vs c1 flips: {flips or 'none'}"
    )  # fmt: skip
    return {
        "candidate1_validation": {k: _pt(c1[k]) for k in ("top3", "top1", "macro_f1")},
        "committed": {
            "selected_model": old["selected_model"],
            "per_family_best": {
                m: {"config": r["config"]} | {k: r[k] for k in keep} for m, r in best_old.items()
            },
            "grid": rows(old["validation_grid"]),
        },  # fmt: skip
        "production": {
            "selected_model": sel["selected_model"],
            "per_family_best": {
                m: {"config": r["config"]} | {k: r[k] for k in keep} for m, r in best_new.items()
            },
            "grid": rows(sel["grid"]),
        },  # fmt: skip
        "families_below_c1_top3": below,
        "top3_minus_c1": vs_c1,
        "verdict": verdict,
    }


def run_qp4(scratch: Path, full_grid: bool) -> dict[str, Any]:
    """Experiment 3: qp4 validation with production vectors (selected hyper-parameters)."""
    import qp4_resolution as q4
    from threadpoolctl import threadpool_limits

    root = build_scratch_root(scratch)
    old = committed("qp4_resolution")
    manifest = REPORTS / f"qp0_split_manifest_{SLUG}_2026-10.json"
    results: dict[str, Any] = {}
    orig_grid = list(q4.GRID)
    try:
        passes = [("grid", orig_grid)] if full_grid else [("M1", [old["selection"]["M1"]]), ("M2", [old["selection"]["M2"]])]  # fmt: skip
        for tag, grid in passes:
            q4.GRID[:] = grid
            rep = scratch / f"qp4_report_{tag}.json"
            rep.unlink(missing_ok=True)
            ram_guard()
            with threadpool_limits(limits=1):
                r = q4.run("validate", root, SLUG, rep, manifest)
            results[tag] = r
    finally:
        q4.GRID[:] = orig_grid
    ids = {"M0a": "qp4_M0a_train_median", "M1": "qp4_M1_hazard", "M2": "qp4_M2_closed_only_regression"}  # fmt: skip
    metrics = ("c_index", "ae_days_median", "ae_days_mean", "ibs_ipcw_train_G", "coverage80")
    table: dict[str, Any] = {}
    for m, mid in ids.items():
        src = results["grid" if full_grid else (m if m != "M0a" else "M1")]
        table[m] = {
            "committed": {
                k: _pt(old["models"][mid]["validation"][k])
                for k in metrics
                if k in old["models"][mid]["validation"]
            },  # fmt: skip
            "production": {
                k: _pt(src["validation"][m][k]) for k in metrics if k in src["validation"][m]
            },  # fmt: skip
        }
    out: dict[str, Any] = {"table": table, "full_grid": full_grid}
    if full_grid:
        out["selection"] = {"committed": old["selection"], "production": results["grid"]["selection"]}  # fmt: skip
    g = lambda m, k, w: table[m][w][k]  # noqa: E731
    beats = {w: g("M2", "ae_days_median", w) < g("M0a", "ae_days_median", w) for w in ("committed", "production")}  # fmt: skip
    ordering = {w: g("M2", "ae_days_median", w) < g("M1", "ae_days_median", w) for w in ("committed", "production")}  # fmt: skip
    changed = beats["committed"] != beats["production"] or ordering["committed"] != ordering["production"]  # fmt: skip
    if full_grid:
        changed = changed or out["selection"]["committed"] != out["selection"]["production"]
    out["verdict"] = ("selection would change: " if changed else "selection unchanged: ") + (
        f"M2 beats naive on validation median AE {beats}; M2 < M1 median AE {ordering}"
        + ("" if full_grid else "; hyper-parameters fixed at the committed selection (grid not re-run)"))  # fmt: skip
    return out


def run_qp3b(scratch: Path) -> dict[str, Any]:
    """Experiment 2: qp3b K1 validation k table with production vectors (needs the qp4 pass)."""
    import qp3b_knn_resolution as q3

    root = build_scratch_root(scratch)
    if not (root / "models" / SLUG / "qp4" / "pred_validation.parquet").exists():
        raise SystemExit("run the qp4 stage first (qp3b reads its feature table and M2 predictions)")  # fmt: skip
    old = committed("qp3b_knn_resolution")
    rep = scratch / "qp3b_report.json"
    rep.unlink(missing_ok=True)
    ram_guard()
    new = q3.run("validate", root, rep, REPORTS / f"qp0_split_manifest_{SLUG}_2026-10.json")
    k1_old = old["models"][q3.IDS["K1"]]["validation"]
    k1_new = new["validation"]["K1"]
    k_old, k_new = old["selected_k"], new["selected_k"]
    verdicts = {}
    for w, k1 in (("committed", k1_old), ("production", k1_new)):
        vs = k1["vs_naive"]
        verdicts[w] = {x: _pt(vs[x]) for x in vs if x.endswith("gain_days") or x == "served"} | {"raw": vs}  # fmt: skip
    served = {w: bool(v.get("served", False)) for w, v in verdicts.items()}
    beats = {w: float(_pt(verdicts[w].get("median_ae_gain_days", 0.0))) > 0 for w in verdicts}
    changed = k_old != k_new or beats["committed"] != beats["production"]
    return {
        "committed": {
            "selected_k": k_old,
            "validation_k_table": old["validation_k_table"],
            "K1_vs_naive": verdicts["committed"]["raw"],
        },  # fmt: skip
        "production": {
            "selected_k": k_new,
            "validation_k_table": new["validation_k_table"],
            "K1_vs_naive": verdicts["production"]["raw"],
        },  # fmt: skip
        "k1_beats_naive_on_validation_median_ae": beats,
        "verdict": ("selection would change: " if changed else "selection unchanged: ")
        + f"k {k_old} -> {k_new}; K1 beats naive on validation median AE {beats}; served flag {served}",
    }


POST_HOC_NOTES = [
    "qp1: C6 vs candidate 1 on validation is 0.9366 vs 0.9407 (production vectors) after +0.008 "
    "with Kaggle vectors; candidate 1's validation top-3 CI is [0.9228, 0.9572] (n=725), so the "
    "earlier edge and the new gap are both inside noise. Read as 'C6 loses its edge', not "
    "'C6 is worse'. (Source: independent verifier, 2026-10.)",
    "qp4: M2's Kaggle-vector validation edge over naive was only ~0.8 d. The Kaggle control here "
    "reproduced the committed M2 only approximately (median AE 34.60 vs committed 33.92; M1 "
    "43.97 vs 42.85; naive 34.63), and the verifier's two runs gave M2 37.21 (production) and "
    "33.83 (Kaggle). Run-to-run wobble of ~0.1-1 d comes from feature columns pca10/pca11 that "
    "differ across runs. On production vectors M2 is robustly ~2.5 d worse than naive on "
    "validation; the earlier Kaggle-vector edge was within noise.",
    "qp3b: conclusion unchanged (K1 does not beat naive on validation); k selection 20 -> 50.",
]


def assemble(scratch: Path) -> dict[str, Any]:
    """Merge the per-experiment JSON dumps written by the stage runs into the committed report."""
    out: dict[str, Any] = {
        "purpose": __doc__.split("\n\n")[0],
        "scope": "VALIDATION ONLY; no test label/target evaluated; no model id tested; no new test numbers",  # fmt: skip
        "vectors": "production recipe `title + '. ' + clean_text(body)` (candidate index bge_v2) vs Kaggle raw-body BGE",  # fmt: skip
    }
    for name in ("qp1", "qp4", "qp4_fullgrid", "qp3b"):
        p = scratch / f"{name}_result.json"
        out[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else "not run"
        c = scratch / CONTROL_DIR / f"{name}_result.json"
        out.setdefault("control_kaggle_vectors", {})[name] = json.loads(c.read_text(encoding="utf-8")) if c.exists() else "not run"  # fmt: skip
    out["verdicts"] = {
        k: v["verdict"] for k, v in out.items() if isinstance(v, dict) and "verdict" in v
    }
    out["base_commit"] = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "origin/main"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    out["post_hoc_notes"] = POST_HOC_NOTES
    out["skipped"] = {"qp257_prior_pr_hybrid": "uses Kaggle PR embeddings; no production recipe exists for PRs"}  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["qp1", "qp4", "qp3b", "report"])
    ap.add_argument("--full-grid", action="store_true", help="qp4: re-run the whole HP grid")
    ap.add_argument(
        "--control-kaggle", action="store_true", help="control run on the Kaggle vectors"
    )
    ap.add_argument("--scratch", type=Path, required=True, help="scratch dir (outside the repo)")
    a = ap.parse_args(argv)
    ap_scratch = a.scratch
    if a.control_kaggle:
        global CONTROL_KAGGLE  # noqa: PLW0603
        CONTROL_KAGGLE, a.scratch = True, a.scratch / CONTROL_DIR
    a.scratch.mkdir(parents=True, exist_ok=True)
    if a.stage == "report":
        OUT_REPORT.write_text(
            json.dumps(assemble(ap_scratch), indent=1, default=str), encoding="utf-8"
        )
        return 0
    res = {"qp1": lambda: run_qp1(a.scratch), "qp4": lambda: run_qp4(a.scratch, a.full_grid),
           "qp3b": lambda: run_qp3b(a.scratch)}[a.stage]()  # fmt: skip
    (a.scratch / f"{a.stage}{'_fullgrid' if a.full_grid else ''}_result.json").write_text(json.dumps(copy.deepcopy(res), indent=1, default=str), encoding="utf-8")  # fmt: skip
    print(a.stage, res["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
