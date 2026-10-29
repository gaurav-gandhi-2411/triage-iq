"""R8: fine-tuned dense retriever for vscode duplicates. Local driver (data, Kaggle, validation).

Design and success criterion: docs/Q5A_FINETUNE_PREREG_2026-10.md (committed before any run).
Subcommands: ``build-dataset`` (train pairs + BM25 hard negatives -> PRIVATE Kaggle dataset),
``push`` (PRIVATE GPU kernel, ONE attempt, a quota error is reported not retried), ``validate``
(VALIDATION metrics from the downloaded outputs; there is deliberately no test stage here, the
single test evaluation needs its own frozen file, as in #345).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qk_kaggle as qk  # noqa: E402
import qp2_vscode_dup_dense as dd  # noqa: E402
import qp2_vscode_dup_eval as qe  # noqa: E402

DS_SLUG, KERNEL = "triageiq-ft-vscode-dup-pairs", "triageiq-dup-finetune-vscode"
TEXTS_DS = "triageiq-texts-microsoft-vscode-issues"
NEG_LO, NEG_HI = 10, 50  # hard negative drawn from BM25 ranks 10-50 (prereg)
CONFIGS = {"F1": (1e-5, False), "F2": (3e-5, False), "F3": (1e-5, True), "F4": (3e-5, True)}
MIN_GAIN = 0.02  # prereg: point gain >= +2.0 points
FROZEN = Path("reports/q5a_vscode_dup_dense_frozen_2026-10.json")
OUT_DIR = Path(r"D:\triageiq-data\models\microsoft__vscode\dup_finetune")


class LeakageError(ValueError):
    """A validation / test (or unknown) issue appears in the training pairs or negatives."""


def train_pairs(dup: pd.DataFrame, split_of: pd.Series, created: pd.Series) -> pd.DataFrame:
    """(source, canonical) rows with BOTH ends in train and the canonical strictly older."""
    s, c = dup["source"].map(split_of), dup["canonical"].map(split_of)
    ok = (s == "train") & (c == "train")
    out = dup.loc[ok, ["source", "canonical"]].drop_duplicates().reset_index(drop=True)
    older = out["canonical"].map(created) < out["source"].map(created)
    return out[older.to_numpy()].reset_index(drop=True)


def components(pairs: pd.DataFrame) -> dict[int, int]:
    """Union-find over duplicate pairs: issue number -> component id (the smallest member)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["source"], pairs["canonical"], strict=True):
        ra, rb = find(int(a)), find(int(b))
        parent[max(ra, rb)] = min(ra, rb)
    return {n: find(n) for n in list(parent)}


def assert_train_only(numbers: set[int], split_of: pd.Series) -> None:
    """Raise LeakageError unless every number is a known train-split issue."""
    bad = sorted(n for n in numbers if split_of.get(n) != "train")
    if bad:
        raise LeakageError(f"{len(bad)} non-train issues in training data, e.g. {bad[:5]}")


def pick_negative(ranked: np.ndarray, rng: np.random.Generator) -> int | None:
    """Pick one hard negative from BM25 ranks NEG_LO..NEG_HI of ``ranked`` (best first).

    Falls back to the whole list when it is shorter than NEG_LO; None when it is empty.
    """
    band = ranked[NEG_LO - 1 : NEG_HI]
    pool = band if len(band) else ranked
    return int(pool[rng.integers(len(pool))]) if len(pool) else None


def hard_negatives(t: dd.Task, pairs: pd.DataFrame, comp: dict[int, int]) -> pd.Series:
    """Per pair: a BM25 hard-negative issue number (train, older than the source, outside group)."""
    q_mat, x_mat = qe.bm25_matrices(t.base, t.fit_mask)  # statistics from train issues only
    queries = {q.source: q for q in qe.build_queries(pairs, t.numbers, t.created)}
    group = np.array([comp.get(int(n), -1) for n in t.numbers])
    rng = np.random.default_rng(qe.SEED)
    negs: dict[int, int | None] = {}
    sources = sorted(pairs["source"].unique())
    for a in range(0, len(sources), qe.CHUNK):
        blk = [queries[int(s)] for s in sources[a : a + qe.CHUNK]]
        scores = (q_mat[[q.pos for q in blk]] @ x_mat.T).toarray()
        for q, sc in zip(blk, scores, strict=True):
            cand = np.flatnonzero(t.fit_mask[: q.m] & (group[: q.m] != comp[q.source]))
            ranked = cand[np.lexsort((-cand, -sc[cand]))][:NEG_HI]  # ties -> more recent
            negs[q.source] = pick_negative(ranked, rng)
    return pairs["source"].map(
        lambda s: None if negs[int(s)] is None else int(t.numbers[negs[int(s)]])
    )


def pair_frame(pairs: pd.DataFrame, neg: pd.Series, comp: dict[int, int]) -> pd.DataFrame:
    """Training file: anchor, pos, neg (issue numbers), comp / neg_comp (group ids; singletons < 0)."""
    df = pd.DataFrame({"anchor": pairs["source"], "pos": pairs["canonical"], "neg": neg})
    df = df.dropna(subset=["neg"]).astype({"neg": "int64"}).reset_index(drop=True)
    df["comp"] = df["anchor"].map(comp).astype("int64")
    df["neg_comp"] = [comp.get(int(n), -int(n)) for n in df["neg"]]
    return df


def build_frames() -> dict[str, pd.DataFrame]:
    """Leak-checked train pair frame + the split table shipped for the in-kernel check."""
    t = dd.Task("validation")  # only used for the chronological table; no validation score here
    d, slug = qe.DATA, qk.slug_of(qe.REPO)
    split_of = pd.concat([pd.read_parquet(d / "splits" / slug / f"{s}.parquet",
                                          columns=["number", "split"])
                          for s in ("train", "validation", "test")]).set_index("number")["split"]  # fmt: skip
    created = pd.Series(t.created, index=t.numbers)
    dup = pd.read_parquet(d / "derived_v2" / slug / "duplicate_pairs.parquet",
                          columns=["source", "canonical"])  # fmt: skip
    pairs = train_pairs(dup, split_of, created)
    comp = components(pairs)
    neg = hard_negatives(t, pairs, comp)
    frame = pair_frame(pairs, neg, comp)
    assert_train_only(set(frame["anchor"]) | set(frame["pos"]) | set(frame["neg"]), split_of)
    print({"pairs": len(pairs), "kept_with_neg": len(frame), "groups": len(set(comp.values()))})
    return {"train_pairs": frame, "splits": split_of.rename_axis("number").reset_index()}


def dataset_meta() -> dict[str, Any]:
    """Dataset metadata; private is enforced here and asserted again before any upload."""
    meta = {"title": DS_SLUG, "id": f"{qk.OWNER}/{DS_SLUG}", "licenses": [{"name": "other"}],
            "isPrivate": True}  # fmt: skip
    if meta["isPrivate"] is not True:
        raise RuntimeError("dataset must be private")
    return meta


def kernel_meta() -> dict[str, Any]:
    """Kernel metadata (private GPU script kernel) with this kernel's code file."""
    meta = {**qk.kernel_metadata(qk.OWNER, KERNEL, [f"{qk.OWNER}/{TEXTS_DS}",
                                                    f"{qk.OWNER}/{DS_SLUG}"]),
            "code_file": "dup_finetune.py"}  # fmt: skip
    if meta["is_private"] is not True:
        raise RuntimeError("kernel must be private")
    return meta


def cmd_build_dataset(_: argparse.Namespace) -> None:
    fr = build_frames()
    out = qk.STAGING / DS_SLUG
    out.mkdir(parents=True, exist_ok=True)
    for k, df in fr.items():
        df.to_parquet(out / f"{k}.parquet", index=False)
    meta = dataset_meta()
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    exe = qk.DEFAULT_KAGGLE
    listing = qk.kaggle(exe, ["datasets", "list", "--mine", "--search", DS_SLUG, "--csv"]).stdout
    verb = "version" if meta["id"] in listing else "create"
    extra = ["-m", "refresh"] if verb == "version" else []
    qk.kaggle(exe, ["datasets", verb, "-p", str(out), *extra, "-t"])


def cmd_push(_: argparse.Namespace) -> None:
    """ONE push attempt; the CLI's own message (e.g. GPU quota) is printed and surfaced."""
    stage = qk.STAGING / f"kernel-{KERNEL}"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree("kaggle/kernels/dup_finetune", stage,
                    ignore=shutil.ignore_patterns("kernel-metadata.json", "__pycache__"))  # fmt: skip
    (stage / "kernel-metadata.json").write_text(json.dumps(kernel_meta(), indent=2))
    res = qk.kaggle(qk.DEFAULT_KAGGLE, ["kernels", "push", "-p", str(stage),
                                        "--accelerator", "NvidiaTeslaT4"])  # fmt: skip
    text = (res.stdout + res.stderr).strip()
    print(text)
    if "successfully pushed" not in text.lower():
        raise RuntimeError(f"kernel push not confirmed: {text[-400:]}")


def verdict(diff: float, ci_lo: float) -> bool:
    """Prereg success: point gain >= +2.0 points AND paired CI lower bound > 0."""
    return diff >= MIN_GAIN and ci_lo > 0


def select_ft(cands: dict[str, dict]) -> str:
    """Highest validation R@5; ties -> MRR -> dense before RRF, smaller k, no hard neg, smaller lr."""

    def key(i: str) -> tuple:
        c = cands[i]["cfg"]
        lr, hard = CONFIGS[c["config"]]
        return (-cands[i]["res"]["recall@5"]["k"], -cands[i]["res"]["mrr"],
                c["kind"] == "rrf", c.get("k", 0), hard, lr)  # fmt: skip

    return sorted(cands, key=key)[0]


def cmd_validate(a: argparse.Namespace) -> None:
    """VALIDATION metrics of every finished config vs the frozen #345 winner (same queries)."""
    out, res_path = Path(a.out), Path(a.result)
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    t = dd.Task("validation")
    ids = pd.read_parquet(out / "ids.parquet")
    if not (ids["number"].to_numpy()[t.order] == t.numbers).all():
        raise ValueError("kernel id order differs from the Task table")
    sc_bm = t.sparse_scorer(frozen["config"]["bm25"])
    ref_ranks, ref_pp = dd.rank_with(t.queries, sc_bm, dd.Scorer(t.emb, t.emb))[10]
    ref = dd.summarize(t.queries, ref_ranks, ref_pp)
    if abs(ref["recall@5"]["rate"] - frozen["validation_recall@5"]) > 1e-9:
        raise ValueError("frozen #345 winner not reproduced on validation")
    cands: dict[str, dict] = {}
    done = sorted(p.name[:2] for p in out.glob("F?_emb.npy"))
    for cid in done:
        t._emb = np.load(out / f"{cid}_emb.npy")[t.order].astype(np.float32)
        dense = dd.Scorer(t._emb, t._emb)
        sets = {"dense": dd.rank_with(t.queries, dense)["x"]}
        sets |= {f"rrf{k}": v for k, v in dd.rank_with(t.queries, sc_bm, dense).items()}
        for name, (r, pp) in sets.items():
            cfg = {"config": cid, "kind": "dense" if name == "dense" else "rrf",
                   **({} if name == "dense" else {"k": int(name[3:])})}  # fmt: skip
            cands[f"{cid}_{name}"] = {"cfg": cfg, "res": dd.summarize(t.queries, r, pp), "ranks": r}
            print(f"{cid}_{name}: R@5={cands[f'{cid}_{name}']['res']['recall@5']['rate']:.4f}")
    win = select_ft(cands)
    pd5 = dd.paired(cands[win]["ranks"], ref_ranks)["recall@5"]
    rep = {"design": "docs/Q5A_FINETUNE_PREREG_2026-10.md", "split": "validation",
           "n_queries": len(t.queries), "configs_done": done, "selected": win,
           "reference_H_k10": ref,
           "candidates": {i: {"config": c["cfg"], **c["res"]} for i, c in cands.items()},
           "paired_selected_minus_H_k10": dd.paired(cands[win]["ranks"], ref_ranks),
           "success": verdict(pd5["diff"], pd5["bootstrap95"][0])}  # fmt: skip
    res_path.write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print("selected", win, "gain", pd5, "success", rep["success"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-dataset").set_defaults(fn=cmd_build_dataset)
    sub.add_parser("push").set_defaults(fn=cmd_push)
    v = sub.add_parser("validate")
    v.add_argument("--out", default=str(OUT_DIR))
    v.add_argument("--result", required=True, help="where to write the validation JSON (scratch)")
    v.set_defaults(fn=cmd_validate)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
