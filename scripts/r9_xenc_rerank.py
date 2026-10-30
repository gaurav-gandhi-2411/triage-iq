"""R9: cross-encoder reranker over the frozen R8 retriever for vscode duplicates. Local driver.

Design and success criterion: docs/Q5B_XENC_RERANK_PREREG_2026-10.md (committed before any run).
Subcommands: ``build-dataset`` (train groups with hard negatives mined from the frozen R8
retriever + validation candidate lists -> PRIVATE Kaggle dataset) and ``push`` (PRIVATE GPU
kernel, ONE attempt, a quota error is reported not retried). No test query is touched here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qk_kaggle as qk  # noqa: E402
import qp2_vscode_dup_dense as dd  # noqa: E402
import qp2_vscode_dup_eval as qe  # noqa: E402
import r8_dup_finetune as r8  # noqa: E402

DS_SLUG, KERNEL = "triageiq-xenc-vscode-dup", "triageiq-xenc-rerank-vscode"
N_LIST, RRF_K = 50, 10  # prereg: rerank the top 50 of the R8 winner (RRF k=10)
NEG_TOP, NEG_POOL, N_NEG = 40, 30, 7  # prereg: top-40, minus own group, first 30, draw 7
REF_HITS_R5 = 524  # prereg: the frozen R8 winner's validation R@5 hits (524 / 1,156)
R8_DIR = r8.OUT_DIR
OUT_DIR = Path(r"D:\triageiq-data\models\microsoft__vscode\xenc_rerank")


@dataclass
class Retrieved:
    """Top-n chronological positions of one query (best first) + the full rank of each target."""

    pos: np.ndarray
    target_ranks: list[int]


def retrieve(
    queries: list[qe.Query], sc_bm: dd.Scorer, dense: dd.Scorer, n: int
) -> list[Retrieved]:
    """Frozen R8 retriever (RRF k=10 of BM25 + dense, same tie rule as ``dd.rank_with``)."""
    out: list[Retrieved] = []
    for a in range(0, len(queries), qe.CHUNK):
        qe._guard_ram()
        blk = queries[a : a + qe.CHUNK]
        s1, s2 = sc_bm.block(blk), dense.block(blk)
        for i, q in enumerate(blk):
            fused = dd.rrf_scores(dd.full_ranks(s1[i], q.m), dd.full_ranks(s2[i], q.m), RRF_K)
            r = dd.full_ranks(fused, q.m)  # 1-based, unique
            top = np.flatnonzero(r <= n)
            out.append(Retrieved(top[np.argsort(r[top])], [int(r[t]) for t in q.targets]))
    return out


def sample_negatives(
    cand_numbers: np.ndarray, own_group: int, comp: dict[int, int], rng: np.random.Generator
) -> list[int] | None:
    """Drop the anchor's duplicate group from the ranked top-NEG_TOP, then draw N_NEG without
    replacement from the first NEG_POOL left; None when fewer than N_NEG are eligible."""
    keep = [int(c) for c in cand_numbers[:NEG_TOP] if comp.get(int(c)) != own_group]
    pool = keep[:NEG_POOL]
    if len(pool) < N_NEG:
        return None
    return [pool[j] for j in rng.choice(len(pool), N_NEG, replace=False)]


def build_groups(
    t: dd.Task, pairs: pd.DataFrame, comp: dict[int, int], ret: dict[int, Retrieved]
) -> pd.DataFrame:
    """One group per train pair: anchor, pos, neg0..neg6 (issue numbers). Dropped pairs counted."""
    rng = np.random.default_rng(qe.SEED)
    rows, dropped = [], 0
    for s, c in zip(pairs["source"], pairs["canonical"], strict=True):
        negs = sample_negatives(t.numbers[ret[int(s)].pos], comp[int(s)], comp, rng)
        if negs is None:
            dropped += 1
            continue
        rows.append([int(s), int(c), *negs])
    print({"pairs": len(pairs), "groups": len(rows), "dropped_few_negatives": dropped})
    return pd.DataFrame(rows, columns=["anchor", "pos", *[f"neg{i}" for i in range(N_NEG)]])


def candidate_frame(t: dd.Task, queries: list[qe.Query], ret: list[Retrieved]) -> pd.DataFrame:
    """Validation candidate lists: (query, cand, ret_rank) rows, query-major, best first. No labels."""
    q_col, c_col, r_col = [], [], []
    for q, r in zip(queries, ret, strict=True):
        q_col += [int(t.numbers[q.pos])] * len(r.pos)
        c_col += [int(n) for n in t.numbers[r.pos]]
        r_col += list(range(1, len(r.pos) + 1))
    return pd.DataFrame({"query": q_col, "cand": c_col, "ret_rank": r_col})


def check_r8_artifacts(frozen: dict[str, Any]) -> None:
    """The F4 embedding used for retrieval must be the one pinned in the R8 freeze."""
    if dd.file_sha(R8_DIR / "F4_emb.npy") != frozen["embedding_sha256"]:
        raise r8.FrozenMismatchError("F4 embedding differs from the R8 freeze: stopping")


def build_frames() -> dict[str, pd.DataFrame]:
    """Leak-checked train groups, validation candidate lists and the split table."""
    frozen = json.loads(r8.FROZEN_FT.read_text(encoding="utf-8"))
    check_r8_artifacts(frozen)
    t = dd.Task("validation")  # chronological tables + validation queries; no labels leave here
    t._emb = np.load(R8_DIR / "F4_emb.npy")[t.order].astype(np.float32)
    sc_bm, dense = t.sparse_scorer(frozen["config"]["bm25"]), dd.Scorer(t.emb, t.emb)
    d, slug = qe.DATA, qk.slug_of(qe.REPO)
    split_of = pd.concat([pd.read_parquet(d / "splits" / slug / f"{s}.parquet",
                                          columns=["number", "split"])
                          for s in ("train", "validation", "test")]).set_index("number")["split"]  # fmt: skip
    created = pd.Series(t.created, index=t.numbers)
    dup = pd.read_parquet(d / "derived_v2" / slug / "duplicate_pairs.parquet",
                          columns=["source", "canonical"])  # fmt: skip
    pairs = r8.train_pairs(dup, split_of, created)
    comp = r8.components(pairs)
    tq = qe.build_queries(pairs, t.numbers, t.created)
    ret = dict(zip((q.source for q in tq), retrieve(tq, sc_bm, dense, NEG_TOP), strict=True))
    groups = build_groups(t, pairs, comp, ret)
    r8.assert_train_only(set(groups.to_numpy().ravel().tolist()), split_of)
    vret = retrieve(t.queries, sc_bm, dense, N_LIST)
    hits = sum(min((*r.target_ranks, qe.INF)) <= 5 for r in vret)
    if hits != REF_HITS_R5 or len(t.queries) != 1156:
        raise ValueError(f"R8 winner not reproduced on validation: {hits}/{len(t.queries)}")
    cand = candidate_frame(t, t.queries, vret)
    ceil = {k: sum(min((*r.target_ranks, qe.INF)) <= k for r in vret) for k in (30, 50)}
    print({"val_queries": len(t.queries), "r5_hits": hits, "ceiling_hits": ceil})
    return {"train_groups": groups, "val_candidates": cand,
            "splits": split_of.rename_axis("number").reset_index()}  # fmt: skip


def dataset_meta() -> dict[str, Any]:
    """Dataset metadata; private is enforced here and asserted again before any upload."""
    meta = {"title": DS_SLUG, "id": f"{qk.OWNER}/{DS_SLUG}", "licenses": [{"name": "other"}],
            "isPrivate": True}  # fmt: skip
    if meta["isPrivate"] is not True:
        raise RuntimeError("dataset must be private")
    return meta


def kernel_meta() -> dict[str, Any]:
    """Kernel metadata (private GPU script kernel) with this kernel's code file."""
    meta = {**qk.kernel_metadata(qk.OWNER, KERNEL, [f"{qk.OWNER}/{r8.TEXTS_DS}",
                                                    f"{qk.OWNER}/{DS_SLUG}"]),
            "code_file": "xenc_rerank.py"}  # fmt: skip
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
    shutil.copytree("kaggle/kernels/xenc_rerank", stage,
                    ignore=shutil.ignore_patterns("kernel-metadata.json", "__pycache__"))  # fmt: skip
    (stage / "kernel-metadata.json").write_text(json.dumps(kernel_meta(), indent=2))
    res = qk.kaggle(qk.DEFAULT_KAGGLE, ["kernels", "push", "-p", str(stage),
                                        "--accelerator", "NvidiaTeslaT4"])  # fmt: skip
    text = (res.stdout + res.stderr).strip()
    print(text)
    if "successfully pushed" not in text.lower():
        raise RuntimeError(f"kernel push not confirmed: {text[-400:]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-dataset").set_defaults(fn=cmd_build_dataset)
    sub.add_parser("push").set_defaults(fn=cmd_push)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
