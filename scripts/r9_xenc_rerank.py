"""R9: cross-encoder reranker over the frozen R8 retriever for vscode duplicates. Local driver.

Design and success criterion: docs/Q5B_XENC_RERANK_PREREG_2026-10.md (committed before any run).
Subcommands: ``build-dataset`` (train groups with hard negatives mined from the frozen R8
retriever + validation candidate lists -> PRIVATE Kaggle dataset) and ``push`` (PRIVATE GPU
kernel, ONE attempt, a quota error is reported not retried) and ``validate`` (VALIDATION metrics
of the downloaded scores vs the frozen R8 winner). The test stage (``test-build``,
``test-push``, ``test``) is guarded: it needs ``--final-test`` and the committed
``--frozen-sha256``, verifies every pinned artifact, and refuses once a test result file
exists (the test is evaluated once).
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
FROZEN_X = Path("reports/r9_xenc_rerank_frozen_2026-10.json")
OUT_TEST = Path("reports/r9_xenc_rerank_test_2026-10.json")
TEST_DS_SLUG, TEST_KERNEL = "triageiq-xenc-x4-test", "triageiq-xenc-score-test-vscode"
TEST_KERNEL_DIR = Path("kaggle/kernels/xenc_rerank_test")
X4_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")


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


CONFIGS = {"X1": ("M", 2e-5), "X2": ("M", 6e-5), "X3": ("B", 1e-5), "X4": ("B", 3e-5)}
MODES = ("pure", "rrf10", "rrf60")
MIN_GAIN = 0.02  # prereg: point gain >= +2.0 points


def rerank_ranks(scores: np.ndarray, mode: str) -> np.ndarray:
    """New 1-based rank of each candidate. ``scores`` are cross-encoder logits in retriever order
    (index 0 = retriever rank 1). pure: by score; rrfK: RRF of the score rank and the retriever
    rank. Every tie goes to the better retriever rank (prereg)."""
    n = len(scores)
    ret = np.arange(1, n + 1)
    x = np.empty(n, dtype=np.int64)
    x[np.lexsort((ret, -scores))] = ret  # cross-encoder rank
    fused = -x.astype(float) if mode == "pure" else (
        1.0 / (int(mode[3:]) + x) + 1.0 / (int(mode[3:]) + ret))  # fmt: skip
    new = np.empty(n, dtype=np.int64)
    new[np.lexsort((ret, -fused))] = ret
    return new


def final_ranks(
    ret: list[Retrieved], scores: list[np.ndarray], mode: str, n_rerank: int | None = None
) -> tuple[np.ndarray, list[list[int]]]:
    """Per-query best target rank and per-target ranks in the FULL list. The top ``n_rerank``
    (default all candidates) are reordered; a target beyond them keeps its retriever rank."""
    best = np.empty(len(ret), dtype=np.int64)
    per_pair: list[list[int]] = []
    for i, (r, s) in enumerate(zip(ret, scores, strict=True)):
        k = len(s) if n_rerank is None else min(n_rerank, len(s))
        new = rerank_ranks(np.asarray(s[:k], dtype=float), mode)
        rs = [int(new[t - 1]) if t <= k else t for t in r.target_ranks]
        per_pair.append(rs)
        best[i] = min(rs, default=qe.INF)
    return best, per_pair


def select_x(cands: dict[str, dict]) -> str:
    """Highest validation R@5; ties -> MRR -> pure before fusion, larger k, M before B, lower lr."""
    order = {"pure": 0, "rrf60": 1, "rrf10": 2}

    def key(i: str) -> tuple:
        unit, mode = i.split("_")
        model, lr = CONFIGS[unit]
        return (-cands[i]["res"]["recall@5"]["k"], -cands[i]["res"]["mrr"],
                order[mode], model == "B", lr)  # fmt: skip

    return sorted(cands, key=key)[0]


def verdict(diff: float, ci_lo: float) -> bool:
    """Prereg success: point gain >= +2.0 points AND paired CI lower bound > 0."""
    return diff >= MIN_GAIN and ci_lo > 0


def load_scores(path: Path, cand: pd.DataFrame, ret: list[Retrieved]) -> list[np.ndarray]:
    """Kernel scores split per query; refuses unless rows are exactly the candidate frame's."""
    df = pd.read_parquet(path)
    if len(df) != len(cand) or not (df[["query", "cand"]].to_numpy()
                                    == cand[["query", "cand"]].to_numpy()).all():  # fmt: skip
        raise ValueError(f"{path.name}: rows differ from the candidate lists")
    if not np.isfinite(df["score"].to_numpy()).all():
        raise ValueError(f"{path.name}: non-finite scores")
    cuts = np.cumsum([len(r.pos) for r in ret])[:-1]
    return np.split(df["score"].to_numpy(dtype=np.float64), cuts)


def cmd_validate(a: argparse.Namespace) -> None:
    """VALIDATION metrics of every finished unit vs the frozen R8 winner (same 1,156 queries)."""
    out, res_path = Path(a.out), Path(a.result)
    frozen = json.loads(r8.FROZEN_FT.read_text(encoding="utf-8"))
    check_r8_artifacts(frozen)
    t = dd.Task("validation")
    t._emb = np.load(R8_DIR / "F4_emb.npy")[t.order].astype(np.float32)
    sc_bm, dense = t.sparse_scorer(frozen["config"]["bm25"]), dd.Scorer(t.emb, t.emb)
    ref_ranks, ref_pp = dd.rank_with(t.queries, sc_bm, dense)[RRF_K]
    ref = dd.summarize(t.queries, ref_ranks, ref_pp)
    if ref["recall@5"]["k"] != REF_HITS_R5 or abs(
        ref["recall@5"]["rate"] - frozen["validation_recall@5"]
    ) > 1e-9:  # fmt: skip
        raise ValueError("frozen R8 winner not reproduced on validation")
    ret = retrieve(t.queries, sc_bm, dense, N_LIST)
    cand = candidate_frame(t, t.queries, ret)
    plain = np.array([min((*r.target_ranks, qe.INF)) for r in ret])
    if not (plain == ref_ranks).all():
        raise ValueError("candidate lists disagree with the R8 ranking path")
    units = [u for u in (*CONFIGS, "Z-M", "Z-B") if (out / f"{u}_val_scores.parquet").exists()]
    cands: dict[str, dict] = {}
    for unit in units:
        sc = load_scores(out / f"{unit}_val_scores.parquet", cand, ret)
        for mode in MODES:
            r, pp = final_ranks(ret, sc, mode)
            cid = f"{unit}_{mode}"
            cands[cid] = {"res": dd.summarize(t.queries, r, pp), "ranks": r, "scores": sc}
            print(f"{cid}: R@5={cands[cid]['res']['recall@5']['rate']:.4f}", flush=True)
    grid = {i: c for i, c in cands.items() if not i.startswith("Z-")}
    win = select_x(grid)
    pd5 = dd.paired(cands[win]["ranks"], ref_ranks)["recall@5"]
    unit, mode = win.split("_")
    r30, pp30 = final_ranks(ret, cands[win]["scores"], mode, 30)
    rep = {"design": "docs/Q5B_XENC_RERANK_PREREG_2026-10.md", "split": "validation",
           "n_queries": len(t.queries), "units_done": units, "selected": win,
           "reference_R8_winner": ref,
           "candidates": {i: c["res"] for i, c in cands.items()},
           "paired_selected_minus_R8_winner": dd.paired(cands[win]["ranks"], ref_ranks),
           "selected_top30_rerank_only": dd.summarize(t.queries, r30, pp30),
           "ceiling_hits_R50": int((plain <= N_LIST).sum()),
           "success": verdict(pd5["diff"], pd5["bootstrap95"][0])}  # fmt: skip
    res_path.write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print("selected", win, "gain", pd5, "success", rep["success"])


def check_inputs(frozen: dict[str, Any], out: Path) -> None:
    """Every pinned artifact (X4 weights, R8 freeze + embedding, harness code) must match."""
    r8f = frozen["r8_freeze"]
    checks = {  # lazy: stop at the first mismatch
        **{f"X4/{n}": (lambda n=n: dd.file_sha(out / "X4_model" / n), h)
           for n, h in frozen["x4_files_sha256"].items()},
        "r8_freeze_file": (lambda: r8.sha_lf(r8.FROZEN_FT), r8f["file_sha256_lf"]),
        "r8_config": (lambda: json.loads(r8.FROZEN_FT.read_text(encoding="utf-8"))["config_sha256"],
                      r8f["config_sha256"]),
        "r8_embedding": (lambda: dd.file_sha(R8_DIR / "F4_emb.npy"), r8f["embedding_sha256"]),
        **{f"code/{n}": (lambda n=n: r8.sha_lf(Path(n)), h)
           for n, h in frozen["code_sha256_lf"].items()},
    }  # fmt: skip
    for name, (get, want) in checks.items():
        if get() != want:
            raise r8.FrozenMismatchError(f"pinned artifact mismatch ({name}): refusing the test")


def final_test_guard(a: argparse.Namespace) -> dict[str, Any]:
    """Flags, committed hash, no earlier result, every pinned artifact: only then the test."""
    if not (a.final_test and a.frozen_sha256):
        raise qe.TestGuardError("test stage needs --final-test and the committed --frozen-sha256")
    frozen = json.loads(FROZEN_X.read_text(encoding="utf-8"))
    r8.check_frozen(frozen, a.frozen_sha256)
    if Path(a.result).exists():
        raise qe.TestGuardError(f"{a.result} exists: the test is evaluated once, never re-run")
    check_inputs(frozen, Path(a.out))
    return frozen


@dataclass
class SplitData:
    """Test task + frozen R8 retriever output; the reference reproduction already passed."""

    t: dd.Task
    ref: dict
    ref_ranks: np.ndarray
    ret: list[Retrieved]
    cand: pd.DataFrame


def setup_split(frozen: dict[str, Any]) -> SplitData:
    """Candidate lists from the frozen R8 retriever; abort unless it reproduces R8's test R@5."""
    t = dd.Task("test")
    t._emb = np.load(R8_DIR / "F4_emb.npy")[t.order].astype(np.float32)
    r8cfg = json.loads(r8.FROZEN_FT.read_text(encoding="utf-8"))
    sc_bm, dense = t.sparse_scorer(r8cfg["config"]["bm25"]), dd.Scorer(t.emb, t.emb)
    ref_ranks, ref_pp = dd.rank_with(t.queries, sc_bm, dense)[frozen["rrf_k"]]
    ref = dd.summarize(t.queries, ref_ranks, ref_pp)
    want = frozen["reference_test"]
    if ref["recall@5"]["k"] != want["recall@5_hits"] or len(t.queries) != want["n_queries"]:
        raise ValueError(f"frozen R8 winner not reproduced on test: {ref['recall@5']['k']}")
    ret = retrieve(t.queries, sc_bm, dense, frozen["n_rerank"])
    if not (np.array([min((*r.target_ranks, qe.INF)) for r in ret]) == ref_ranks).all():
        raise ValueError("candidate lists disagree with the R8 ranking path")
    return SplitData(t, ref, ref_ranks, ret, candidate_frame(t, t.queries, ret))


def cmd_test_build(a: argparse.Namespace) -> None:
    """Guarded: test candidate lists (no labels) + frozen X4 weights -> PRIVATE Kaggle dataset."""
    frozen = final_test_guard(a)
    ts = setup_split(frozen)
    if set(ts.cand.columns) != {"query", "cand", "ret_rank"}:
        raise ValueError("candidate frame must carry no label column")
    out = qk.STAGING / TEST_DS_SLUG
    out.mkdir(parents=True, exist_ok=True)
    ts.cand.to_parquet(out / "test_candidates.parquet", index=False)
    shutil.copyfile(FROZEN_X, out / "freeze.json")
    for n in X4_FILES:
        shutil.copyfile(Path(a.out) / "X4_model" / n, out / n)
    meta = {**dataset_meta(), "title": TEST_DS_SLUG, "id": f"{qk.OWNER}/{TEST_DS_SLUG}"}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    exe = qk.DEFAULT_KAGGLE
    listing = qk.kaggle(exe, ["datasets", "list", "--mine", "--search", TEST_DS_SLUG, "--csv"])
    verb = "version" if meta["id"] in listing.stdout else "create"
    extra = ["-m", "test-stage"] if verb == "version" else []
    qk.kaggle(exe, ["datasets", verb, "-p", str(out), *extra, "-t"])
    print({"test_queries": len(ts.t.queries), "pairs": len(ts.cand)})


def cmd_test_push(a: argparse.Namespace) -> None:
    """Guarded: ONE push of the scoring-only PRIVATE kernel; a quota error is reported, not retried."""
    frozen = final_test_guard(a)
    src = TEST_KERNEL_DIR / "xenc_rerank_test.py"
    if r8.sha_lf(src) != frozen["code_sha256_lf"][src.as_posix()]:
        raise r8.FrozenMismatchError("test kernel code differs from the freeze")
    stage = qk.STAGING / f"kernel-{TEST_KERNEL}"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(TEST_KERNEL_DIR, stage, ignore=shutil.ignore_patterns("__pycache__"))
    meta = {**qk.kernel_metadata(qk.OWNER, TEST_KERNEL, [f"{qk.OWNER}/{r8.TEXTS_DS}",
                                                         f"{qk.OWNER}/{TEST_DS_SLUG}"]),
            "code_file": "xenc_rerank_test.py"}  # fmt: skip
    if meta["is_private"] is not True:
        raise RuntimeError("kernel must be private")
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    res = qk.kaggle(qk.DEFAULT_KAGGLE, ["kernels", "push", "-p", str(stage),
                                        "--accelerator", "NvidiaTeslaT4"])  # fmt: skip
    text = (res.stdout + res.stderr).strip()
    print(text)
    if "successfully pushed" not in text.lower():
        raise RuntimeError(f"kernel push not confirmed: {text[-400:]}")


def cmd_test(a: argparse.Namespace) -> None:
    """The single test evaluation: frozen X4_rrf10 vs the frozen R8 winner on the same queries."""
    frozen = final_test_guard(a)
    info = json.loads((Path(a.scores) / "run_info.json").read_text(encoding="utf-8"))
    if info["weights_sha256"] != frozen["x4_files_sha256"]:
        raise r8.FrozenMismatchError("kernel scored with weights other than the frozen X4")
    ts = setup_split(frozen)
    sc = load_scores(Path(a.scores) / "X4_test_scores.parquet", ts.cand, ts.ret)
    ranks, pp = final_ranks(ts.ret, sc, frozen["mode"])
    r30, pp30 = final_ranks(ts.ret, sc, frozen["mode"], 30)
    res = dd.summarize(ts.t.queries, ranks, pp)
    paired_ = dd.paired(ranks, ts.ref_ranks)
    lo, hi = paired_["recall@5"]["bootstrap95"]
    rep = {"design": frozen["design"], "split": "test", "frozen": frozen,
           "n_queries": len(ts.t.queries), "reference_R8_winner": ts.ref,
           "reference_reproduced_R8_test": True, "selected": frozen["selected"],
           "selected_result": res, "paired_selected_minus_R8_winner": paired_,
           "selected_top30_rerank_only": dd.summarize(ts.t.queries, r30, pp30),
           "ceiling_hits_R50": int(sum(min((*r.target_ranks, qe.INF)) <= frozen["n_rerank"]
                                       for r in ts.ret)),
           "kernel_run_info": info,
           "criterion_paired_r5_ci_excludes_zero": bool(lo > 0 or hi < 0),
           "target_80_met": bool(res["recall@5"]["rate"] >= 0.8)}  # fmt: skip
    Path(a.result).write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print("test R@5", res["recall@5"], "paired", paired_["recall@5"])


PARITY_N, LAT_QUERIES, LAT_BATCH, LAT_THREADS = 300, 40, 16, 4
MIN_SPEARMAN = 0.999  # prereg parity bound: kernel (fp16 autocast) vs local CPU fp32 scores


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    ra, rb = (pd.Series(v).rank().to_numpy() for v in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def latency_stats(ms: list[float]) -> dict[str, float]:
    """Median / p95 / mean of per-query latencies in milliseconds."""
    v = np.asarray(ms, dtype=float)
    return {"median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)),
            "mean_ms": float(v.mean()), "n": int(len(v))}  # fmt: skip


def cmd_cpu_checks(a: argparse.Namespace) -> None:
    """Kernel-vs-CPU parity and per-query rerank latency (fp32, CPU) of one trained unit."""
    import time

    import torch  # lazy: the other subcommands and the tests need no torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(LAT_THREADS)
    out, unit = Path(a.out), a.unit
    cand = pd.read_parquet(qk.STAGING / DS_SLUG / "val_candidates.parquet")
    kern = pd.read_parquet(out / f"{unit}_val_scores.parquet")
    if not (kern[["query", "cand"]].to_numpy() == cand[["query", "cand"]].to_numpy()).all():
        raise ValueError("kernel scores differ from the candidate lists")
    t = dd.Task("validation")
    text = {int(n): s[:3000] for n, s in zip(t.numbers, t.base, strict=True)}  # kernel side_text
    tok = AutoTokenizer.from_pretrained(out / f"{unit}_model")
    model = AutoModelForSequenceClassification.from_pretrained(out / f"{unit}_model").float().eval()

    def score(rows: pd.DataFrame) -> np.ndarray:
        res: list[np.ndarray] = []
        for s in range(0, len(rows), LAT_BATCH):
            blk = rows.iloc[s : s + LAT_BATCH]
            enc = tok([text[int(q)] for q in blk["query"]], [text[int(c)] for c in blk["cand"]],
                      padding=True, truncation="longest_first", max_length=512,
                      return_tensors="pt")  # fmt: skip
            with torch.inference_mode():
                res.append(model(**enc).logits.squeeze(-1).numpy())
        return np.concatenate(res)

    rng = np.random.default_rng(qe.SEED)
    idx = np.sort(rng.choice(len(cand), PARITY_N, replace=False))
    local = score(cand.iloc[idx])
    rho = spearman(kern["score"].to_numpy()[idx], local)
    diff = float(np.abs(kern["score"].to_numpy()[idx] - local).max())
    queries = rng.choice(cand["query"].unique(), LAT_QUERIES, replace=False)
    score(cand[cand["query"] == queries[0]].head(LAT_BATCH))  # warm-up, not timed
    lat: dict[str, Any] = {}
    for n in (30, 50):
        ms = []
        for q in queries:
            rows = cand[cand["query"] == q].head(n)
            t0 = time.perf_counter()
            score(rows)
            ms.append(1000 * (time.perf_counter() - t0))
        lat[f"N{n}"] = latency_stats(ms)
    rep = {"unit": unit, "parity": {"n": PARITY_N, "spearman": rho, "max_abs_diff": diff,
                                    "ok": rho >= MIN_SPEARMAN},
           "latency_cpu_fp32": lat, "threads": LAT_THREADS, "batch": LAT_BATCH,
           "includes_tokenization": True}  # fmt: skip
    Path(a.result).write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(rep)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-dataset").set_defaults(fn=cmd_build_dataset)
    sub.add_parser("push").set_defaults(fn=cmd_push)
    v = sub.add_parser("validate")
    v.add_argument("--out", default=str(OUT_DIR))
    v.add_argument("--result", required=True, help="where to write the validation JSON (scratch)")
    v.set_defaults(fn=cmd_validate)
    c = sub.add_parser("cpu-checks")
    c.add_argument("--unit", required=True, choices=list(CONFIGS))
    c.add_argument("--out", default=str(OUT_DIR))
    c.add_argument("--result", required=True, help="where to write the JSON (scratch)")
    c.set_defaults(fn=cmd_cpu_checks)
    for name, fn in (
        ("test-build", cmd_test_build),
        ("test-push", cmd_test_push),
        ("test", cmd_test),
    ):
        f = sub.add_parser(name)
        f.add_argument("--final-test", action="store_true")
        f.add_argument("--frozen-sha256", default="")
        f.add_argument("--out", default=str(OUT_DIR))
        f.add_argument("--result", default=str(OUT_TEST))
        if name == "test":
            f.add_argument("--scores", required=True, help="downloaded kernel output dir")
        f.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
