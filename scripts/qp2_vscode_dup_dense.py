"""vscode duplicate retrieval, part 2 (runbook R8): BM25 variants, dense BGE, RRF.

Pre-registered in docs/Q5A_DENSE_PREREG_2026-10.md (committed before any number of this part).
Reuses the task, time-respecting index, tie rule and metrics of scripts/qp2_vscode_dup_eval.py.
Stages: ``validation`` runs the whole grid, selects on validation R@5 and writes the frozen
config + SHA-256; ``test`` (needs --final-test and the committed --frozen-sha256) evaluates the
frozen winner and the B2 baseline once on the same test queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qk_kaggle  # noqa: E402
import qp2_vscode_dup_eval as qe  # noqa: E402

EMB_DIR = qe.DATA / "kaggle_out" / "vscode"
OUT_VAL = Path("reports/q5a_vscode_dup_dense_validation_2026-10.json")
OUT_FROZEN = Path("reports/q5a_vscode_dup_dense_frozen_2026-10.json")
OUT_TEST = Path("reports/q5a_vscode_dup_dense_test_2026-10.json")
RRF_KS = (10, 30, 60)
RAW_CHARS, MAX_ERR_LINES, MAX_ERR_LEN = 20_000, 5, 300

_CAND = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Za-z_]\w*")
_CAMEL = re.compile(r"[a-z0-9][A-Z]")
_ERR = re.compile(
    r"\b(?:error|exception|failed|failure|cannot|can't|unable|unhandled|fatal|traceback"
    r"|ENOENT|EACCES)\b",
    re.IGNORECASE,
)


def code_tokens(raw: str) -> list[str]:
    """Code-like tokens (``_``, camelCase or dotted path): whole (dots -> ``_``) + sub-words."""
    out: list[str] = []
    for tok in dict.fromkeys(_CAND.findall(raw[:RAW_CHARS])):  # ordered unique
        if "_" not in tok and "." not in tok and not _CAMEL.search(tok):
            continue
        out.append(tok.replace(".", "_").lower())
        spaced = re.sub(
            r"([A-Z]+)([A-Z][a-z])", r"\1 \2", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tok)
        )
        out += [p.lower() for p in re.split(r"[_.\s]+", spaced) if len(p) >= 2]
    return out


def error_lines(raw: str) -> list[str]:
    """First MAX_ERR_LINES non-empty lines (<= MAX_ERR_LEN chars) matching the error regex."""
    lines = (ln.strip() for ln in raw[:RAW_CHARS].splitlines())
    hits = [ln for ln in lines if ln and len(ln) <= MAX_ERR_LEN and _ERR.search(ln)]
    return hits[:MAX_ERR_LINES]


def variant_texts(
    base: list[str], titles: list[str], raw: list[str], w: int, ids: bool, errs: bool
) -> list[str]:
    """Variant text: title repeated w times, plus optional identifier / error-line extras."""
    out = []
    for b, t, r in zip(base, titles, raw, strict=True):
        s = (t + ". ") * (w - 1) + b
        if ids:
            s += " " + " ".join(code_tokens(r))
        if errs:
            s += " " + " ".join(error_lines(r))
        out.append(s)
    return out


def rrf_scores(rank_a: np.ndarray, rank_b: np.ndarray, k: int) -> np.ndarray:
    """Reciprocal rank fusion of two 1-based rank vectors."""
    return 1.0 / (k + rank_a) + 1.0 / (k + rank_b)


def full_ranks(scores: np.ndarray, m: int) -> np.ndarray:
    """1-based rank of every one of the first m candidates; ties go to the more recent one."""
    rev = scores[:m][::-1]
    order = m - 1 - np.argsort(-rev, kind="stable")  # chronological positions, best first
    r = np.empty(m, dtype=np.float64)
    r[order] = np.arange(1, m + 1)
    return r


class Scorer:
    """Block scorer: ``block(queries)`` -> dense (len, N) score matrix."""

    def __init__(self, q: Any, x: Any) -> None:
        self.q, self.sparse = q, sp.issparse(q)
        self.xt = x.T.tocsc() if self.sparse else x.T

    def block(self, blk: list[qe.Query]) -> np.ndarray:
        rows = self.q[[q.pos for q in blk]]
        s = rows @ self.xt
        return s.toarray() if self.sparse else s


def rank_with(
    queries: list[qe.Query], sc: Scorer, other: Scorer | None = None
) -> dict[Any, tuple[np.ndarray, list[list[int]]]]:
    """Ranks per candidate key. Single scorer -> key 'x'; with ``other`` -> RRF keys k."""
    keys = ["x"] if other is None else list(RRF_KS)
    ranks = {k: np.empty(len(queries), dtype=np.int64) for k in keys}
    per_pair: dict[Any, list[list[int]]] = {k: [] for k in keys}
    for a in range(0, len(queries), qe.CHUNK):
        qe._guard_ram()
        blk = queries[a : a + qe.CHUNK]
        s1 = sc.block(blk)
        s2 = other.block(blk) if other is not None else None
        for i, q in enumerate(blk):
            if s2 is None:
                sets = {"x": s1[i]}
            else:
                r1, r2 = full_ranks(s1[i], q.m), full_ranks(s2[i], q.m)
                sets = {k: rrf_scores(r1, r2, k) for k in keys}
            for k, s in sets.items():
                rs = [qe.target_rank(s, q.m, t) for t in q.targets]
                per_pair[k].append(rs)
                ranks[k][a + i] = min(rs, default=qe.INF)
    return {k: (ranks[k], per_pair[k]) for k in keys}


def summarize(queries: list[qe.Query], ranks: np.ndarray, per_pair: list[list[int]]) -> dict:
    """Report block for one candidate (same fields as qp2_vscode_dup_eval.evaluate)."""
    idx = qe.boot_idx(len(queries))
    res = qe.recall_table(ranks)
    for key, v in qe.per_query_values(ranks).items():
        res[key + "_bootstrap95"] = qe.boot_ci(v, idx)
    res["pair_level"] = qe.pair_level(queries, per_pair)
    return res


def paired(ra: np.ndarray, rb: np.ndarray) -> dict[str, Any]:
    """Paired bootstrap difference (a - b) for every metric, shared resamples."""
    idx = qe.boot_idx(len(ra))
    va, vb = qe.per_query_values(ra), qe.per_query_values(rb)
    return {
        k: {"diff": float((va[k] - vb[k]).mean()), "bootstrap95": qe.boot_ci(va[k] - vb[k], idx)}
        for k in va
    }


def sha_json(obj: Any) -> str:
    """SHA-256 of canonical (sorted-key, compact) JSON."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def simplicity(c: dict[str, Any]) -> tuple:
    """Pre-registered simplicity order: B1 < V (w-1+ids+errs, w) < D1 < H (k)."""
    kind = {"tfidf": 0, "bm25": 1, "dense": 2, "rrf": 3}[c["kind"]]
    return (
        kind,
        (c.get("w", 1) - 1 + c.get("ids", 0) + c.get("errs", 0)),
        c.get("w", 1),
        c.get("k", 0),
    )


def select(cands: dict[str, dict]) -> str:
    """Highest validation R@5; ties -> higher MRR -> simpler (``cands``: id -> {cfg, res})."""
    return sorted(
        cands,
        key=lambda i: (-cands[i]["res"]["recall@5"]["k"], -cands[i]["res"]["mrr"],
                       simplicity(cands[i]["cfg"])),
    )[0]  # fmt: skip


class Task:
    """Loaded split-independent data in chronological order, plus per-split queries."""

    def __init__(self, split: str) -> None:
        d, slug = qe.DATA, qk_kaggle.slug_of(qe.REPO)
        tdf = qk_kaggle.build_texts(qe.REPO, "issues", d)
        iss = pd.read_parquet(d / "derived_v2" / slug / "issues.parquet",
                              columns=["number", "created_at", "body"])  # fmt: skip
        assert (tdf["number"].to_numpy() == iss["number"].to_numpy()).all()  # noqa: S101
        ids = pd.read_parquet(EMB_DIR / "microsoft__vscode__issues_ids.parquet")
        assert (ids["number"].to_numpy() == tdf["number"].to_numpy()).all()  # noqa: S101
        o = qe.chrono_order(iss["created_at"], iss["number"])
        self.numbers, self.created = iss["number"].to_numpy()[o], iss["created_at"].to_numpy()[o]
        self.base = tdf["text"].to_numpy()[o].tolist()
        self.titles = tdf["title"].to_numpy()[o].tolist()
        self.raw = (tdf["title"] + "\n" + iss["body"].fillna("")).to_numpy()[o].tolist()
        self.order = o
        spl = pd.concat([pd.read_parquet(d / "splits" / slug / f"{s}.parquet",
                                         columns=["number", "split"])
                         for s in ("train", "validation", "test")]).set_index("number")["split"]  # fmt: skip
        self.fit_mask = np.array([spl.get(int(n)) == "train" for n in self.numbers])
        pairs = pd.read_parquet(d / "splits" / slug / "dup_task_pairs.parquet")
        self.pairs = pairs[pairs["split_of_source"] == split]
        self.queries = qe.build_queries(self.pairs, self.numbers, self.created)
        self._emb: np.ndarray | None = None

    @property
    def emb(self) -> np.ndarray:
        """Kaggle BGE vectors, float32, chronological order (L2-normalised by the recipe)."""
        if self._emb is None:
            e = np.load(EMB_DIR / "microsoft__vscode__issues_emb.npy")
            self._emb = e[self.order].astype(np.float32)
        return self._emb

    def sparse_scorer(self, c: dict[str, Any]) -> Scorer:
        """Scorer for a tfidf / bm25 variant config."""
        if c["kind"] == "tfidf":
            x = qe.tfidf_matrix(self.base, self.fit_mask)
            return Scorer(x, x)
        texts = variant_texts(
            self.base, self.titles, self.raw, c["w"], bool(c["ids"]), bool(c["errs"])
        )
        q, x = qe.bm25_matrices(texts, self.fit_mask)
        return Scorer(q, x)


def run_candidate(t: Task, c: dict[str, Any]) -> dict[Any, tuple[np.ndarray, list[list[int]]]]:
    """Ranks for one candidate config; RRF configs carry their BM25 variant under ``bm25``."""
    if c["kind"] == "dense":
        return rank_with(t.queries, Scorer(t.emb, t.emb))
    if c["kind"] == "rrf":
        return rank_with(t.queries, t.sparse_scorer(c["bm25"]), Scorer(t.emb, t.emb))
    return rank_with(t.queries, t.sparse_scorer(c))


def grid() -> dict[str, dict[str, Any]]:
    """Pre-registered candidate grid (B1, 12 BM25 variants, D1)."""
    g: dict[str, dict[str, Any]] = {"B1_tfidf": {"kind": "tfidf"}, "D1_dense": {"kind": "dense"}}
    for w in (1, 2, 3):
        for ids in (0, 1):
            for errs in (0, 1):
                g[f"V_w{w}_ids{ids}_err{errs}"] = {
                    "kind": "bm25", "w": w, "ids": ids, "errs": errs
                }  # fmt: skip
    return g


def validation_stage() -> dict[str, Any]:
    """Run the whole pre-registered grid on validation, select, write report + frozen config."""
    t = Task("validation")
    cands: dict[str, dict] = {}
    for cid, cfg in grid().items():
        ranks, pp = run_candidate(t, cfg)["x"]
        cands[cid] = {"cfg": cfg, "res": summarize(t.queries, ranks, pp), "ranks": ranks}
        print(f"{cid}: R@5={cands[cid]['res']['recall@5']['rate']:.4f}", flush=True)
    bm = {i: c for i, c in cands.items() if c["cfg"]["kind"] == "bm25"}
    best_bm = select(bm)
    allk = run_candidate(t, {"kind": "rrf", "bm25": cands[best_bm]["cfg"]})  # all k in one pass
    for k in RRF_KS:
        cfg = {"kind": "rrf", "k": k, "bm25": cands[best_bm]["cfg"]}
        ranks, pp = allk[k]
        cands[f"H_k{k}"] = {"cfg": cfg, "res": summarize(t.queries, ranks, pp), "ranks": ranks}
        print(f"H_k{k}: R@5={cands[f'H_k{k}']['res']['recall@5']['rate']:.4f}", flush=True)
    win = select(cands)
    base = "V_w1_ids0_err0"
    rep = {
        "design": "docs/Q5A_DENSE_PREREG_2026-10.md", "split": "validation",
        "n_queries": len(t.queries), "best_bm25_variant": best_bm, "selected": win,
        "ceiling_share_queries_with_indexed_canonical":
            qe._rate(sum(bool(q.targets) for q in t.queries), len(t.queries)),
        "candidates": {i: {"config": c["cfg"], **c["res"]} for i, c in cands.items()},
        "paired_selected_minus_B2_baseline": paired(cands[win]["ranks"], cands[base]["ranks"]),
    }  # fmt: skip
    frozen = {
        "selected": win, "config": cands[win]["cfg"], "baseline": "B2",
        "baseline_config": cands[base]["cfg"], "bm25": {"k1": qe.K1, "b": qe.B},
        "embedding_sha256": file_sha(EMB_DIR / "microsoft__vscode__issues_emb.npy"),
        "script_sha256": file_sha(Path(__file__)),
        "validation_recall@5": cands[win]["res"]["recall@5"]["rate"],
    }  # fmt: skip
    frozen["config_sha256"] = sha_json({k: v for k, v in frozen.items()})
    OUT_VAL.write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    OUT_FROZEN.write_text(json.dumps(frozen, indent=1, sort_keys=True), encoding="utf-8")
    print("selected", win, "config_sha256", frozen["config_sha256"])
    return rep


def test_stage(frozen_sha: str) -> dict[str, Any]:
    """Single test evaluation of the frozen winner and the B2 baseline on the same queries."""
    frozen = json.loads(OUT_FROZEN.read_text(encoding="utf-8"))
    body = {k: v for k, v in frozen.items() if k != "config_sha256"}
    if sha_json(body) != frozen["config_sha256"] or frozen["config_sha256"] != frozen_sha:
        raise ValueError("frozen config hash mismatch: refusing the test evaluation")
    if file_sha(Path(__file__)) != frozen["script_sha256"]:
        raise ValueError("script changed since the freeze: refusing the test evaluation")
    t = Task("test")
    res: dict[str, Any] = {}
    ranks: dict[str, np.ndarray] = {}
    for name, cfg in (("winner", frozen["config"]), ("baseline_B2", frozen["baseline_config"])):
        key = cfg["k"] if cfg["kind"] == "rrf" else "x"
        r, pp = run_candidate(t, cfg)[key]
        ranks[name], res[name] = r, {"config": cfg, **summarize(t.queries, r, pp)}
    rep = {
        "design": "docs/Q5A_DENSE_PREREG_2026-10.md", "split": "test", "frozen": frozen,
        "n_queries": len(t.queries),
        "ceiling_share_queries_with_indexed_canonical":
            qe._rate(sum(bool(q.targets) for q in t.queries), len(t.queries)),
        "candidates": res, "paired_winner_minus_B2": paired(ranks["winner"], ranks["baseline_B2"]),
    }  # fmt: skip
    OUT_TEST.write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vscode duplicate retrieval, dense + BM25 variants")
    ap.add_argument("--stage", choices=("validation", "test"), default="validation")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--frozen-sha256", default="")
    a = ap.parse_args(argv)
    if a.stage == "test":
        if not (a.final_test and a.frozen_sha256):
            raise qe.TestGuardError("test needs --final-test and the committed --frozen-sha256")
        test_stage(a.frozen_sha256)
    else:
        validation_stage()
    return 0


if __name__ == "__main__":
    sys.exit(main())
