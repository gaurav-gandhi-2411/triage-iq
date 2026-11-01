"""Offline baselines and served-artifact scores for the public numbers (zero tokens, no selection).

Not model selection and nothing is tuned: these are (a) the most-common-labels top-3 baseline of the
k8s `sig/*` task and (b) the SERVED similarity indexes (pinned by data/models/MANIFEST.sha256, shas
re-checked here) scored on the existing 2026 duplicate-task test split (`dup_task_pairs.parquet`,
the Q5a/ADR-0066 task: query = the later issue of a duplicate pair, hit = a duplicate canonical in the
top k). The served retriever is used as production uses it (SimilarIssueRetriever.load, its own query
instruction, production text recipe with the model tokenizer's token-based body cut), over the
index exactly as shipped: no created-before filter (serving has none), the query's own issue excluded.

A query is ANSWERABLE when at least one of its canonicals is in the served index. The served indexes
are small slices of the corpora, so most test queries are not answerable; the counts are reported
instead of forcing a number. Primary reported rate: Recall@5 over answerable queries with a Wilson
interval; also Recall@5 over all queries (an unanswerable query counts as a miss).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU only: laptop rule, nothing here needs a GPU

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from qp0_fixlink_audit import wilson  # noqa: E402

DATA = Path(r"D:\triageiq-data")
OTHER = "other"
SERVED = {
    "microsoft/vscode": "similar_issue_index_microsoft_vscode_bge",
    "kubernetes/kubernetes": "similar_issue_index_kubernetes_kubernetes_bge_v2",
}
MIN_FREE_KIB = 8_388_608  # the shared-laptop RAM guard (8 GiB), as in the other zero-token jobs


def rate(k: int, n: int) -> dict[str, Any]:
    """k of n with its Wilson 95% interval (None bounds when n == 0)."""
    w = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n if n else None), "wilson95": (list(w) if w else None)}


def majority_topk(train_primary: Sequence[str], test_primary: Sequence[str], k: int) -> dict[str, Any]:
    """Strict top-k most-common-labels baseline: the k most frequent TRAIN labels, `other` excluded.

    Strict as ADR-0070 / the k8s strict re-score: a gold `other` is a miss and a predicted `other` never
    earns a hit, so `other` is never ranked. Ties are broken by label name (deterministic).
    """
    counts = Counter(p for p in train_primary if p != OTHER)
    top = [lbl for lbl, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]
    hits = sum(1 for p in test_primary if p in top)
    return {"labels": top, **rate(hits, len(test_primary))}


@dataclass(frozen=True)
class DupQuery:
    """A duplicate-task query: the later issue `source` and the canonicals that are in the index."""

    source: int
    canonicals_indexed: tuple[int, ...]
    n_canonicals: int


def build_dup_queries(pairs: Any, indexed: set[int], in_table: set[int]) -> tuple[list[DupQuery], dict[str, int]]:
    """Group (source, canonical) pairs by source; keep sources present in the issue table.

    Returns the queries and the counts needed to explain a thin evaluation.
    """
    queries: list[DupQuery] = []
    n_missing = 0
    for source, g in pairs.groupby("source", sort=True):
        s = int(source)
        if s not in in_table:
            n_missing += 1
            continue
        canon = sorted({int(c) for c in g["canonical"]})
        queries.append(DupQuery(s, tuple(c for c in canon if c in indexed and c != s), len(canon)))
    counts = {
        "n_pairs": int(len(pairs)),
        "n_queries": len(queries),
        "n_sources_missing_from_table": n_missing,
        "n_answerable": sum(1 for q in queries if q.canonicals_indexed),
        "n_query_issue_itself_in_index": sum(1 for q in queries if q.source in indexed),
    }
    return queries, counts


def recall_at_k(
    queries: Sequence[DupQuery], retrieve: Callable[[DupQuery], list[int]], ks: tuple[int, ...] = (1, 5, 10)
) -> dict[str, Any]:
    """Recall@k over the answerable queries; `retrieve` returns ranked issue numbers (query excluded)."""
    answerable = [q for q in queries if q.canonicals_indexed]
    ranked = {q.source: retrieve(q) for q in answerable}
    out: dict[str, Any] = {}
    for k in ks:
        hit = sum(1 for q in answerable if set(ranked[q.source][:k]) & set(q.canonicals_indexed))
        out[f"recall@{k}_answerable"] = rate(hit, len(answerable))
        out[f"recall@{k}_all_queries"] = rate(hit, len(queries))
    return out


def check_manifest(index_dir: Path, manifest: Path) -> dict[str, str]:
    """Recompute the index files' sha256 and require them to equal the manifest pins (else raise)."""
    import hashlib

    pins = {}
    for ln in manifest.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) == 2:
            pins[parts[1]] = parts[0]
    out = {}
    for name in ("index.faiss", "meta.pkl"):
        rel = f"data/models/{index_dir.name}/{name}"
        h = hashlib.sha256((index_dir / name).read_bytes()).hexdigest()
        if pins.get(rel) != h:
            raise RuntimeError(f"{rel}: recomputed sha256 {h} != manifest pin {pins.get(rel)}")
        out[name] = h
    return out


def _free_kib() -> int | None:
    """Free RAM in KiB, or None when psutil is not installed (optional, as in qp2_vscode_dup_eval)."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available // 1024)


def score_served_index(repo: str, data: Path = DATA) -> dict[str, Any]:
    """Score one served index on the 2026 test dup task; raises on a sha mismatch or low RAM."""
    import joblib
    import pandas as pd
    import qk_kaggle

    from triage_iq.models.similar_issues import SimilarIssueRetriever, _build_text

    free = _free_kib()
    if free is not None and free < MIN_FREE_KIB:
        raise RuntimeError(f"free RAM below the {MIN_FREE_KIB} KiB guard; defer")
    slug = qk_kaggle.slug_of(repo)
    index_dir = ROOT / "data" / "models" / SERVED[repo]
    shas = check_manifest(index_dir, ROOT / "data" / "models" / "MANIFEST.sha256")
    meta = joblib.load(index_dir / "meta.pkl")
    indexed = {int(n) for n in meta["issue_numbers"]}
    pairs = pd.read_parquet(data / "splits" / slug / "dup_task_pairs.parquet")
    pairs = pairs[pairs["split_of_source"] == "test"]
    texts = qk_kaggle.build_texts(repo, "issues", data)
    queries, counts = build_dup_queries(pairs, indexed, set(texts["number"].astype(int)))
    res: dict[str, Any] = {"repo": repo, "index_dir": index_dir.name, "index_sha256": shas, "n_index_vectors": len(indexed),
                           "split": "test", **counts,
                           "answerable_share_of_queries": rate(counts["n_answerable"], counts["n_queries"])}
    answerable = [q for q in queries if q.canonicals_indexed]
    if not answerable:
        res["note"] = "no query has a canonical in the served index; nothing to score"
        return res
    ret = SimilarIssueRetriever.load(str(index_dir))
    by_num = texts.set_index(texts["number"].astype(int))
    qtext = {}
    for q in answerable:
        row = by_num.loc[q.source]
        qtext[q.source] = _build_text(pd.Series([row["title"]]), pd.Series([row["body_clean"]]), ret.model.tokenizer)[0]
    res["recall"] = recall_at_k(
        queries,
        lambda q: [h["number"] for h in ret.retrieve(qtext[q.source], k=10, exclude_number=q.source)],
    )
    res["embedding_model"] = "BAAI/bge-base-en-v1.5 (the served retriever's own model; CPU)"
    return res


def k8s_sig_majority(data: Path = DATA) -> dict[str, Any]:
    """Most-common-labels top-3 / top-1 for the k8s sig/* task: train frequencies, test n = 1,384."""
    import qp1_tfidf_lr as t1

    derived, splits = data / "derived_v2" / "kubernetes__kubernetes", data / "splits" / "kubernetes__kubernetes"
    tr_all = t1.load_split(derived, splits, ("train",))
    modelled, _ = t1.define_classes(tr_all)
    tr = tr_all[tr_all["classifier_eligible"]].reset_index(drop=True)
    te = t1.load_split(derived, splits, ("test",))
    te = te[te["classifier_eligible"]].reset_index(drop=True)
    ytr, _ = t1.gold(tr, modelled)
    yte, _ = t1.gold(te, modelled)
    return {
        "task": "kubernetes sig/* (ADR-0067), strict, test eligible labelled issues",
        "train_population": "train-split classifier_eligible issues, primary label (modelled classes + other)",
        "n_train": len(ytr),
        "top3": majority_topk(ytr, yte, 3),
        "top1": majority_topk(ytr, yte, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "served_artifacts_offline_scoring_2026-10.json")
    a = ap.parse_args(argv)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()  # noqa: S607
    rep = {
        "statement": __doc__.split("\n\n")[0],
        "base_commit": head,
        "k8s_sig_majority_baseline": k8s_sig_majority(),
        "served_indexes": [score_served_index(r) for r in SERVED],
    }
    a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
