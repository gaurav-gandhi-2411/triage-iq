"""Lever 3 RE-RUN: bge-large-en-v1.5 vs bge-base, corrected harness. Per ADR-0046's audit: every
prior generation of this lever ran against a broken harness (title-only queries and/or the
stale, pre-ADR-0040 char-truncated corpus index) -- it has never once been correctly measured.
This is the free (~5min, no training) rerun ADR-0046 flagged as an outstanding gap.

Same two corrections as scripts/d3_eval_finetuned.py, applied to the ORIGINAL scripts/
lever3_stronger_embedder.py methodology (no fine-tuning, single-swap pretrained embedder only):
  1. Query construction: full title+body, untruncated, from the processed corpus.
  2. Corpus/baseline index: data/models/dup_index_{repo}_bge (current, live-serving-matching,
     token-truncated), not the stale d1_full_corpus_index_{repo}_bge.

Reads:  reports/d1_eval_set_{task}.json
        data/models/dup_index_{repo}_bge
        data/processed/issues_{repo}.parquet
Writes: reports/lever3_stronger_embedder_rerun.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from _retrieval_eval_common import paired_bootstrap_ci  # noqa: E402

from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 2000
K_VALUES = [1, 5, 10]
LARGE_MODEL = "BAAI/bge-large-en-v1.5"

TASKS = {
    "k8s_related": ("kubernetes_kubernetes", "reports/d1_eval_set_k8s_related.json"),
    "vscode_duplicate": ("microsoft_vscode", "reports/d1_eval_set_vscode_duplicate.json"),
}
DATA = Path("data")
MODELS_DIR = DATA / "models"
K8S_CLEAN_EVAL = Path("reports/track2_k8s_clean_eval.json")


def bootstrap_ci_recall(hits: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(hits)
    means = [hits[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hit_vectors(retriever, pairs, body_by_num) -> dict:
    live_numbers = {int(n) for n in retriever.issue_numbers}
    usable = [
        p for p in pairs
        if int(p["query_number"]) in live_numbers and int(p["original_number"]) in live_numbers
    ]
    k_max = max(K_VALUES)
    hit_lists = []
    for row in usable:
        q = int(row["query_number"])
        body = body_by_num.get(q, "")
        query_text = f"{row['query_title']}. {body}"
        results = retriever.retrieve(query_text, k=k_max, exclude_number=q)
        retrieved = [r["number"] for r in results]
        pos = int(row["original_number"])
        hit_lists.append([n == pos for n in retrieved])
    out: dict = {"n_usable": len(usable), "n_total": len(pairs)}
    for k in K_VALUES:
        out[f"hits_at_{k}"] = np.array([float(any(h[:k])) for h in hit_lists])
    out["mrr"] = np.array([1.0 / (h.index(True) + 1) if True in h else 0.0 for h in hit_lists])
    return out


def score(result_key, result, base_vecs, large_vecs) -> None:
    if base_vecs["n_usable"] != large_vecs["n_usable"]:
        raise SystemExit(f"[{result_key}] usable-pair mismatch")
    entry: dict = {"n_pairs": base_vecs["n_total"], "n_evaluated": base_vecs["n_usable"]}
    for k in K_VALUES:
        b, t = base_vecs[f"hits_at_{k}"], large_vecs[f"hits_at_{k}"]
        ci_lo, ci_hi, delta = paired_bootstrap_ci(b, t)
        entry[f"recall_at_{k}"] = {
            "bge_base": float(b.mean()),
            "bge_base_ci95": list(round(c, 4) for c in bootstrap_ci_recall(b)),
            "bge_large": float(t.mean()),
            "bge_large_ci95": list(round(c, 4) for c in bootstrap_ci_recall(t)),
            "delta": round(delta, 4),
            "delta_ci95_paired": [round(ci_lo, 4), round(ci_hi, 4)],
            "excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        }
    result[result_key] = entry


def main() -> None:
    result: dict = {}
    for task, (repo, eval_path) in TASKS.items():
        pairs = json.loads(Path(eval_path).read_text(encoding="utf-8"))
        corpus = pd.read_parquet(DATA / "processed" / f"issues_{repo}.parquet")
        body_by_num = dict(zip(corpus["number"].astype(int), corpus["body_clean"].astype(str), strict=True))

        logger.info("[%s] loading current baseline index (dup_index_%s_bge)", task, repo)
        baseline = SimilarIssueRetriever.load(str(MODELS_DIR / f"dup_index_{repo}_bge"))
        base_vecs = hit_vectors(baseline, pairs, body_by_num)

        logger.info("[%s] building bge-large index over the same corpus", task)
        large = SimilarIssueRetriever(repo=repo, model_key=LARGE_MODEL)
        large.build_index(corpus)
        large_vecs = hit_vectors(large, pairs, body_by_num)

        score("full_eval_set", result.setdefault(task, {}), base_vecs, large_vecs)

        if task == "k8s_related" and K8S_CLEAN_EVAL.exists():
            clean = json.loads(K8S_CLEAN_EVAL.read_text(encoding="utf-8"))["pairs"]
            valid_keys = {
                (int(p["query_number"]), int(p["target_number"])) for p in clean if p["label"] == "VALID"
            }
            valid_pairs = [p for p in pairs if (int(p["query_number"]), int(p["original_number"])) in valid_keys]
            base_valid = hit_vectors(baseline, valid_pairs, body_by_num)
            large_valid = hit_vectors(large, valid_pairs, body_by_num)
            score("valid_subset_66", result[task], base_valid, large_valid)

        r5_key = "valid_subset_66" if "valid_subset_66" in result[task] else "full_eval_set"
        r5 = result[task][r5_key]["recall_at_5"]
        logger.info(
            "[%s] (%s) R@5: bge_base=%.3f -> bge_large=%.3f  delta=%+.3f  CI95=%s  excludes_zero=%s",
            task, r5_key, r5["bge_base"], r5["bge_large"], r5["delta"], r5["delta_ci95_paired"], r5["excludes_zero"],
        )

    out_path = Path("reports/lever3_stronger_embedder_rerun.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
