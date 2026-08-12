"""D3: mine hard negatives for the fine-tune, same method as scripts/d2_mine_train_negatives.py,
pointed at the expanded mining-precision training pools (scripts/d3_assert_leakage_guard.py's
TASKS) instead of D1's original pools.

Leakage guard: candidate corpus for negative mining is restricted to TRAINING-pool issues only
(scripts/d3_assert_leakage_guard.py re-asserted as a pre-flight gate here, same as D2).

Reads:  reports/mining_precision_train_pool_{task}.json
        reports/d1_eval_set_{task}.json          (disjointness pre-flight only, unchanged)
        data/processed/issues_{repo}.parquet
Writes: data/d3_hard_negatives_{task}.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from d3_assert_leakage_guard import TASKS, assert_task_disjoint
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
MAX_BODY = 512
NEGATIVES_PER_PAIR = 5

REPO_BY_TASK = {
    "vscode_duplicate": "microsoft_vscode",
    "k8s_related": "kubernetes_kubernetes",
    "k8s_related_valid_subset": "kubernetes_kubernetes",
    "k8s_related_fullcorpus_negs": "kubernetes_kubernetes",
}
REPORTS = Path("reports")
DATA = Path("data")


def _build_text(title: object, body: object) -> str:
    t = (str(title) if title is not None else "").strip()
    b = (str(body) if body is not None else "").strip()[:MAX_BODY]
    return f"{t}. {b}"


def mine_task(task: str, model: SentenceTransformer, full_corpus_negatives: bool = False) -> pd.DataFrame:
    disjoint = assert_task_disjoint(task)
    logger.info("[%s] leakage pre-flight: %s", task, disjoint)

    train_file, eval_file = TASKS[task]
    repo = REPO_BY_TASK[task]
    train_pairs = json.loads((REPORTS / train_file).read_text(encoding="utf-8"))

    train_issue_nums = {int(p["query_number"]) for p in train_pairs} | {
        int(p["original_number"]) for p in train_pairs
    }
    corpus = pd.read_parquet(DATA / "processed" / f"issues_{repo}.parquet")

    if full_corpus_negatives:
        # D3a candidate-B test: candidate pool for NEGATIVE mining is the full live corpus (this
        # is what ADR-0048 flagged as untested -- eval ranks against the full ~30k-issue corpus,
        # but D2/D3 always mined negatives from the 792-issue training-pool-restricted space).
        # Eval-set issues are still excluded from the candidate pool -- negatives are a different
        # leakage surface than positives (scripts/d3_assert_leakage_guard.py only checks positive
        # query/original pairs), but exposing the model to eval issues' text as training-time
        # negatives would be a real, avoidable leakage path, so it's excluded explicitly here.
        eval_pairs = json.loads((REPORTS / eval_file).read_text(encoding="utf-8"))
        eval_issue_nums = {int(p["query_number"]) for p in eval_pairs} | {
            int(p["original_number"]) for p in eval_pairs
        }
        pool = corpus[~corpus["number"].isin(eval_issue_nums)].reset_index(drop=True)
        logger.info(
            "[%s] FULL-CORPUS candidate pool: %d issues (%d corpus issues minus %d eval-set "
            "issues excluded)",
            task, len(pool), len(corpus), len(eval_issue_nums),
        )
    else:
        pool = corpus[corpus["number"].isin(train_issue_nums)].reset_index(drop=True)
        logger.info(
            "[%s] restricted candidate corpus: %d/%d train-pool issues found in processed corpus",
            task, len(pool), len(train_issue_nums),
        )

    pool_nums = pool["number"].astype(int).to_numpy()
    pool_texts = [_build_text(t, b) for t, b in zip(pool["title"], pool["body_clean"], strict=True)]
    num_to_text = dict(zip(pool_nums.tolist(), pool_texts, strict=True))

    embs = model.encode(
        pool_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    num_to_idx = {n: i for i, n in enumerate(pool_nums.tolist())}

    pos_by_query: dict[int, set[int]] = {}
    for p in train_pairs:
        q, o = int(p["query_number"]), int(p["original_number"])
        pos_by_query.setdefault(q, set()).add(o)
        pos_by_query.setdefault(o, set()).add(q)

    unique_queries = sorted({int(p["query_number"]) for p in train_pairs})
    query_idx = [num_to_idx[q] for q in unique_queries if q in num_to_idx]
    query_embs = embs[query_idx]
    k_search = min(NEGATIVES_PER_PAIR + len(pool_nums), len(pool_nums))
    scores_all, indices_all = index.search(query_embs, k_search)

    negs_by_query: dict[int, list[tuple[int, int, float]]] = {}
    for qnum, scores, indices in zip(
        [q for q in unique_queries if q in num_to_idx], scores_all, indices_all, strict=True
    ):
        excluded = pos_by_query.get(qnum, set()) | {qnum}
        negs: list[tuple[int, int, float]] = []
        rank = 0
        for score, idx in zip(scores, indices, strict=True):
            if idx < 0:
                continue
            neg_num = int(pool_nums[idx])
            if neg_num in excluded:
                continue
            rank += 1
            negs.append((neg_num, rank, float(score)))
            if len(negs) >= NEGATIVES_PER_PAIR:
                break
        negs_by_query[qnum] = negs

    records: list[dict] = []
    for p in train_pairs:
        q, o = int(p["query_number"]), int(p["original_number"])
        for neg_num, neg_rank, neg_score in negs_by_query.get(q, []):
            records.append(
                {
                    "repo": repo, "query_number": q, "original_number": o,
                    "neg_number": neg_num, "neg_rank": neg_rank, "neg_score": neg_score,
                    "neg_text": num_to_text.get(neg_num, ""),
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        "[%s] mined %d hard-neg records for %d train pairs (%d pairs had zero negatives found)",
        task, len(df), len(train_pairs),
        sum(1 for p in train_pairs if not negs_by_query.get(int(p["query_number"]))),
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task", choices=list(TASKS), default=None,
        help="mine a single task only; default mines every task in TASKS (existing behavior)",
    )
    ap.add_argument(
        "--full-corpus-negatives", action="store_true",
        help="D3a candidate-B test: mine negatives from the full corpus (minus eval-set issues) "
        "instead of the training-pool-restricted candidate space. Requires --task.",
    )
    args = ap.parse_args()
    if args.full_corpus_negatives and not args.task:
        raise SystemExit("--full-corpus-negatives requires --task")

    # Restricted-pool mining (default, existing behavior) is small (<=3.3k issues) and stays on
    # CPU as before. Full-corpus mining embeds ~30k issues -- CPU would take well over an hour,
    # so use GPU when available (same RTX 3070 d3_train.py already trains on; encoding is a
    # deterministic forward pass, not the stochastic training this project seeds for reproducibility).
    device = "cuda" if args.full_corpus_negatives and torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    logger.info("encoding device: %s", device)
    DATA.mkdir(exist_ok=True)
    for task in [args.task] if args.task else TASKS:
        df = mine_task(task, model, full_corpus_negatives=args.full_corpus_negatives)
        out = DATA / f"d3_hard_negatives_{task}.parquet"
        df.to_parquet(out, index=False)
        logger.info("[%s] wrote %d rows -> %s", task, len(df), out)


if __name__ == "__main__":
    main()
