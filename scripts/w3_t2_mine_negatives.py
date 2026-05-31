"""T2: Mine hard negatives for W3 bi-encoder fine-tuning.

For each positive pair in gold_related.parquet, query the pre-built BGE FAISS
top-50 and collect up to NEGATIVES_PER_PAIR hard negatives per pair (those ranked
highest in FAISS but absent from the gold positive set for that query — i.e. the
hardest plausible negatives).

Outputs:
  data/w3_hard_negatives.parquet
    columns: repo, query_number, original_number, neg_number, neg_rank, neg_score

Stop conditions (manual — caller decides):
  >20% sampled negatives are actually related issues → abort T3+T4.
"""
from __future__ import annotations

import logging
import random
import sys

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPOS = ["kubernetes_kubernetes", "microsoft_vscode"]
INDEX_DIR = "data/models/dup_index_{repo}_bge"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
TOP_K_FAISS = 50
NEGATIVES_PER_PAIR = 10  # max per (query, original) pair
MAX_BODY = 512
SPOT_CHECK_N = 20
SPOT_CHECK_SEED = 42


def _build_text(title: str | None, body: str | None) -> str:
    t = (title or "").strip()
    b = (body or "").strip()[:MAX_BODY]
    return f"{t}. {b}"


def mine_repo(
    repo: str,
    gold: pd.DataFrame,
    issues: pd.DataFrame,
    model: SentenceTransformer,
) -> pd.DataFrame:
    idx_dir = INDEX_DIR.format(repo=repo)
    meta = joblib.load(f"{idx_dir}/meta.pkl")
    index = faiss.read_index(f"{idx_dir}/index.faiss")
    issue_numbers: np.ndarray = meta["issue_numbers"]
    issue_texts: list[str] = meta["texts"]

    # Build num→text map for retrieved negatives
    num_to_text = {int(issue_numbers[i]): issue_texts[i] for i in range(len(issue_numbers))}

    repo_gold = gold[gold["repo"] == repo].copy()
    logger.info("[%s] %d gold pairs", repo, len(repo_gold))

    # All gold positives per query (bidirectional — exclude both directions)
    pos_by_query: dict[int, set[int]] = {}
    for _, row in repo_gold.iterrows():
        q, o = int(row["query_number"]), int(row["original_number"])
        pos_by_query.setdefault(q, set()).add(o)
        pos_by_query.setdefault(o, set()).add(q)

    # Encode all unique query issues
    unique_q = repo_gold[["query_number", "query_title", "query_body"]].drop_duplicates("query_number")
    qnums = unique_q["query_number"].astype(int).tolist()
    qtexts = [
        _build_text(r["query_title"], r["query_body"])
        for _, r in unique_q.iterrows()
    ]
    logger.info("[%s] Encoding %d unique query texts…", repo, len(qnums))
    embs = model.encode(
        qtexts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    logger.info("[%s] Searching FAISS top-%d…", repo, TOP_K_FAISS)
    scores_all, indices_all = index.search(embs, TOP_K_FAISS)

    records: list[dict] = []
    for qnum, scores, indices in zip(qnums, scores_all, indices_all):
        # Set of issue numbers to exclude (self + gold positives)
        excluded: set[int] = pos_by_query.get(qnum, set()) | {qnum}

        # Hard negatives: highest-ranked non-excluded results
        negs: list[tuple[int, int, float]] = []  # (num, rank_in_filtered, score)
        filtered_rank = 0
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            neg_num = int(issue_numbers[idx])
            if neg_num in excluded:
                continue
            filtered_rank += 1
            negs.append((neg_num, filtered_rank, float(score)))
            if len(negs) >= NEGATIVES_PER_PAIR:
                break

        # One row per (query, original, neg) triple
        originals = repo_gold.loc[repo_gold["query_number"] == qnum, "original_number"].astype(int).tolist()
        for orig_num in originals:
            for neg_num, neg_rank, neg_score in negs:
                records.append({
                    "repo": repo,
                    "query_number": qnum,
                    "original_number": orig_num,
                    "neg_number": neg_num,
                    "neg_rank": neg_rank,
                    "neg_score": neg_score,
                    "neg_text": num_to_text.get(neg_num, ""),
                })

    df = pd.DataFrame(records)
    logger.info(
        "[%s] Mined %d hard-neg records (%d pairs, median %.1f neg/pair)",
        repo, len(df), len(repo_gold),
        df.groupby(["query_number", "original_number"]).size().median(),
    )
    return df


def spot_check(
    hard_negs: pd.DataFrame,
    gold: pd.DataFrame,
    issues_map: dict[str, pd.DataFrame],
    n: int = SPOT_CHECK_N,
    seed: int = SPOT_CHECK_SEED,
) -> None:
    """Print N random hard negatives with context for manual review."""
    rng = random.Random(seed)
    sample_idx = rng.sample(range(len(hard_negs)), min(n, len(hard_negs)))
    sample = hard_negs.iloc[sample_idx].copy()

    print("\n" + "=" * 80)
    print(f"SPOT-CHECK: {n} random hard negatives (seed={seed})")
    print("Mark as TP (actually related) or TN (genuinely unrelated)")
    print("Stop-condition: >20% TP -- abort T3+T4")
    print("=" * 80)

    # Build gold positive lookup for annotation context
    gold_pairs = set(
        zip(gold["query_number"].astype(int), gold["original_number"].astype(int))
    )

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        repo = row["repo"]
        qn, on, nn = int(row["query_number"]), int(row["original_number"]), int(row["neg_number"])
        issues = issues_map[repo]

        def get_issue(num: int) -> tuple[str, str]:
            rows = issues[issues["number"] == num]
            if rows.empty:
                return ("(not found)", "")
            r = rows.iloc[0]
            return str(r.get("title", "")), str(r.get("body_clean", ""))[:200]

        q_title, q_body = get_issue(qn)
        o_title, o_body = get_issue(on)
        n_title, n_body = get_issue(nn)

        print(f"\n[{i}/{n}] repo={repo}  neg_rank={int(row['neg_rank'])}  neg_score={row['neg_score']:.4f}")
        print(f"  QUERY   #{qn}: {q_title[:80]}")
        print(f"  POSITIVE #{on}: {o_title[:80]}")
        print(f"  HARDNEG  #{nn}: {n_title[:80]}")
        print(f"  neg body snippet: {n_body[:120]}")


def main() -> None:
    gold = pd.read_parquet("data/gold_related.parquet")
    logger.info("Gold pairs: %d", len(gold))

    issues_map = {
        "kubernetes_kubernetes": pd.read_parquet("data/processed/issues_kubernetes_kubernetes.parquet"),
        "microsoft_vscode": pd.read_parquet("data/processed/issues_microsoft_vscode.parquet"),
    }

    logger.info("Loading BGE model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    dfs = [mine_repo(repo, gold, issues_map[repo], model) for repo in REPOS]
    result = pd.concat(dfs, ignore_index=True)

    out = "data/w3_hard_negatives.parquet"
    result.to_parquet(out, index=False)
    logger.info("Saved %d records → %s", len(result), out)

    per_pair = result.groupby(["repo", "query_number", "original_number"]).size()
    logger.info(
        "Neg/pair stats: median=%.1f  p5=%.1f  p95=%.1f  min=%d  max=%d",
        per_pair.median(), per_pair.quantile(0.05), per_pair.quantile(0.95),
        per_pair.min(), per_pair.max(),
    )
    logger.info(
        "Score distribution: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        result["neg_score"].mean(), result["neg_score"].std(),
        result["neg_score"].min(), result["neg_score"].max(),
    )

    spot_check(result, gold, issues_map)


if __name__ == "__main__":
    main()
