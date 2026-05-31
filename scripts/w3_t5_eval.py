"""T5: Evaluate fine-tuned BGE bi-encoder vs baseline on the test split.

Protocol:
  1. Rebuild FAISS index using the fine-tuned model (same full corpus as baseline).
  2. Eval on test split (all test-split pairs per repo).
  3. Eval on n=100 seed=42 sample (same protocol as W1.3 reranker screening for
     comparability with prior results in reports/related_issue_results.json).
  4. Bootstrap 95% CI on delta R@5 vs baseline BGE-base.
  5. Decision logic:
       ≥3pp R@5 BOTH repos, CI lower bound >0 → PASS (Track A success)
       ≥3pp point estimate but CI lower bound <0 → escalate to n=300
       <3pp or degradation → PARTIAL/FAILURE per repo

Reads:
  reports/w3_t4_val_results.json     — winner model dir(s)
  data/w3_split.parquet              — test split pairs
  data/gold_related.parquet          — full gold (for n=100 sample protocol)
  data/processed/issues_*.parquet    — full corpus for FAISS rebuild

Outputs:
  reports/w3_t5_eval_results.json    — full eval table
  data/models/bge_finetuned_*_index/ — rebuilt FAISS indexes
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
MAX_BODY = 512
TOP_K = 20  # max candidates to retrieve
EVAL_K_VALUES = [1, 5, 10, 20]
N_SAMPLE = 100        # W1.3 comparable screening sample per repo
N_BOOTSTRAP = 2000    # bootstrap iterations for CI
N300_THRESHOLD = 3    # pp point estimate that triggers n=300 if CI lower bound <0

BASELINE_R5 = {
    "kubernetes_kubernetes": 0.4102,
    "microsoft_vscode": 0.3674,
}

REPO_INDEX_ALIAS = {
    "kubernetes_kubernetes": "k8s",
    "microsoft_vscode": "vsc",
}


def build_text(title: str | None, body: str | None) -> str:
    t = (title or "").strip()
    b = (body or "").strip()[:MAX_BODY]
    return f"{t}. {b}"


def build_faiss_index(
    model: SentenceTransformer,
    repo: str,
    out_dir: str,
) -> tuple[faiss.IndexFlatIP, np.ndarray, list[str]]:
    """Embed full repo corpus and build IndexFlatIP. Returns (index, numbers, texts)."""
    df = pd.read_parquet(
        f"data/processed/issues_{repo}.parquet",
        columns=["number", "title", "body_clean"],
    )
    texts = [build_text(r["title"], r["body_clean"]) for _, r in df.iterrows()]
    numbers = df["number"].astype(int).values

    logger.info("[%s] Encoding %d issues for FAISS rebuild…", repo, len(texts))
    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p / "index.faiss"))
    np.save(str(p / "numbers.npy"), numbers)
    with open(p / "texts.json", "w") as f:
        json.dump(texts, f)
    logger.info("[%s] FAISS index saved → %s  (n=%d, dim=%d)", repo, out_dir, len(texts), dim)
    return index, numbers, texts


def retrieve_batch(
    query_texts: list[str],
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    numbers: np.ndarray,
    k: int,
) -> list[list[int]]:
    """Return list of top-k issue numbers per query."""
    embs = model.encode(
        query_texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    scores_all, indices_all = index.search(embs, k)
    results = []
    for indices in indices_all:
        results.append([int(numbers[i]) for i in indices if i >= 0])
    return results


def recall_at_k(hits: list[bool], k: int) -> float:
    """R@k = fraction of queries where positive appears in top-k retrieved."""
    total = len(hits)
    if total == 0:
        return 0.0
    return sum(h[:k] for h in hits) / total  # h[:k] is a list of bools


def bootstrap_ci(values: list[float], n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float]:
    """Return (lo, hi) 95% CI via bootstrap."""
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return lo, hi


def eval_on_pairs(
    pairs: pd.DataFrame,
    repo: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    numbers: np.ndarray,
) -> dict:
    """Compute recall@k metrics on a set of (query, positive) pairs."""
    repo_pairs = pairs[pairs["repo"] == repo].copy()
    if repo_pairs.empty:
        return {}

    query_texts = [
        build_text(r["query_title"], r["query_body"])
        for _, r in repo_pairs.iterrows()
    ]
    query_nums = repo_pairs["query_number"].astype(int).tolist()
    positive_nums = repo_pairs["original_number"].astype(int).tolist()

    # Retrieve k+1 so we can exclude the query issue itself (matches baseline behaviour)
    top_k_results = retrieve_batch(query_texts, model, index, numbers, k=max(EVAL_K_VALUES) + 1)

    # For each query, did the positive appear in top-k (self excluded)?
    hit_lists: list[list[bool]] = []
    for top_k, pos_num, q_num in zip(top_k_results, positive_nums, query_nums):
        filtered = [n for n in top_k if n != q_num][:max(EVAL_K_VALUES)]
        hit_lists.append([n == pos_num for n in filtered])

    result = {
        "n_pairs": len(repo_pairs),
        "repo": repo,
    }
    for k in EVAL_K_VALUES:
        hits_k = [any(h[:k]) for h in hit_lists]
        result[f"recall_at_{k}"] = float(np.mean(hits_k))

    r5_hits = [any(h[:5]) for h in hit_lists]
    lo, hi = bootstrap_ci([float(h) for h in r5_hits])
    result["r5_ci_lo"] = lo
    result["r5_ci_hi"] = hi

    return result


def sample_gold(gold: pd.DataFrame, repo: str, n: int, seed: int) -> pd.DataFrame:
    """Exact same sampling as W1.3 fast benchmark for comparability."""
    repo_gold = gold[gold["repo"] == repo].copy()
    rng = random.Random(seed)
    idxs = rng.sample(range(len(repo_gold)), min(n, len(repo_gold)))
    return repo_gold.iloc[idxs]


def main() -> None:
    # Load config from T4 output
    with open("reports/w3_t4_val_results.json") as f:
        t4 = json.load(f)

    winner = t4["winner"]
    winner_dirs = t4["winner_model_dirs"]
    logger.info("T4 winner: %s", winner)

    gold = pd.read_parquet("data/gold_related.parquet")
    split_df = pd.read_parquet("data/w3_split.parquet")

    # Test pairs — split_df already carries all gold text columns from T3
    test_pairs = split_df[split_df["split"] == "test"].copy()

    all_results: dict = {"winner": winner, "repos": {}}

    for repo in ["kubernetes_kubernetes", "microsoft_vscode"]:
        alias = REPO_INDEX_ALIAS[repo]

        # Determine model dir
        if winner == "combined":
            model_dir = winner_dirs["combined"]
        else:
            model_dir = winner_dirs.get(alias, winner_dirs.get(repo))

        logger.info("[%s] Loading fine-tuned model from %s", repo, model_dir)
        model = SentenceTransformer(model_dir)

        # Rebuild FAISS index
        index_out_dir = f"data/models/bge_finetuned_{alias}_index"
        index, numbers, _ = build_faiss_index(model, repo, index_out_dir)

        # A: Test split evaluation
        test_result = eval_on_pairs(test_pairs, repo, model, index, numbers)
        logger.info(
            "[%s] Test split R@5=%.4f  R@1=%.4f  (n=%d)",
            repo, test_result.get("recall_at_5", 0), test_result.get("recall_at_1", 0),
            test_result.get("n_pairs", 0),
        )

        # B: n=100 seed=42 sample (W1.3 comparable protocol)
        sample_pairs = sample_gold(gold, repo, N_SAMPLE, SEED)
        sample_result = eval_on_pairs(sample_pairs, repo, model, index, numbers)
        logger.info(
            "[%s] n=%d sample R@5=%.4f  (baseline=%.4f  delta=%.4f)",
            repo, N_SAMPLE,
            sample_result.get("recall_at_5", 0),
            BASELINE_R5[repo],
            sample_result.get("recall_at_5", 0) - BASELINE_R5[repo],
        )

        delta_r5 = sample_result.get("recall_at_5", 0) - BASELINE_R5[repo]
        ci_lo = sample_result.get("r5_ci_lo", 0) - BASELINE_R5[repo]
        ci_hi = sample_result.get("r5_ci_hi", 0) - BASELINE_R5[repo]

        # Decision logic
        if delta_r5 >= N300_THRESHOLD / 100 and ci_lo > 0:
            verdict = "PASS"
        elif delta_r5 >= N300_THRESHOLD / 100 and ci_lo <= 0:
            verdict = "ESCALATE_n300"
        elif delta_r5 < 0:
            verdict = "REGRESSION"
        else:
            verdict = "BELOW_GATE"

        logger.info(
            "[%s] delta=%.4f  CI=[%.4f, %.4f]  verdict=%s",
            repo, delta_r5, ci_lo, ci_hi, verdict,
        )

        all_results["repos"][repo] = {
            "test_split": test_result,
            "n100_sample": sample_result,
            "baseline_r5": BASELINE_R5[repo],
            "delta_r5_point": delta_r5,
            "delta_r5_ci_lo": ci_lo,
            "delta_r5_ci_hi": ci_hi,
            "verdict": verdict,
        }

    # Overall verdict
    verdicts = [v["verdict"] for v in all_results["repos"].values()]
    if all(v == "PASS" for v in verdicts):
        overall = "TRACK_A_SUCCESS"
    elif "ESCALATE_n300" in verdicts:
        overall = "ESCALATE_n300"
    elif "REGRESSION" in verdicts:
        overall = "TRACK_A_FAILURE"
    elif all(v == "PASS" for v in verdicts):
        overall = "TRACK_A_SUCCESS"
    else:
        # Check partial: at least one PASS
        if any(v == "PASS" for v in verdicts):
            overall = "PARTIAL_PASS"
        else:
            overall = "BELOW_GATE"

    all_results["overall_verdict"] = overall
    logger.info("=== OVERALL: %s ===", overall)

    out = "reports/w3_t5_eval_results.json"
    Path("reports").mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Full results → %s", out)

    # Print ablation table
    print("\n=== W3 Track A Ablation Table ===")
    print(f"{'Repo':<30} {'Metric':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>8} {'CI 95%':>18} {'Verdict'}")
    print("-" * 100)
    for repo, r in all_results["repos"].items():
        bl = r["baseline_r5"]
        ft = bl + r["delta_r5_point"]
        delta = r["delta_r5_point"]
        ci = f"[{r['delta_r5_ci_lo']:+.4f}, {r['delta_r5_ci_hi']:+.4f}]"
        print(f"{repo:<30} {'R@5':<12} {bl:>10.4f} {ft:>12.4f} {delta:>+8.4f} {ci:>18} {r['verdict']}")
    print()


if __name__ == "__main__":
    main()
