"""T5: Evaluate fine-tuned BGE bi-encoder vs baseline on the test split.

Protocol (canonical, post-W3-correction):
  1. Load pre-built baseline FAISS index (dup_index_*_bge, BAAI/bge-base-en-v1.5).
  2. Rebuild fine-tuned FAISS index from the T4 winner model.
  3. Evaluate BOTH models on test-split pairs ONLY (zero training overlap guaranteed
     by assert_eval_disjoint_from_train).
  4. Bootstrap 95% PAIRED CI on delta R@5.
  5. Decision logic:
       >=3pp delta R@5 BOTH repos, CI lower bound >0 -> PASS (Track A success)
       >=3pp point estimate but CI lower bound <=0   -> ESCALATE_n300
       <3pp or degradation                           -> PARTIAL/FAILURE per repo

IMPORTANT: Hardcoded baseline constants are FORBIDDEN here.
Baseline is always computed on the SAME query set and protocol as the fine-tuned
model. A baseline measured on a different sample makes the delta meaningless.
See ADR-0010 correction note (2026-05-31): the original eval used sample_gold()
which sampled from the full gold corpus, contaminating 66-71% of eval pairs with
training data and inflating the reported delta to ~2x the true value.

Reads:
  reports/w3_t4_val_results.json          -- winner model dir(s)
  data/w3_split.parquet                   -- split assignments (test pairs used)
  data/processed/issues_*.parquet         -- full corpus for fine-tuned FAISS rebuild
  data/models/dup_index_*_bge/            -- pre-built baseline FAISS indexes

Outputs:
  reports/w3_t5_eval_results.json         -- full eval table
  data/models/bge_finetuned_*_index/      -- rebuilt fine-tuned FAISS indexes
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
MAX_BODY = 512
TOP_K = 20
EVAL_K_VALUES = [1, 5, 10, 20]
N_BOOTSTRAP = 2000
PP_GATE = 3              # minimum pp delta R@5 for PASS verdict

BASELINE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

REPO_INDEX_ALIAS = {
    "kubernetes_kubernetes": "k8s",
    "microsoft_vscode": "vsc",
}

# Pre-built baseline FAISS indexes — must cover the same corpus as the fine-tuned indexes.
# If the processed corpus changes, rebuild with scripts/03_build_index.py before re-running T5.
BASELINE_INDEX_DIRS = {
    "kubernetes_kubernetes": "data/models/dup_index_kubernetes_kubernetes_bge",
    "microsoft_vscode": "data/models/dup_index_microsoft_vscode_bge",
}


def build_text(title: str | None, body: str | None) -> str:
    t = (title or "").strip()
    b = (body or "").strip()[:MAX_BODY]
    return f"{t}. {b}"


def assert_eval_disjoint_from_train(test_pairs: pd.DataFrame, split_df: pd.DataFrame) -> None:
    """Verify eval pairs have zero overlap with training pairs.

    Raises AssertionError loudly if violated — prevents silent eval contamination.
    This is the regression gate for the ADR-0010 contamination bug.
    """
    train_keys = frozenset(
        zip(
            split_df[split_df["split"] == "train"]["repo"],
            split_df[split_df["split"] == "train"]["query_number"].astype(int),
            split_df[split_df["split"] == "train"]["original_number"].astype(int),
        )
    )
    eval_keys = frozenset(
        zip(
            test_pairs["repo"],
            test_pairs["query_number"].astype(int),
            test_pairs["original_number"].astype(int),
        )
    )
    overlap = eval_keys & train_keys
    if overlap:
        n = len(overlap)
        sample = sorted(overlap)[:5]
        raise AssertionError(
            f"EVAL/TRAIN LEAK: {n} eval pairs found in training set. "
            f"First 5: {sample}. "
            "Eval MUST use only held-out test-split pairs — see ADR-0010 correction note."
        )
    logger.info("Disjoint check PASSED: 0 overlap between test_pairs and training split.")


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

    logger.info("[%s] Encoding %d issues for FAISS rebuild...", repo, len(texts))
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
    logger.info("[%s] FAISS index saved -> %s  (n=%d, dim=%d)", repo, out_dir, len(texts), dim)
    return index, numbers, texts


def load_baseline_index(repo: str) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """Load pre-built baseline BGE FAISS index (W1.1 artifact)."""
    d = BASELINE_INDEX_DIRS[repo]
    index = faiss.read_index(f"{d}/index.faiss")
    meta = joblib.load(f"{d}/meta.pkl")
    numbers = np.array(meta["issue_numbers"], dtype=np.int64)
    logger.info("[%s] Loaded baseline FAISS from %s  (n=%d)", repo, d, len(numbers))
    return index, numbers


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


def eval_on_pairs(
    pairs: pd.DataFrame,
    repo: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    numbers: np.ndarray,
) -> tuple[dict, list[float]]:
    """Compute recall@k on (query, positive) pairs.

    Returns (metrics_dict, per_query_r5_hit_flags) — the hit flags are used for
    paired bootstrap CI so baseline and fine-tuned are compared on the same queries.
    """
    repo_pairs = pairs[pairs["repo"] == repo].copy()
    if repo_pairs.empty:
        return {}, []

    query_texts = [
        build_text(r["query_title"], r["query_body"])
        for _, r in repo_pairs.iterrows()
    ]
    query_nums = repo_pairs["query_number"].astype(int).tolist()
    positive_nums = repo_pairs["original_number"].astype(int).tolist()

    top_k_results = retrieve_batch(query_texts, model, index, numbers, k=max(EVAL_K_VALUES) + 1)

    hit_lists: list[list[bool]] = []
    for top_k, pos_num, q_num in zip(top_k_results, positive_nums, query_nums):
        filtered = [n for n in top_k if n != q_num][:max(EVAL_K_VALUES)]
        hit_lists.append([n == pos_num for n in filtered])

    result: dict = {"n_pairs": len(repo_pairs), "repo": repo}
    for k in EVAL_K_VALUES:
        result[f"recall_at_{k}"] = float(np.mean([any(h[:k]) for h in hit_lists]))

    r5_hits = [float(any(h[:5])) for h in hit_lists]
    return result, r5_hits


def bootstrap_delta_ci(
    base_hits: list[float],
    ft_hits: list[float],
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float]:
    """Paired bootstrap 95% CI on delta R@5 (fine-tuned minus baseline)."""
    rng = np.random.default_rng(seed)
    b = np.array(base_hits)
    f = np.array(ft_hits)
    n = len(b)
    deltas = [f[rng.integers(0, n, n)].mean() - b[rng.integers(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main() -> None:
    with open("reports/w3_t4_val_results.json") as f:
        t4 = json.load(f)

    winner = t4["winner"]
    winner_dirs = t4["winner_model_dirs"]
    logger.info("T4 winner: %s", winner)

    split_df = pd.read_parquet("data/w3_split.parquet")
    test_pairs = split_df[split_df["split"] == "test"].copy()

    # Fail loudly before any eval if test/train overlap exists
    assert_eval_disjoint_from_train(test_pairs, split_df)

    logger.info("Loading baseline model: %s", BASELINE_MODEL_NAME)
    baseline_model = SentenceTransformer(BASELINE_MODEL_NAME)

    # Load fine-tuned model (combined covers both repos)
    if winner == "combined":
        ft_model_dir = winner_dirs["combined"]
    else:
        ft_model_dir = None  # handled per-repo below
    ft_model_combined = SentenceTransformer(ft_model_dir) if ft_model_dir else None

    all_results: dict = {"winner": winner, "repos": {}}

    for repo in ["kubernetes_kubernetes", "microsoft_vscode"]:
        alias = REPO_INDEX_ALIAS[repo]

        # Baseline eval — pre-built dup_index, queries encoded with baseline model
        base_index, base_numbers = load_baseline_index(repo)
        baseline_metrics, base_hits = eval_on_pairs(
            test_pairs, repo, baseline_model, base_index, base_numbers,
        )
        baseline_r5 = baseline_metrics.get("recall_at_5", 0.0)
        logger.info(
            "[%s] Baseline  R@5=%.4f  R@1=%.4f  (n=%d)",
            repo, baseline_r5, baseline_metrics.get("recall_at_1", 0), baseline_metrics.get("n_pairs", 0),
        )

        # Fine-tuned eval — rebuild FAISS with fine-tuned weights
        if ft_model_combined is not None:
            ft_model = ft_model_combined
        else:
            model_dir = winner_dirs.get(alias, winner_dirs.get(repo))
            ft_model = SentenceTransformer(model_dir)

        index_out_dir = f"data/models/bge_finetuned_{alias}_index"
        ft_index, ft_numbers, _ = build_faiss_index(ft_model, repo, index_out_dir)
        ft_metrics, ft_hits = eval_on_pairs(test_pairs, repo, ft_model, ft_index, ft_numbers)
        ft_r5 = ft_metrics.get("recall_at_5", 0.0)
        logger.info(
            "[%s] Fine-tuned R@5=%.4f  R@1=%.4f  (n=%d)",
            repo, ft_r5, ft_metrics.get("recall_at_1", 0), ft_metrics.get("n_pairs", 0),
        )

        delta_r5 = ft_r5 - baseline_r5
        ci_lo, ci_hi = bootstrap_delta_ci(base_hits, ft_hits)

        if delta_r5 >= PP_GATE / 100 and ci_lo > 0:
            verdict = "PASS"
        elif delta_r5 >= PP_GATE / 100 and ci_lo <= 0:
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
            "baseline_metrics": baseline_metrics,
            "finetuned_metrics": ft_metrics,
            "baseline_r5": baseline_r5,
            "finetuned_r5": ft_r5,
            "delta_r5_point": delta_r5,
            "delta_r5_ci_lo": ci_lo,
            "delta_r5_ci_hi": ci_hi,
            "verdict": verdict,
        }

    verdicts = [v["verdict"] for v in all_results["repos"].values()]
    if all(v == "PASS" for v in verdicts):
        overall = "TRACK_A_SUCCESS"
    elif "ESCALATE_n300" in verdicts:
        overall = "ESCALATE_n300"
    elif "REGRESSION" in verdicts:
        overall = "TRACK_A_FAILURE"
    elif any(v == "PASS" for v in verdicts):
        overall = "PARTIAL_PASS"
    else:
        overall = "BELOW_GATE"

    all_results["overall_verdict"] = overall
    logger.info("=== OVERALL: %s ===", overall)

    out = "reports/w3_t5_eval_results.json"
    Path("reports").mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Full results -> %s", out)

    print("\n=== W3 Track A Ablation Table (test split, zero training overlap) ===")
    print(f"{'Repo':<30} {'Metric':<12} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>8} {'CI 95%':>20} {'Verdict'}")
    print("-" * 100)
    for repo, r in all_results["repos"].items():
        bl = r["baseline_r5"]
        ft_v = r["finetuned_r5"]
        delta = r["delta_r5_point"]
        ci = f"[{r['delta_r5_ci_lo']:+.4f}, {r['delta_r5_ci_hi']:+.4f}]"
        print(f"{repo:<30} {'R@5':<12} {bl:>10.4f} {ft_v:>12.4f} {delta:>+8.4f} {ci:>20} {r['verdict']}")
    print()


if __name__ == "__main__":
    main()
