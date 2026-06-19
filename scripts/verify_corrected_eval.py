"""Corrected evaluation: baseline vs fine-tuned on test-split pairs only.

T1 confirmed 66-71% contamination in the n=100 seed=42 sample.
This script computes an honest delta using ONLY test-split pairs (zero training overlap).

Also covers T4: qualitative sanity check (10 random test queries, top-5 comparison).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd

SEED = 42
MAX_BODY = 512
TOP_K = 20
EVAL_K_VALUES = [1, 5, 10, 20]
N_BOOTSTRAP = 2000

REPOS = ["kubernetes_kubernetes", "microsoft_vscode"]
REPO_ALIAS = {"kubernetes_kubernetes": "k8s", "microsoft_vscode": "vsc"}

BASELINE_INDEX = {
    "kubernetes_kubernetes": "data/models/dup_index_kubernetes_kubernetes_bge",
    "microsoft_vscode": "data/models/dup_index_microsoft_vscode_bge",
}
FINETUNED_INDEX = {
    "kubernetes_kubernetes": "data/models/bge_finetuned_k8s_index",
    "microsoft_vscode": "data/models/bge_finetuned_vsc_index",
}


def build_text(title: str | None, body: str | None) -> str:
    t = (title or "").strip()
    b = (body or "").strip()[:MAX_BODY]
    return f"{t}. {b}"


def load_baseline_index(repo: str) -> tuple[faiss.IndexFlatIP, np.ndarray, list[str]]:
    d = BASELINE_INDEX[repo]
    index = faiss.read_index(f"{d}/index.faiss")
    meta = joblib.load(f"{d}/meta.pkl")
    numbers = np.array(meta["issue_numbers"], dtype=np.int64)
    texts = meta["texts"]
    return index, numbers, texts


def load_finetuned_index(repo: str) -> tuple[faiss.IndexFlatIP, np.ndarray, list[str]]:
    d = FINETUNED_INDEX[repo]
    index = faiss.read_index(f"{d}/index.faiss")
    numbers = np.load(f"{d}/numbers.npy").astype(np.int64)
    with open(f"{d}/texts.json") as f:
        texts = json.load(f)
    return index, numbers, texts


def retrieve_by_text(
    query_texts: list[str],
    corpus_texts: list[str],
    numbers: np.ndarray,
    index: faiss.IndexFlatIP,
    k: int,
) -> list[list[int]]:
    """Encode query texts by position lookup in corpus, then FAISS search."""
    text_to_idx = {t: i for i, t in enumerate(corpus_texts)}
    results = []

    # We need embeddings — but we're using pre-built FAISS, so we use stored vectors.
    # Strategy: for each query, find its index position and retrieve its stored embedding,
    # then search. Works when query is in corpus; for queries not in corpus, skip.
    # Since test queries ARE in the full corpus, this is valid.
    embs = []
    valid_pairs: list[tuple[int, int]] = []  # (query_idx_in_list, faiss_pos)
    for qi, qt in enumerate(query_texts):
        pos = text_to_idx.get(qt)
        if pos is None:
            embs.append(None)
        else:
            valid_pairs.append((qi, pos))

    # Reconstruct embeddings from FAISS index
    n_vectors = index.ntotal
    dim = index.d
    all_embs = np.zeros((n_vectors, dim), dtype=np.float32)
    index.reconstruct_n(0, n_vectors, all_embs)

    results_map: dict[int, list[int]] = {}
    for qi, pos in valid_pairs:
        q_emb = all_embs[pos:pos+1]
        scores, idxs = index.search(q_emb, k)
        results_map[qi] = [int(numbers[i]) for i in idxs[0] if i >= 0]

    return [results_map.get(qi, []) for qi in range(len(query_texts))]


def eval_on_test_pairs(
    test_pairs: pd.DataFrame,
    repo: str,
    index: faiss.IndexFlatIP,
    numbers: np.ndarray,
    corpus_texts: list[str],
    gold: pd.DataFrame,
) -> dict:
    """Evaluate on test-split pairs. Self-exclusion applied."""
    repo_pairs = test_pairs[test_pairs["repo"] == repo].copy()
    if repo_pairs.empty:
        return {}

    gold_idx = gold.set_index(["repo", "query_number", "original_number"])
    text_to_pos = {t: i for i, t in enumerate(corpus_texts)}

    # Pre-reconstruct all embeddings once
    n_vectors = index.ntotal
    dim = index.d
    all_embs = np.zeros((n_vectors, dim), dtype=np.float32)
    index.reconstruct_n(0, n_vectors, all_embs)

    hit_lists: list[list[bool]] = []
    skipped = 0
    for _, row in repo_pairs.iterrows():
        key = (row["repo"], int(row["query_number"]), int(row["original_number"]))
        if key not in gold_idx.index:
            skipped += 1
            continue
        g = gold_idx.loc[key]
        if not isinstance(g, pd.Series):
            g = g.iloc[0]

        q_text = build_text(g["query_title"], g["query_body"])
        pos_num = int(row["original_number"])
        q_num = int(row["query_number"])

        pos = text_to_pos.get(q_text)
        if pos is None:
            skipped += 1
            continue

        q_emb = all_embs[pos:pos+1]
        _, idxs = index.search(q_emb, TOP_K + 1)
        retrieved = [int(numbers[i]) for i in idxs[0] if i >= 0]
        filtered = [n for n in retrieved if n != q_num][:TOP_K]
        hit_lists.append([n == pos_num for n in filtered])

    if skipped > 0:
        print(f"  [WARN] Skipped {skipped} pairs (text not found in corpus)")

    n = len(hit_lists)
    if n == 0:
        return {}

    result: dict = {"n_pairs": n, "repo": repo}
    for k in EVAL_K_VALUES:
        hits_k = [any(h[:k]) for h in hit_lists]
        result[f"recall_at_{k}"] = float(np.mean(hits_k))

    r5_hits = [any(h[:5]) for h in hit_lists]
    rng = np.random.default_rng(SEED)
    arr = np.array(r5_hits, dtype=float)
    means = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(N_BOOTSTRAP)]
    result["r5_ci_lo"] = float(np.percentile(means, 2.5))
    result["r5_ci_hi"] = float(np.percentile(means, 97.5))
    return result


def qualitative_check(
    test_pairs: pd.DataFrame,
    repo: str,
    baseline_index: faiss.IndexFlatIP,
    baseline_numbers: np.ndarray,
    baseline_texts: list[str],
    ft_index: faiss.IndexFlatIP,
    ft_numbers: np.ndarray,
    ft_texts: list[str],
    gold: pd.DataFrame,
    n_samples: int = 10,
) -> None:
    """T4: Show top-5 from both models for n random test queries."""
    repo_pairs = test_pairs[test_pairs["repo"] == repo].copy().reset_index(drop=True)
    rng = random.Random(SEED + 1)
    sample_idxs = rng.sample(range(len(repo_pairs)), min(n_samples, len(repo_pairs)))

    gold_idx = gold.set_index(["repo", "query_number", "original_number"])

    # Pre-reconstruct both embeddings
    n_b = baseline_index.ntotal
    dim_b = baseline_index.d
    base_embs = np.zeros((n_b, dim_b), dtype=np.float32)
    baseline_index.reconstruct_n(0, n_b, base_embs)

    n_f = ft_index.ntotal
    dim_f = ft_index.d
    ft_embs = np.zeros((n_f, dim_f), dtype=np.float32)
    ft_index.reconstruct_n(0, n_f, ft_embs)

    base_text_to_pos = {t: i for i, t in enumerate(baseline_texts)}
    ft_text_to_pos = {t: i for i, t in enumerate(ft_texts)}

    print(f"\n=== T4 QUALITATIVE CHECK: {REPO_ALIAS[repo]} ({n_samples} test queries) ===\n")

    for sample_i, idx in enumerate(sample_idxs):
        row = repo_pairs.iloc[idx]
        key = (row["repo"], int(row["query_number"]), int(row["original_number"]))
        if key not in gold_idx.index:
            continue
        g = gold_idx.loc[key]
        if not isinstance(g, pd.Series):
            g = g.iloc[0]

        q_text = build_text(g["query_title"], g["query_body"])
        pos_num = int(row["original_number"])
        q_num = int(row["query_number"])
        pos_title = str(g.get("original_title", ""))[:60]

        print(f"--- Query {sample_i+1}: #{q_num} ---")
        print(f"  Q: {str(g.get('query_title', ''))[:80]}")
        print(f"  TRUE POS: #{pos_num}  \"{pos_title}\"")
        print()

        for label, embs, text_to_pos, numbers in [
            ("BASELINE BGE-base", base_embs, base_text_to_pos, baseline_numbers),
            ("FINE-TUNED BGE", ft_embs, ft_text_to_pos, ft_numbers),
        ]:
            pos = text_to_pos.get(q_text)
            if pos is None:
                print(f"  [{label}] query not in corpus — skip")
                continue
            q_emb = embs[pos:pos+1]
            if label.startswith("BASELINE"):
                _, idxs = baseline_index.search(q_emb, TOP_K + 1)
            else:
                _, idxs = ft_index.search(q_emb, TOP_K + 1)
            top5_nums = [int(numbers[i]) for i in idxs[0] if i >= 0]
            top5_nums = [n for n in top5_nums if n != q_num][:5]
            print(f"  [{label}] top-5:")
            for rank, num in enumerate(top5_nums, 1):
                hit = "*** HIT ***" if num == pos_num else ""
                print(f"    {rank}. #{num:6d}  {hit}")
        print()


def main() -> None:
    gold = pd.read_parquet("data/gold_related.parquet")
    split_df = pd.read_parquet("data/w3_split.parquet")
    test_pairs = split_df[split_df["split"] == "test"].copy()

    print("=== CORRECTED EVALUATION: TEST SPLIT ONLY ===")
    print(f"Test pairs: k8s={len(test_pairs[test_pairs['repo']=='kubernetes_kubernetes'])}, "
          f"vsc={len(test_pairs[test_pairs['repo']=='microsoft_vscode'])}")
    print()

    corrected: dict = {}

    for repo in REPOS:
        alias = REPO_ALIAS[repo]
        print(f"[{alias}] Loading indexes...")
        base_idx, base_nums, base_texts = load_baseline_index(repo)
        ft_idx, ft_nums, ft_texts = load_finetuned_index(repo)

        print(f"[{alias}] Evaluating baseline on test split...")
        baseline_result = eval_on_test_pairs(test_pairs, repo, base_idx, base_nums, base_texts, gold)

        print(f"[{alias}] Evaluating fine-tuned on test split...")
        ft_result = eval_on_test_pairs(test_pairs, repo, ft_idx, ft_nums, ft_texts, gold)

        bl_r5 = baseline_result.get("recall_at_5", 0.0)
        ft_r5 = ft_result.get("recall_at_5", 0.0)
        delta = ft_r5 - bl_r5

        # Bootstrap CI on the delta (paired)
        # Compute per-pair hit vectors for both models
        repo_pairs_df = test_pairs[test_pairs["repo"] == repo].copy()
        gold_idx_df = gold.set_index(["repo", "query_number", "original_number"])

        # Reconstruct base and ft embs for side-by-side comparison
        n_b = base_idx.ntotal
        dim_b = base_idx.d
        base_embs = np.zeros((n_b, dim_b), dtype=np.float32)
        base_idx.reconstruct_n(0, n_b, base_embs)
        base_t2p = {t: i for i, t in enumerate(base_texts)}

        n_f = ft_idx.ntotal
        dim_f = ft_idx.d
        ft_embs = np.zeros((n_f, dim_f), dtype=np.float32)
        ft_idx.reconstruct_n(0, n_f, ft_embs)
        ft_t2p = {t: i for i, t in enumerate(ft_texts)}

        base_hits_r5: list[float] = []
        ft_hits_r5: list[float] = []
        for _, row in repo_pairs_df.iterrows():
            key = (row["repo"], int(row["query_number"]), int(row["original_number"]))
            if key not in gold_idx_df.index:
                continue
            g = gold_idx_df.loc[key]
            if not isinstance(g, pd.Series):
                g = g.iloc[0]
            q_text = build_text(g["query_title"], g["query_body"])
            pos_num = int(row["original_number"])
            q_num = int(row["query_number"])

            b_pos = base_t2p.get(q_text)
            f_pos = ft_t2p.get(q_text)
            if b_pos is None or f_pos is None:
                continue

            # Baseline
            q_emb = base_embs[b_pos:b_pos+1]
            _, idxs = base_idx.search(q_emb, TOP_K + 1)
            top = [int(base_nums[i]) for i in idxs[0] if i >= 0]
            top = [n for n in top if n != q_num][:5]
            base_hits_r5.append(float(pos_num in top))

            # Fine-tuned
            q_emb = ft_embs[f_pos:f_pos+1]
            _, idxs = ft_idx.search(q_emb, TOP_K + 1)
            top = [int(ft_nums[i]) for i in idxs[0] if i >= 0]
            top = [n for n in top if n != q_num][:5]
            ft_hits_r5.append(float(pos_num in top))

        base_arr = np.array(base_hits_r5)
        ft_arr = np.array(ft_hits_r5)
        rng = np.random.default_rng(SEED)
        delta_boots = []
        n = len(base_arr)
        for _ in range(N_BOOTSTRAP):
            idxs = rng.integers(0, n, n)
            delta_boots.append(ft_arr[idxs].mean() - base_arr[idxs].mean())
        ci_lo = float(np.percentile(delta_boots, 2.5))
        ci_hi = float(np.percentile(delta_boots, 97.5))

        print(f"\n[{alias}] CORRECTED TEST-SPLIT RESULTS:")
        print(f"  Baseline   R@5 = {bl_r5:.4f}  (n={baseline_result.get('n_pairs', 0)})")
        print(f"  Fine-tuned R@5 = {ft_r5:.4f}  (n={ft_result.get('n_pairs', 0)})")
        print(f"  Delta          = {delta:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        verdict = "PASS" if delta >= 0.03 and ci_lo > 0 else (
            "ESCALATE_n300" if delta >= 0.03 else ("REGRESSION" if delta < 0 else "BELOW_GATE")
        )
        print(f"  Verdict        = {verdict}")
        print()

        corrected[alias] = {
            "baseline_r5": bl_r5,
            "finetuned_r5": ft_r5,
            "delta": delta,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_pairs": ft_result.get("n_pairs", 0),
            "verdict": verdict,
        }

        # T4: qualitative check
        qualitative_check(
            test_pairs, repo,
            base_idx, base_nums, base_texts,
            ft_idx, ft_nums, ft_texts,
            gold,
        )

    # Summary
    print("\n=== CORRECTED SUMMARY (TEST SPLIT — ZERO TRAINING OVERLAP) ===")
    print(f"{'Repo':<6} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>8} {'CI 95%':>20} {'Verdict'}")
    print("-" * 75)
    for alias, r in corrected.items():
        ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
        print(f"{alias:<6} {r['baseline_r5']:>10.4f} {r['finetuned_r5']:>12.4f} "
              f"{r['delta']:>+8.4f} {ci:>20}  {r['verdict']}")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/w3_corrected_eval_results.json", "w") as f:
        json.dump(corrected, f, indent=2)
    print("\nSaved -> reports/w3_corrected_eval_results.json")


if __name__ == "__main__":
    main()
