"""T5 (v2, stratified): evaluate fine-tuned BGE vs baseline on the grown corpus, per stratum.

Pre-registered design (ADR-0027, committed BEFORE training ran):
  - Eval strata per repo (from gold_related_v2 stratum column):
      gate    — the powered proxy stratum, CI-GATED:
                k8s = PR-query -> issue-target; vscode = dup_comment pairs
      product — issue->issue product-task stratum, DIRECTIONAL SECONDARY, never gated
      train_only pairs are excluded from all eval (PR targets etc.; train signal only)
  - Gate verdict per repo: PASS iff delta R@5 >= 3pp AND paired-bootstrap 95% CI lower
    bound > 0 on the gate stratum. Product stratum reports delta + CI, no gate.
  - HEADLINE rule (locked): the headline result is the product-task directional deltas on
    BOTH repos, reported alongside the gated proxy CIs. "Proxy improved, product-task
    unproven" is an acceptable, pre-registered outcome — it must not be reported as
    "retrieval improved +X pp".
  - Bootstrap correction: the ADR-0016-era script resampled baseline and fine-tuned hit
    vectors with INDEPENDENT indices (unpaired, wider CI) while calling it paired. v2
    uses a true paired bootstrap (same resample indices, primary) and also reports the
    legacy unpaired CI for comparison with ADR-0016.
  - Baseline is computed live on the SAME v2 corpus index and query sets as the
    fine-tuned model (no hardcoded baselines; ADR-0016 correction kept). Baselines are
    NOT comparable to W3/ADR-0016 numbers: different corpus size, different gold mix.

Reads:
  reports/w3_t4_val_results_v2.json     -- winner model dir(s)
  data/w3_split_v2.parquet              -- split + stratum assignments
  data/processed/issues_*.parquet       -- grown corpus for FAISS builds
  data/models/dup_index_*_bge_v2/       -- baseline v2 indexes (same corpus)

Outputs:
  reports/w3_t5_eval_results_v2.json
  data/models/bge_finetuned_*_v2_index/
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
EVAL_K_VALUES = [1, 5, 10, 20]
N_BOOTSTRAP = 2000
PP_GATE = 3  # minimum pp delta R@5 for a gate-stratum PASS

BASELINE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
REPO_INDEX_ALIAS = {"kubernetes_kubernetes": "k8s", "microsoft_vscode": "vsc"}
BASELINE_INDEX_DIRS = {
    "kubernetes_kubernetes": "data/models/dup_index_kubernetes_kubernetes_bge_v2",
    "microsoft_vscode": "data/models/dup_index_microsoft_vscode_bge_v2",
}
EVAL_STRATA = ("gate", "product")  # train_only never evaluated


def build_text(title: str | None, body: str | None) -> str:
    t = (title or "").strip()
    b = (body or "").strip()[:MAX_BODY]
    return f"{t}. {b}"


def assert_eval_disjoint_from_train(test_pairs: pd.DataFrame, split_df: pd.DataFrame) -> None:
    """Zero overlap between eval pairs and training pairs — ADR-0016 regression gate."""
    train_keys = frozenset(
        zip(
            split_df[split_df["split"] == "train"]["repo"],
            split_df[split_df["split"] == "train"]["query_number"].astype(int),
            split_df[split_df["split"] == "train"]["original_number"].astype(int),
            strict=True,
        )
    )
    eval_keys = frozenset(
        zip(
            test_pairs["repo"],
            test_pairs["query_number"].astype(int),
            test_pairs["original_number"].astype(int),
            strict=True,
        )
    )
    overlap = eval_keys & train_keys
    if overlap:
        raise AssertionError(
            f"EVAL/TRAIN LEAK: {len(overlap)} eval pairs found in training set. "
            f"First 5: {sorted(overlap)[:5]}. See ADR-0016 correction note."
        )
    logger.info("Disjoint check PASSED: 0 overlap between test pairs and training split.")


def assert_issue_level_disjoint(test_pairs: pd.DataFrame, split_df: pd.DataFrame) -> None:
    """No eval-pair ISSUE appears in any training pair (component split invariant)."""
    for repo in test_pairs["repo"].unique():
        train = split_df[(split_df["split"] == "train") & (split_df["repo"] == repo)]
        train_issues = set(train["query_number"].astype(int)) | set(
            train["original_number"].astype(int)
        )
        te = test_pairs[test_pairs["repo"] == repo]
        test_issues = set(te["query_number"].astype(int)) | set(te["original_number"].astype(int))
        leak = train_issues & test_issues
        if leak:
            raise AssertionError(f"[{repo}] issue-level leak: {sorted(leak)[:5]} ...")
    logger.info("Issue-level disjoint check PASSED.")


def build_faiss_index(model: SentenceTransformer, repo: str, out_dir: str):
    df = pd.read_parquet(
        f"data/processed/issues_{repo}.parquet", columns=["number", "title", "body_clean"]
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
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p / "index.faiss"))
    np.save(str(p / "numbers.npy"), numbers)
    logger.info("[%s] FAISS index saved -> %s (n=%d)", repo, out_dir, len(texts))
    return index, numbers


def load_baseline_index(repo: str):
    d = BASELINE_INDEX_DIRS[repo]
    index = faiss.read_index(f"{d}/index.faiss")
    meta = joblib.load(f"{d}/meta.pkl")
    numbers = np.array(meta["issue_numbers"], dtype=np.int64)
    logger.info("[%s] Loaded baseline v2 FAISS from %s (n=%d)", repo, d, len(numbers))
    return index, numbers


def eval_on_pairs(pairs: pd.DataFrame, model, index, numbers) -> tuple[dict, np.ndarray]:
    """Recall@k on (query, positive) pairs; returns per-query R@5 hit flags for pairing."""
    if pairs.empty:
        return {}, np.array([])
    query_texts = [build_text(r["query_title"], r["query_body"]) for _, r in pairs.iterrows()]
    query_nums = pairs["query_number"].astype(int).tolist()
    positive_nums = pairs["original_number"].astype(int).tolist()

    embs = model.encode(
        query_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    _, indices_all = index.search(embs, max(EVAL_K_VALUES) + 1)

    hit_lists: list[list[bool]] = []
    for indices, pos, qn in zip(indices_all, positive_nums, query_nums, strict=True):
        retrieved = [int(numbers[i]) for i in indices if i >= 0]
        filtered = [n for n in retrieved if n != qn][: max(EVAL_K_VALUES)]
        hit_lists.append([n == pos for n in filtered])

    result: dict = {"n_pairs": int(len(pairs))}
    for k in EVAL_K_VALUES:
        result[f"recall_at_{k}"] = float(np.mean([any(h[:k]) for h in hit_lists]))
    r5 = np.array([float(any(h[:5])) for h in hit_lists])
    return result, r5


def paired_bootstrap_ci(base: np.ndarray, ft: np.ndarray) -> tuple[float, float]:
    """TRUE paired bootstrap: same resample indices for both models (primary, v2)."""
    rng = np.random.default_rng(SEED)
    n = len(base)
    d = ft - base
    deltas = [d[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def unpaired_bootstrap_ci_legacy(base: np.ndarray, ft: np.ndarray) -> tuple[float, float]:
    """ADR-0016-era method (independent resamples; wider). Reported for comparison only."""
    rng = np.random.default_rng(SEED)
    n = len(base)
    deltas = [
        ft[rng.integers(0, n, n)].mean() - base[rng.integers(0, n, n)].mean()
        for _ in range(N_BOOTSTRAP)
    ]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main() -> None:
    with open("reports/w3_t4_val_results_v2.json") as f:
        t4 = json.load(f)
    winner = t4["winner"]
    winner_dirs = t4["winner_model_dirs"]
    logger.info("T4 winner: %s", winner)

    split_df = pd.read_parquet("data/w3_split_v2.parquet")
    test_all = split_df[split_df["split"] == "test"].copy()
    test_eval = test_all[test_all["stratum"].isin(EVAL_STRATA)].copy()
    logger.info(
        "test pairs: %d total, %d eval-eligible (%d train_only excluded)",
        len(test_all),
        len(test_eval),
        len(test_all) - len(test_eval),
    )

    assert_eval_disjoint_from_train(test_eval, split_df)
    assert_issue_level_disjoint(test_eval, split_df)

    baseline_model = SentenceTransformer(BASELINE_MODEL_NAME)
    ft_model_combined = (
        SentenceTransformer(winner_dirs["combined"]) if winner == "combined" else None
    )

    results: dict = {"winner": winner, "design": "ADR-0027 pre-registered strata", "repos": {}}

    for repo in ["kubernetes_kubernetes", "microsoft_vscode"]:
        alias = REPO_INDEX_ALIAS[repo]
        base_index, base_numbers = load_baseline_index(repo)
        ft_model = ft_model_combined or SentenceTransformer(
            winner_dirs.get(alias, winner_dirs.get(repo))
        )
        ft_index, ft_numbers = build_faiss_index(
            ft_model, repo, f"data/models/bge_finetuned_{alias}_v2_index"
        )

        repo_result: dict = {"strata": {}}
        for stratum in EVAL_STRATA:
            pairs = test_eval[(test_eval["repo"] == repo) & (test_eval["stratum"] == stratum)]
            base_metrics, base_hits = eval_on_pairs(pairs, baseline_model, base_index, base_numbers)
            ft_metrics, ft_hits = eval_on_pairs(pairs, ft_model, ft_index, ft_numbers)
            if not len(base_hits):
                repo_result["strata"][stratum] = {"n_pairs": 0}
                continue
            delta = float(ft_hits.mean() - base_hits.mean())
            ci_lo, ci_hi = paired_bootstrap_ci(base_hits, ft_hits)
            leg_lo, leg_hi = unpaired_bootstrap_ci_legacy(base_hits, ft_hits)

            entry = {
                "n_pairs": int(len(pairs)),
                "baseline": base_metrics,
                "finetuned": ft_metrics,
                "delta_r5": delta,
                "paired_ci95": [ci_lo, ci_hi],
                "legacy_unpaired_ci95": [leg_lo, leg_hi],
                "role": "CI-GATED" if stratum == "gate" else "DIRECTIONAL (never gated)",
            }
            if stratum == "gate":
                if delta >= PP_GATE / 100 and ci_lo > 0:
                    entry["verdict"] = "PASS"
                elif delta < 0:
                    entry["verdict"] = "REGRESSION"
                else:
                    entry["verdict"] = "NOT_DEMONSTRATED"
            else:
                entry["verdict"] = (
                    "DIRECTIONAL_POSITIVE"
                    if delta > 0
                    else ("DIRECTIONAL_NEGATIVE" if delta < 0 else "DIRECTIONAL_FLAT")
                )
            repo_result["strata"][stratum] = entry
            logger.info(
                "[%s/%s] n=%d base_R@5=%.4f ft_R@5=%.4f delta=%+.4f paired_CI=[%.4f,%.4f] %s",
                repo,
                stratum,
                len(pairs),
                base_metrics.get("recall_at_5", 0),
                ft_metrics.get("recall_at_5", 0),
                delta,
                ci_lo,
                ci_hi,
                entry["verdict"],
            )
        results["repos"][repo] = repo_result

    gate_verdicts = {
        r: v["strata"].get("gate", {}).get("verdict") for r, v in results["repos"].items()
    }
    results["gate_verdicts"] = gate_verdicts
    results["headline_rule"] = (
        "Headline = product-task directional deltas on BOTH repos alongside gated proxy CIs. "
        "'Proxy improved, product-task unproven' is the honest framing if product strata do "
        "not clear; never report as 'retrieval improved +X pp'."
    )

    out = "reports/w3_t5_eval_results_v2.json"
    Path("reports").mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Full results -> %s", out)

    print("\n=== W3-retry stratified eval (v2 corpus, test split, zero training overlap) ===")
    hdr = (
        f"{'Repo':<26} {'Stratum':<9} {'n':>5} {'Base R@5':>9} {'FT R@5':>8} "
        f"{'Delta':>8} {'Paired CI95':>22} {'Verdict'}"
    )
    print(hdr)
    print("-" * len(hdr))
    for repo, r in results["repos"].items():
        for stratum, e in r["strata"].items():
            if e.get("n_pairs", 0) == 0:
                continue
            ci = f"[{e['paired_ci95'][0]:+.4f}, {e['paired_ci95'][1]:+.4f}]"
            print(
                f"{repo:<26} {stratum:<9} {e['n_pairs']:>5} "
                f"{e['baseline']['recall_at_5']:>9.4f} {e['finetuned']['recall_at_5']:>8.4f} "
                f"{e['delta_r5']:>+8.4f} {ci:>22} {e['verdict']}"
            )
    print()


if __name__ == "__main__":
    main()
