"""D3: evaluate a fine-tuned embedder against D1's UNCHANGED held-out eval sets, paired against
the CURRENT, correct baseline -- not scripts/d2_eval_finetuned.py's methodology, which is
deliberately frozen to D1's original (title-only query, pre-ADR-0040 char-truncated corpus)
measurement for its own apples-to-apples comparison and is now stale relative to what this
project actually ships.

Two corrections vs. d2_eval_finetuned.py, both required to be comparable to the 39.39% k8s
clean-subset number and the current 53.50% vscode number this run is measured against:
  1. Query construction: full title+body, UNTRUNCATED, looked up from the processed corpus by
     issue number (the eval-set JSON's own query_body field is empty) -- matches production
     (triage.py::_collect_signals) and the 2026-08-11 k8s clean-eval methodology exactly, not
     D1's original title-only measurement.
  2. Baseline/corpus index: data/models/dup_index_{repo}_bge -- confirmed byte-identical (SHA-256)
     to the live-serving index by the 2026-08-10 investigation, i.e. the CURRENT token-based
     corpus truncation (ADR-0040), not D1's stale d1_full_corpus_index_{repo}_bge (pre-ADR-0040,
     char-truncated).

For k8s_related, reports BOTH the full 150-pair number (for continuity/sanity-check against
ADR-0040's reproduced 24.67%) AND the 66-pair VALID-only subset (reports/track2_k8s_clean_eval.json's
strict-rubric labels) -- the number this run is actually measured against (39.39% baseline).

Leakage guard: re-asserts train/eval issue-level disjointness before eval (scripts/
d3_assert_leakage_guard.py) -- hard pre-flight gate.

Usage:
  python scripts/d3_eval_finetuned.py --task k8s_related --model-dir data/models/d3_finetuned_k8s_related
  python scripts/d3_eval_finetuned.py --task vscode_duplicate --model-dir data/models/d3_finetuned_vscode_duplicate

Reads:  reports/d1_eval_set_{task}.json
        reports/track2_k8s_clean_eval.json        (k8s_related only, for the VALID-66 subset)
        data/models/dup_index_{repo}_bge           (current live-serving-matching baseline)
        data/processed/issues_{repo}.parquet
        --model-dir
Writes: reports/d3_eval_{task}.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from d3_assert_leakage_guard import TASKS, assert_task_disjoint

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from _retrieval_eval_common import paired_bootstrap_ci  # noqa: E402

from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 2000
K_VALUES = [1, 5, 10]

REPO_BY_TASK = {
    "vscode_duplicate": "microsoft_vscode",
    "k8s_related": "kubernetes_kubernetes",
    "k8s_related_valid_subset": "kubernetes_kubernetes",
    "k8s_related_fullcorpus_negs": "kubernetes_kubernetes",
}
REPORTS = Path("reports")
DATA = Path("data")
MODELS_DIR = DATA / "models"
K8S_CLEAN_EVAL = REPORTS / "track2_k8s_clean_eval.json"


def bootstrap_ci_recall(hits: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(hits)
    means = [hits[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_body_lookup(repo: str) -> dict[int, str]:
    corpus = pd.read_parquet(DATA / "processed" / f"issues_{repo}.parquet")
    return dict(zip(corpus["number"].astype(int), corpus["body_clean"].astype(str), strict=True))


def hit_vectors(
    retriever: SimilarIssueRetriever, pairs: list[dict], body_by_num: dict[int, str]
) -> dict[str, np.ndarray]:
    live_numbers = {int(n) for n in retriever.issue_numbers}
    usable = [
        p for p in pairs
        if int(p["query_number"]) in live_numbers and int(p["original_number"]) in live_numbers
    ]
    k_max = max(K_VALUES)
    hit_lists: list[list[bool]] = []
    for row in usable:
        q = int(row["query_number"])
        # CORRECTED vs d2_eval_finetuned.py: real, full, untruncated body from the corpus --
        # matches production (triage.py::_collect_signals) and the 2026-08-11 investigation's
        # methodology, not D1's original title-only measurement.
        body = body_by_num.get(q, "")
        query_text = f"{row['query_title']}. {body}"
        results = retriever.retrieve(query_text, k=k_max, exclude_number=q)
        retrieved = [r["number"] for r in results]
        pos = int(row["original_number"])
        hit_lists.append([n == pos for n in retrieved])

    out: dict = {"n_usable": len(usable), "n_total": len(pairs), "_pair_ids": [
        (int(p["query_number"]), int(p["original_number"])) for p in usable
    ]}
    for k in K_VALUES:
        out[f"hits_at_{k}"] = np.array([float(any(h[:k])) for h in hit_lists])
    out["mrr"] = np.array([1.0 / (h.index(True) + 1) if True in h else 0.0 for h in hit_lists])
    return out


def build_finetuned_index(repo: str, model_dir: str) -> SimilarIssueRetriever:
    corpus = pd.read_parquet(DATA / "processed" / f"issues_{repo}.parquet")
    retriever = SimilarIssueRetriever(repo=repo, model_key=model_dir)
    retriever.build_index(corpus)
    return retriever


def load_baseline(repo: str) -> SimilarIssueRetriever:
    path = MODELS_DIR / f"dup_index_{repo}_bge"
    logger.info("loading CURRENT baseline index (byte-identical to live-serving): %s", path)
    return SimilarIssueRetriever.load(str(path))


def score_pairs(
    result_key: str, result: dict, base_vecs: dict, trained_vecs: dict
) -> None:
    if base_vecs["n_usable"] != trained_vecs["n_usable"]:
        raise SystemExit(
            f"[{result_key}] usable-pair count mismatch between baseline ({base_vecs['n_usable']}) "
            f"and trained ({trained_vecs['n_usable']}) -- indices cover different corpora."
        )
    entry: dict = {"n_pairs": base_vecs["n_total"], "n_evaluated": base_vecs["n_usable"]}
    for k in K_VALUES:
        base_hits = base_vecs[f"hits_at_{k}"]
        trained_hits = trained_vecs[f"hits_at_{k}"]
        ci_lo, ci_hi, delta = paired_bootstrap_ci(base_hits, trained_hits)
        entry[f"recall_at_{k}"] = {
            "baseline": float(base_hits.mean()),
            "baseline_ci95": list(round(c, 4) for c in bootstrap_ci_recall(base_hits)),
            "trained": float(trained_hits.mean()),
            "trained_ci95": list(round(c, 4) for c in bootstrap_ci_recall(trained_hits)),
            "delta": round(delta, 4),
            "delta_ci95_paired": [round(ci_lo, 4), round(ci_hi, 4)],
            "excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        }
    entry["mrr_baseline"] = float(base_vecs["mrr"].mean())
    entry["mrr_trained"] = float(trained_vecs["mrr"].mean())
    result[result_key] = entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    assert_task_disjoint(args.task)

    _, eval_file = TASKS[args.task]
    repo = REPO_BY_TASK[args.task]
    pairs = json.loads((REPORTS / eval_file).read_text(encoding="utf-8"))
    body_by_num = load_body_lookup(repo)

    baseline = load_baseline(repo)
    base_vecs = hit_vectors(baseline, pairs, body_by_num)

    logger.info("[%s] building fine-tuned index from %s", args.task, args.model_dir)
    trained = build_finetuned_index(repo, args.model_dir)
    trained_vecs = hit_vectors(trained, pairs, body_by_num)

    result: dict = {"task": args.task, "repo": repo, "model_dir": args.model_dir}
    score_pairs("full_eval_set", result, base_vecs, trained_vecs)

    k8s_tasks = ("k8s_related", "k8s_related_valid_subset", "k8s_related_fullcorpus_negs")
    if args.task in k8s_tasks and K8S_CLEAN_EVAL.exists():
        clean = json.loads(K8S_CLEAN_EVAL.read_text(encoding="utf-8"))["pairs"]
        valid_keys = {
            (int(p["query_number"]), int(p["target_number"])) for p in clean if p["label"] == "VALID"
        }
        valid_pairs = [
            p for p in pairs
            if (int(p["query_number"]), int(p["original_number"])) in valid_keys
        ]
        logger.info(
            "[%s] VALID-subset (strict rubric): %d/%d eval pairs", args.task, len(valid_pairs), len(pairs)
        )
        base_valid = hit_vectors(baseline, valid_pairs, body_by_num)
        trained_valid = hit_vectors(trained, valid_pairs, body_by_num)
        score_pairs("valid_subset_66", result, base_valid, trained_valid)

    r5_key = "valid_subset_66" if "valid_subset_66" in result else "full_eval_set"
    r5 = result[r5_key]["recall_at_5"]
    logger.info(
        "[%s] (%s) R@5: baseline=%.3f %s -> trained=%.3f %s  delta=%+.3f  CI95=%s  excludes_zero=%s",
        args.task, r5_key, r5["baseline"], r5["baseline_ci95"], r5["trained"], r5["trained_ci95"],
        r5["delta"], r5["delta_ci95_paired"], r5["excludes_zero"],
    )

    out_path = REPORTS / f"d3_eval_{args.task}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
