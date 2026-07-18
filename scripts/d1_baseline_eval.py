"""D1 task 3: honest clean-eval baseline for the current (untrained) retriever.

Runs the off-the-shelf pretrained BGE retriever's Recall@1/5/10 (+ MRR) against D1's frozen,
hand-verified held-out eval sets, over the full-corpus D1-scoped index
(`data/models/d1_full_corpus_index_{repo}_bge`, built by scripts/d1_build_full_corpus_index.py --
NOT the stale served index, which only covers a fraction of these eval pairs). Same bootstrap
method as ADR-0027/ADR-0030/ADR-0032 (percentile, 2000 resamples, seed=42) so numbers are
comparable to prior baselines.

Three eval sets, reported SEPARATELY per GG's checkpoint-1 decision -- never blended:
  - k8s_related       (150 pairs, gateable)
  - vscode_duplicate  (200 pairs, gateable)
  - vscode_related    (19 pairs, directional-only, wide CI, NOT a gate)

Reads:  reports/d1_eval_set_k8s_related.json
        reports/d1_eval_set_vscode_duplicate.json
        reports/d1_eval_set_vscode_related.json
        data/models/d1_full_corpus_index_{repo}_bge/
Writes: reports/d1_clean_eval_baseline.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 2000
K_VALUES = [1, 5, 10, 20]
MAX_BODY = 512

MODELS_DIR = Path("data/models")
REPORTS = Path("reports")

EVAL_SETS = [
    ("k8s_related", "kubernetes_kubernetes", "d1_eval_set_k8s_related.json"),
    ("vscode_duplicate", "microsoft_vscode", "d1_eval_set_vscode_duplicate.json"),
    ("vscode_related", "microsoft_vscode", "d1_eval_set_vscode_related.json"),
]


def bootstrap_ci_recall(hits: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(hits)
    means = [hits[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_ci_mrr(rr: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(rr)
    means = [rr[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def eval_one(label: str, repo: str, eval_path: str, detector: SimilarIssueRetriever) -> dict:
    pairs = json.loads((REPORTS / eval_path).read_text(encoding="utf-8"))
    live_numbers = {int(n) for n in detector.issue_numbers}
    missing = [
        p
        for p in pairs
        if p["query_number"] not in live_numbers or p["original_number"] not in live_numbers
    ]
    if missing:
        log.warning(
            "[%s] %d/%d eval pairs missing from the full-corpus index (issue not in "
            "data/processed corpus at all) -- excluded, not silently zero-scored",
            label,
            len(missing),
            len(pairs),
        )
    usable = [p for p in pairs if p not in missing]

    k_max = max(K_VALUES)
    hit_lists: list[list[bool]] = []
    for row in usable:
        query_text = f"{row['query_title']}. {row.get('query_body', '')[:MAX_BODY]}"
        results = detector.retrieve(query_text, k=k_max, exclude_number=int(row["query_number"]))
        retrieved = [r["number"] for r in results]
        pos = int(row["original_number"])
        hit_lists.append([n == pos for n in retrieved])

    result: dict = {
        "label": label,
        "repo": repo,
        "n_eval_pairs_total": len(pairs),
        "n_missing_from_index": len(missing),
        "n_evaluated": len(usable),
        "index": f"d1_full_corpus_index_{repo}_bge ({len(live_numbers)} records)",
    }
    for k in K_VALUES:
        result[f"recall_at_{k}"] = (
            float(np.mean([any(h[:k]) for h in hit_lists])) if hit_lists else None
        )

    r1_hits = np.array([float(any(h[:1])) for h in hit_lists])
    r5_hits = np.array([float(any(h[:5])) for h in hit_lists])
    r10_hits = np.array([float(any(h[:10])) for h in hit_lists])
    mrr_vals = np.array([1.0 / (h.index(True) + 1) if True in h else 0.0 for h in hit_lists])

    if len(hit_lists):
        result["recall_at_1_ci95"] = list(round(c, 4) for c in bootstrap_ci_recall(r1_hits))
        result["recall_at_5_ci95"] = list(round(c, 4) for c in bootstrap_ci_recall(r5_hits))
        result["recall_at_10_ci95"] = list(round(c, 4) for c in bootstrap_ci_recall(r10_hits))
        result["mrr"] = float(mrr_vals.mean())
        result["mrr_ci95"] = list(round(c, 4) for c in bootstrap_ci_mrr(mrr_vals))
        result["ci_note"] = (
            "small-n, illustrative CI only -- directional, not a gate"
            if len(hit_lists) < 30
            else None
        )

    log.info(
        "[%s] n=%d  R@1=%.3f  R@5=%.3f %s  R@10=%.3f  MRR=%.3f",
        label,
        len(hit_lists),
        result.get("recall_at_1", float("nan")),
        result.get("recall_at_5", float("nan")),
        result.get("recall_at_5_ci95"),
        result.get("recall_at_10", float("nan")),
        result.get("mrr", float("nan")),
    )
    return result


def main() -> None:
    detectors = {}
    results = []
    for label, repo, eval_path in EVAL_SETS:
        if repo not in detectors:
            detectors[repo] = SimilarIssueRetriever.load(
                str(MODELS_DIR / f"d1_full_corpus_index_{repo}_bge")
            )
        results.append(eval_one(label, repo, eval_path, detectors[repo]))

    out = {
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": SEED, "method": "percentile"},
        "results": results,
    }
    (REPORTS / "d1_clean_eval_baseline.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    log.info("Wrote reports/d1_clean_eval_baseline.json")


if __name__ == "__main__":
    main()
