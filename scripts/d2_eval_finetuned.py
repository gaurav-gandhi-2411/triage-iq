"""D2: evaluate a fine-tuned embedder against D1's held-out eval set, paired against the honest
pretrained baseline (ADR-0033 / reports/d1_clean_eval_baseline.json).

Leakage guard: re-asserts train/eval issue-level disjointness before eval -- hard pre-flight gate.

Builds a full-corpus index with the FINE-TUNED model (same corpus, same construction as D1's
`d1_full_corpus_index_{repo}_bge`, scripts/d1_build_full_corpus_index.py) so trained-vs-baseline
numbers share the identical candidate pool (ADR-0033's "corpus consistency" eval-param rule) --
never re-uses the stale served index.

Query construction is IDENTICAL to scripts/d1_baseline_eval.py's eval_one() (title + query_body,
where query_body is always absent from the eval-set JSON and so is always "") -- this is
deliberate, not a bug fixed here: D1's reported baseline (9.3% / 43.5%) was measured this way, so
matching it exactly is what makes the comparison apples-to-apples. Changing it would invalidate
the comparison, not improve it.

Uses TRUE paired bootstrap (scripts/_retrieval_eval_common.py::paired_bootstrap_ci, same resample
indices for both arms) on the per-pair R@5 hit vectors -- the project's established ship-bar
method (ADR-0027). Ships only if the paired CI on the improvement excludes zero AND the effect
size is meaningful (spec.md's ship bar) -- this script reports the numbers, it does not decide.

Usage:
  python scripts/d2_eval_finetuned.py --task vscode_duplicate --model-dir data/models/d2_finetuned_vscode_duplicate

Reads:  reports/d1_eval_set_{task}.json
        data/models/d1_full_corpus_index_{repo}_bge/     (pretrained baseline index)
        data/processed/issues_{repo}.parquet             (to build the fine-tuned index)
        --model-dir                                       (the fine-tuned SentenceTransformer)
Writes: reports/d2_eval_{task}[_{run_name}].json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from _retrieval_eval_common import paired_bootstrap_ci  # noqa: E402
from d2_assert_leakage_guard import TASKS, assert_task_disjoint  # noqa: E402

from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 2000
K_VALUES = [1, 5, 10]
MAX_BODY = 512

REPO_BY_TASK = {
    "vscode_duplicate": "microsoft_vscode",
    "k8s_related": "kubernetes_kubernetes",
}
REPORTS = Path("reports")
DATA = Path("data")
MODELS_DIR = DATA / "models"


def bootstrap_ci_recall(hits: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(hits)
    means = [hits[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hit_vectors(retriever: SimilarIssueRetriever, pairs: list[dict]) -> dict[str, np.ndarray]:
    live_numbers = {int(n) for n in retriever.issue_numbers}
    usable = [
        p for p in pairs
        if int(p["query_number"]) in live_numbers and int(p["original_number"]) in live_numbers
    ]
    k_max = max(K_VALUES)
    hit_lists: list[list[bool]] = []
    for row in usable:
        # Byte-identical to production (triage.py::_collect_signals): f"{title}. {body}",
        # UNTRUNCATED. Matches the d1_baseline_eval.py fix; see the ADR correcting
        # ADR-0031/0033/0034. Corpus-side truncation (_build_text, MAX_BODY) is untouched.
        query_text = f"{row['query_title']}. {row.get('query_body', '')}"
        results = retriever.retrieve(query_text, k=k_max, exclude_number=int(row["query_number"]))
        retrieved = [r["number"] for r in results]
        pos = int(row["original_number"])
        hit_lists.append([n == pos for n in retrieved])

    out = {"n_usable": len(usable), "n_total": len(pairs)}
    for k in K_VALUES:
        out[f"hits_at_{k}"] = np.array([float(any(h[:k])) for h in hit_lists])
    out["mrr"] = np.array([1.0 / (h.index(True) + 1) if True in h else 0.0 for h in hit_lists])
    return out


def build_finetuned_index(repo: str, model_dir: str) -> SimilarIssueRetriever:
    corpus = pd.read_parquet(DATA / "processed" / f"issues_{repo}.parquet")
    retriever = SimilarIssueRetriever(repo=repo, model_key=model_dir)  # model_key doubles as HF/local path
    retriever.build_index(corpus)
    return retriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--model-dir", required=True, help="path to the fine-tuned SentenceTransformer dir")
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    assert_task_disjoint(args.task)  # hard pre-flight gate, non-negotiable

    _, eval_file = TASKS[args.task]
    repo = REPO_BY_TASK[args.task]
    pairs = json.loads((REPORTS / eval_file).read_text(encoding="utf-8"))

    logger.info("[%s] loading pretrained baseline index (d1_full_corpus_index_%s_bge)", args.task, repo)
    baseline = SimilarIssueRetriever.load(str(MODELS_DIR / f"d1_full_corpus_index_{repo}_bge"))
    base_vecs = hit_vectors(baseline, pairs)

    logger.info("[%s] building fine-tuned index from %s", args.task, args.model_dir)
    trained = build_finetuned_index(repo, args.model_dir)
    trained_vecs = hit_vectors(trained, pairs)

    if base_vecs["n_usable"] != trained_vecs["n_usable"]:
        raise SystemExit(
            f"[{args.task}] usable-pair count mismatch between baseline ({base_vecs['n_usable']}) "
            f"and trained ({trained_vecs['n_usable']}) -- indices cover different corpora, "
            f"paired bootstrap requires identical pair sets. Check corpus consistency."
        )

    result: dict = {"task": args.task, "repo": repo, "model_dir": args.model_dir,
                     "n_eval_pairs": base_vecs["n_total"], "n_evaluated": base_vecs["n_usable"]}
    for k in K_VALUES:
        base_hits = base_vecs[f"hits_at_{k}"]
        trained_hits = trained_vecs[f"hits_at_{k}"]
        ci_lo, ci_hi, delta = paired_bootstrap_ci(base_hits, trained_hits)
        result[f"recall_at_{k}"] = {
            "baseline": float(base_hits.mean()),
            "baseline_ci95": list(round(c, 4) for c in bootstrap_ci_recall(base_hits)),
            "trained": float(trained_hits.mean()),
            "trained_ci95": list(round(c, 4) for c in bootstrap_ci_recall(trained_hits)),
            "delta": round(delta, 4),
            "delta_ci95_paired": [round(ci_lo, 4), round(ci_hi, 4)],
            "excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        }
    result["mrr_baseline"] = float(base_vecs["mrr"].mean())
    result["mrr_trained"] = float(trained_vecs["mrr"].mean())

    r5 = result["recall_at_5"]
    logger.info(
        "[%s] R@5: baseline=%.3f %s -> trained=%.3f %s  delta=%+.3f  CI95=%s  excludes_zero=%s",
        args.task, r5["baseline"], r5["baseline_ci95"], r5["trained"], r5["trained_ci95"],
        r5["delta"], r5["delta_ci95_paired"], r5["excludes_zero"],
    )

    suffix = f"_{args.run_name}" if args.run_name else ""
    out_path = REPORTS / f"d2_eval_{args.task}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
