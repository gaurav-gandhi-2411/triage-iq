"""LEVER 1 + LEVER 2 combined re-measurement.

Compares three variants, all against D1's frozen, hand-verified held-out eval sets
(ADR-0033), same paired-bootstrap methodology as scripts/_retrieval_eval_common.py and
scripts/w3_t5_eval.py:

  - BASELINE:    d1_full_corpus_index_{repo}_bge          (512-char corpus cut, no query instr.)
  - LEVER1:      d1_full_corpus_index_{repo}_bge_lever1    (tokenizer corpus truncation, no instr.)
  - LEVER1+2:    d1_full_corpus_index_{repo}_bge_lever1    (same index, + BGE query instruction)

Query text is always untruncated title+body, byte-identical to production
(src/triage_iq/models/triage.py::_collect_signals) -- unaffected by either lever, since both
levers only change corpus-side text (LEVER1) or query-side instruction prefix (LEVER2), not
query truncation (already fixed, ADR-0035).

Reads:  reports/d1_eval_set_{k8s_related,vscode_duplicate,vscode_related}.json
        data/models/d1_full_corpus_index_{repo}_bge[/_lever1]/
Writes: reports/lever12_eval_results.json
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
K_MAX = 20

MODELS_DIR = Path("data/models")
REPORTS = Path("reports")

EVAL_SETS = [
    ("k8s_related", "kubernetes_kubernetes", "d1_eval_set_k8s_related.json"),
    ("vscode_duplicate", "microsoft_vscode", "d1_eval_set_vscode_duplicate.json"),
    ("vscode_related", "microsoft_vscode", "d1_eval_set_vscode_related.json"),
]

VARIANTS = ["baseline", "lever1", "lever1_2"]


def query_text(row: dict) -> str:
    # Byte-identical to production: f"{title}. {body}", UNTRUNCATED (ADR-0035 fix).
    return f"{row['query_title']}. {row.get('query_body', '')}"


def paired_bootstrap_ci(base_hits: np.ndarray, new_hits: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    n = len(base_hits)
    d = new_hits - base_hits
    deltas = [d[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float(d.mean())


def bootstrap_ci_recall(hits: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(hits)
    means = [hits[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def eval_variant(
    detector: SimilarIssueRetriever, pairs: list[dict], apply_query_instruction: bool | None
) -> tuple[np.ndarray, int, int]:
    live_numbers = {int(n) for n in detector.issue_numbers}
    usable = [
        p
        for p in pairs
        if p["query_number"] in live_numbers and p["original_number"] in live_numbers
    ]
    hits = []
    for row in usable:
        qt = query_text(row)
        results = detector.retrieve(
            qt,
            k=K_MAX,
            exclude_number=int(row["query_number"]),
            apply_query_instruction=apply_query_instruction,
        )
        retrieved = [r["number"] for r in results]
        pos = int(row["original_number"])
        hits.append(any(n == pos for n in retrieved[:5]))
    return np.array(hits, dtype=float), len(usable), len(pairs) - len(usable)


def main() -> None:
    out: dict = {"bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": SEED}, "results": {}}

    detectors_baseline: dict[str, SimilarIssueRetriever] = {}
    detectors_lever1: dict[str, SimilarIssueRetriever] = {}

    for label, repo, eval_path in EVAL_SETS:
        if repo not in detectors_baseline:
            log.info("[%s] loading baseline + lever1 indexes...", repo)
            detectors_baseline[repo] = SimilarIssueRetriever.load(
                str(MODELS_DIR / f"d1_full_corpus_index_{repo}_bge")
            )
            detectors_lever1[repo] = SimilarIssueRetriever.load(
                str(MODELS_DIR / f"d1_full_corpus_index_{repo}_bge_lever1")
            )

        pairs = json.loads((REPORTS / eval_path).read_text(encoding="utf-8"))

        base_hits, n_used, n_missing = eval_variant(
            detectors_baseline[repo], pairs, apply_query_instruction=False
        )
        l1_hits, _, _ = eval_variant(detectors_lever1[repo], pairs, apply_query_instruction=False)
        l12_hits, _, _ = eval_variant(detectors_lever1[repo], pairs, apply_query_instruction=True)

        base_r5 = float(base_hits.mean())
        l1_r5 = float(l1_hits.mean())
        l12_r5 = float(l12_hits.mean())

        base_ci = bootstrap_ci_recall(base_hits)
        l1_ci = bootstrap_ci_recall(l1_hits)
        l12_ci = bootstrap_ci_recall(l12_hits)

        l1_lo, l1_hi, l1_delta = paired_bootstrap_ci(base_hits, l1_hits)
        l12_lo, l12_hi, l12_delta = paired_bootstrap_ci(base_hits, l12_hits)
        l2_only_lo, l2_only_hi, l2_only_delta = paired_bootstrap_ci(l1_hits, l12_hits)

        result = {
            "repo": repo,
            "n_eval_pairs_total": len(pairs),
            "n_evaluated": n_used,
            "n_missing_from_index": n_missing,
            "baseline_r5": base_r5,
            "baseline_r5_ci95": list(round(c, 4) for c in base_ci),
            "lever1_r5": l1_r5,
            "lever1_r5_ci95": list(round(c, 4) for c in l1_ci),
            "lever1_2_r5": l12_r5,
            "lever1_2_r5_ci95": list(round(c, 4) for c in l12_ci),
            "delta_lever1_vs_baseline": round(l1_delta, 4),
            "delta_lever1_vs_baseline_ci95": [round(l1_lo, 4), round(l1_hi, 4)],
            "delta_lever1_2_vs_baseline": round(l12_delta, 4),
            "delta_lever1_2_vs_baseline_ci95": [round(l12_lo, 4), round(l12_hi, 4)],
            "delta_lever2_only_vs_lever1": round(l2_only_delta, 4),
            "delta_lever2_only_vs_lever1_ci95": [round(l2_only_lo, 4), round(l2_only_hi, 4)],
        }
        out["results"][label] = result

        log.info(
            "[%s] n=%d  BASELINE R@5=%.4f %s  LEVER1 R@5=%.4f %s (delta %+.4f CI[%+.4f,%+.4f])  "
            "LEVER1+2 R@5=%.4f %s (delta vs baseline %+.4f CI[%+.4f,%+.4f])",
            label,
            n_used,
            base_r5,
            base_ci,
            l1_r5,
            l1_ci,
            l1_delta,
            l1_lo,
            l1_hi,
            l12_r5,
            l12_ci,
            l12_delta,
            l12_lo,
            l12_hi,
        )

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "lever12_eval_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote reports/lever12_eval_results.json")


if __name__ == "__main__":
    main()
