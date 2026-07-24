"""Assemble the master retrieval-quality-improvement report from the baseline reproduction
plus the three lever results. Analysis only -- combines already-computed JSON, does not
re-run any eval.

Output: reports/retrieval_quality.json
Reproduce: python scripts/assemble_retrieval_quality_report.py (after
  phaseC_k8s_live_product_eval.py, phaseC_vscode_live_product_eval.py,
  lever1_hybrid_bm25_rrf.py, lever2_reranker.py, lever3_stronger_embedder.py)
"""

from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path("reports")


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def main() -> None:
    k8s_baseline = load("phaseC_k8s_live_product_eval.json")
    vscode_baseline = load("phaseC_vscode_live_product_eval.json")
    lever1 = load("lever1_hybrid_bm25_rrf.json")
    lever2 = load("lever2_reranker.json")
    lever3 = load("lever3_stronger_embedder.json")

    report = {
        "generated_by": "scripts/assemble_retrieval_quality_report.py",
        "spec": "spec.md (Retrieval Quality Improvement)",
        "adr": "docs/architecture/adr/0031-retrieval-quality-improvement.md",
        "bar": (
            "product-task Recall@5 per repo vs live v1 baseline, paired bootstrap CI "
            "(2000 resamples, seed=42). Ships only if CI excludes zero on k8s (n=277, "
            "primary gate); vscode (n~=290) also reported. Magnitude reported alongside "
            "significance -- a lift that clears CI but is ~3pp on a ~23-27% base is "
            "practically marginal, not automatically shipped."
        ),
        "0_baseline": {
            "note": (
                "k8s reproduces byte-identically against ADR-0030. vscode's previously-"
                "reported 22.4%/n=254 (ADR-0028) does not reproduce from committed code+data "
                "-- provenance unrecoverable (denominator 281 doesn't match the 505-row "
                "product stratum via any channel/confidence/dedup filter tried). Adopted "
                "n=292/26.71% as the working vscode baseline going forward: script-"
                "reproducible via scripts/phaseC_vscode_live_product_eval.py, same method "
                "as k8s."
            ),
            "k8s": {
                "n": k8s_baseline["n_pairs"],
                "recall_at_5": k8s_baseline["recall_at_5"],
                "ci95": k8s_baseline["recall_at_5_ci95"],
            },
            "vscode": {
                "n": vscode_baseline["n_pairs"],
                "recall_at_5": vscode_baseline["recall_at_5"],
                "ci95": vscode_baseline["recall_at_5_ci95"],
                "superseded_number_adr0028": vscode_baseline["adr0028_reported"],
            },
        },
        "1_lever1_hybrid_bm25_rrf": lever1,
        "2_lever2_reranker": lever2,
        "3_lever3_stronger_embedder": lever3,
        "headline": {
            "verdict": "NO LEVER SHIPS",
            "summary": (
                "Three untried, high-headroom levers (hybrid BM25+RRF, pretrained cross-"
                "encoder reranker, stronger pretrained embedder) were each measured against "
                "the identical bar (paired bootstrap CI on product-task R@5, k8s primary "
                "gate). None clears it. RRF (Lever 1 primary) and the stronger embedder "
                "(Lever 3) both leave the k8s CI crossing zero. The reranker (Lever 2) "
                "regresses quality on both repos and adds 190-330x latency. The one variant "
                "that technically clears CI on both repos -- Lever 1's weighted score-"
                "fusion, +3.25pp k8s / +4.11pp vscode -- has a k8s CI lower bound of only "
                "0.35pp and matches the magnitude of the already-NO-GO'd W3 fine-tune; "
                "judged not worth shipping for the same reason. The ~23-27% product-task "
                "base rate does not move with any pretrained, zero-training lever tried "
                "here. Retrieval quality remains the weakest model in the pipeline; closing "
                "the gap likely requires either in-domain fine-tuning (reopening the "
                "leakage question ADR-0030 deliberately avoided) or a different mining/"
                "labeling strategy, not a pretrained-component swap."
            ),
        },
    }

    OUTPUT_PATH = REPORTS / "retrieval_quality.json"
    OUTPUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
