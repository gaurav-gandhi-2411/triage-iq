"""D1 final deliverable: assemble reports/retrieval_clean_data.json (spec.md success criteria).

Pulls together every artifact this phase produced into the single report ADR-0033 cites:
clean pool decisions, frozen eval sets + provenance, the honest clean-eval baseline, and the
recommended eval params for D2.

Reads:  reports/d1_clean_pool_checkpoint1.json
        reports/d1_eval_set_summary.json
        reports/d1_clean_eval_baseline.json
Writes: reports/retrieval_clean_data.json
"""

from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path("reports")


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def main() -> None:
    checkpoint1 = load("d1_clean_pool_checkpoint1.json")
    eval_summary = load("d1_eval_set_summary.json")
    baseline = load("d1_clean_eval_baseline.json")

    report = {
        "phase": "D1 -- clean retrieval data + trustworthy eval foundation",
        "adr": "ADR-0033",
        "clean_training_pool": checkpoint1["summary"],
        "clean_training_pool_channel_decisions": checkpoint1["channel_decisions"],
        "held_out_eval_sets": eval_summary,
        "eval_set_disjointness": "asserted programmatically (issue-level) for both gateable "
        "eval sets -- scripts/d1_build_eval_set.py::main",
        "eval_set_precision": "100% by construction -- every eval pair was individually "
        "hand-verified genuine (not a channel-precision sample); clears the >=90% hard rule "
        "trivially. See reports/d1_pair_quality_review.json + "
        "reports/d1_eval_carve_review_{k8s,vscode}.json for every verdict + reason.",
        "clean_eval_baseline": baseline["results"],
        "baseline_caveats": [
            "k8s_related's honest R@5 (9.3% [5.3,14.0], n=150) is far below the previously "
            "reported 23.5% (ADR-0030) -- that number was measured on a noisier gold set "
            "including title_sim contamination (ADR-0032). Two factors both contribute to the "
            "drop, not disentangled at D1's power: (1) genuinely harder/cleaner pairs -- "
            "title_sim pairs had trivially high lexical overlap that inflated recall on "
            "incidental matches; (2) a ~2x larger candidate corpus (30,000 vs the old live "
            "index's 15,000) mechanically lowers recall by adding distractors, independent of "
            "pair quality. Only 16 of the 150 k8s eval pairs fall within the old 15k-issue "
            "range -- too few (n=16) to cleanly separate the two effects. Recommendation: D2 "
            "should treat 9.3% as the real, current, honest baseline to beat -- it reflects "
            "the actual corpus scale the product should be benchmarked against, not an "
            "artifact worth re-litigating.",
            "vscode_duplicate (43.5% [37.0,50.5], n=200) and vscode_related (57.9% "
            "[36.8,78.9], n=19, directional-only) are NOT comparable to each other or to k8s "
            "-- three different tasks, three different corpora, reported separately by design.",
        ],
        "recommended_eval_params_for_d2": {
            "primary_gate_metric": "Recall@5 (matches the product surface: top-5 similar "
            "issues shown to the triager)",
            "secondary_metrics": "Recall@1, Recall@10 for shape; MRR for rank-position signal "
            "beyond hit/miss (already computed here, not a gate)",
            "ci_method": "percentile bootstrap, 2000 resamples, seed=42 for single-arm numbers "
            "(this report); TRUE PAIRED bootstrap (same resample indices for both arms, per "
            "scripts/_retrieval_eval_common.py::paired_bootstrap_ci, ADR-0027's corrected "
            "method) for any D2 trained-vs-baseline delta -- a lever/model ships only if the "
            "paired CI on the improvement excludes zero, per this project's established bar",
            "gateable_tasks": ["k8s_related", "vscode_duplicate"],
            "directional_only_tasks": [
                "vscode_related (n=19, underpowered by construction, "
                "never gated, never blended with vscode_duplicate)"
            ],
            "corpus_consistency": "D2 must evaluate against the CURRENT full-corpus index "
            "(this report's d1_full_corpus_index_*, or its successor), never a stale subset -- "
            "corpus size directly affects recall, so baseline and trained-model numbers must "
            "share the same candidate pool to be comparable",
        },
    }

    (REPORTS / "retrieval_clean_data.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Wrote reports/retrieval_clean_data.json")


if __name__ == "__main__":
    main()
