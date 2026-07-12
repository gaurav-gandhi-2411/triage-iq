"""Phase C: assemble the master feasibility report from the three prior analysis passes plus
the manual precision review. Analysis only -- combines already-computed JSON, does not re-mine.

Output: reports/phaseC_feasibility.json
Reproduce: python scripts/phaseC_assemble_report.py (after the three phaseC_*.py scripts + a
manual precision pass recorded in reports/phaseC_precision_review.json)
"""

from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path("reports")


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def main() -> None:
    channel_mining = load("phaseC_channel_mining.json")
    live_probe = load("phaseC_live_probe.json")
    power_liveindex = load("phaseC_power_and_liveindex.json")
    precision_review = load("phaseC_precision_review.json")

    k8s_comment_scrape_estimate = {
        "k8s_issues_with_comments_gt_0": 28755,
        "k8s_total_processed_rows": 29994,
        "probe_hit_rate": 0.08,
        "probe_n": 25,
        "probe_hit_rate_wilson_95ci_approx": [0.01, 0.26],
        "extrapolated_yield_point_estimate": int(28755 * 0.08),
        "caveat": "n=25 probe gives a wide CI on the 8% rate (Wilson approx [1%,26%]) -- this is "
        "a feasibility signal, not a scale-mining estimate. Realizing it requires a NEW comment "
        "scrape for all ~30K k8s issues (comments were never fetched for k8s, original or "
        "forward-scrape) -- out of scope for this analysis-only phase.",
    }

    report = {
        "generated_by": "scripts/phaseC_assemble_report.py",
        "spec": "spec.md (Phase C: Product-Task Gold Feasibility)",
        "inputs": [
            "reports/phaseC_channel_mining.json",
            "reports/phaseC_live_probe.json",
            "reports/phaseC_power_and_liveindex.json",
            "reports/phaseC_precision_review.json (manual precision judging on fixed-seed samples)",
        ],
        "1_channels": {
            "local_mining_counts": channel_mining["repos"],
            "live_probe_counts": live_probe["repos"],
            "precision_review": precision_review["channels"],
            "k8s_comments_scrape_feasibility_estimate": k8s_comment_scrape_estimate,
            "summary": precision_review["cross_channel_conclusion"],
        },
        "2_power": power_liveindex["repos"],
        "3_live_index": {
            repo: {
                "live_index_size": entry["live_index_size"],
                "product_pairs_usable_now": entry["live_index_measurability_now"],
            }
            for repo, entry in power_liveindex["repos"].items()
        },
        "4_go_no_go": {
            "measure_k8s_live_recall": {
                "verdict": "GO -- near-free, largely actionable with EXISTING data",
                "reasoning": "277 of k8s's 776 product-stratum pairs already fall inside the "
                "live index's number range (#1-15,002) and are usable for measurement with ZERO "
                "leakage risk: the live-serving retriever is an off-the-shelf pretrained embedder "
                "(BAAI/bge-base-en-v1.5) never trained on any gold pair, so the w3-retry train/"
                "val/test split labels (which exist only to protect the FINE-TUNED variant's "
                "training) do not apply. At a plausible live-index recall prior (p=0.25-0.30, "
                "informed by the smaller/easier v1 index vs the v2 baseline's 0.228-0.263), "
                "n=277 already sits at or near a +-5pp CI half-width (+-5.1 to +-5.4pp); only "
                "under the maximally conservative p=0.5 assumption does it fall short (need 108 "
                "more). This is a RE-EVAL action (re-run the recall@5 eval against the v1 "
                "'dup_index_kubernetes_kubernetes_bge' index using the 277 in-range product "
                "pairs, ignoring their w3-retry split label), not a new mining/scraping effort.",
                "effort": "near-zero: no new data collection; a scoped eval-script change "
                "(select product-stratum pairs by live-index membership instead of split label) "
                "plus, optionally, ~50-100 more pairs from a filtered channel D pilot for a "
                "tighter CI under the conservative prior.",
            },
            "ship_k8s_finetune_product_gate": {
                "verdict": "NO-GO at current scope -- data bar is disproportionate to the corpus",
                "reasoning": "80%-power gating needs 466 test pairs (vs 57 now) / ~6,075 TOTAL "
                "product pairs at the current split algorithm's 7.67% test fraction -- roughly "
                "8x k8s's entire current product stratum (776 pairs) and >1.5x the entire "
                "current k8s gold set across all strata (4,030 pairs). No channel found here "
                "gets meaningfully close: channel A is mined out (9 candidates), channel E is "
                "too noisy to use (3% strict precision), channel B needs an unproven k8s scrape, "
                "channel D needs a hub-filter that hasn't been designed yet. The point estimate "
                "(+3.51pp) is also close to the near-zero flag threshold (0.03) -- if the true "
                "effect is smaller, the ask roughly quadruples (1,864 test pairs). Not "
                "recommended to pursue at this repo's current data scale.",
            },
            "ship_vscode_finetune_product_gate": {
                "verdict": "MIXED -- bounded gap, not yet closed by mineable channels found here",
                "reasoning": "80%-power gating needs 709 test pairs (vs 281 now, +428) / ~1,169 "
                "total product pairs at the current 60.7% test fraction -- a real but "
                "proportionate ask relative to vscode's current 505-pair product stratum "
                "(roughly 2.3x growth). The best channel found (B, comments) yields only ~75 net "
                "new high-precision pairs locally, nowhere near 428 by itself. Channel D "
                "(cross-referenced, hub-filtered) is promising -- ~48% of sampled issues have an "
                "issue-sourced cross-ref -- but needs a validated hub-exclusion design before "
                "scale mining, which is out of scope for this analysis pass. Point estimate "
                "(+3.20pp) has the same near-zero-threshold caveat as k8s.",
            },
            "channel_c_native_linked_issues": {"verdict": "NO-GO -- dead channel, 0% yield both repos, no further investment"},
            "channel_e_label_cluster": {"verdict": "NO-GO as a standalone channel -- 3% strict precision on k8s; would need a text-similarity co-filter not evaluated here"},
        },
        "5_collection_scope_if_pursuing_vscode_finetune_gate": {
            "recommended_channels": ["B_comments (already available, ~75 pairs)", "D_cross_referenced (needs hub-filter design + validation pilot before scaling)"],
            "not_recommended": ["A_extended_body (30% precision)", "E_label_cluster (3% precision)", "C_native_linked (dead)"],
            "estimated_gap_after_channel_B": "428 - 75 = ~353 test-equivalent pairs still needed; channel D's realistic yield at scale is unquantified (would require a proper hub-filter design and a larger, live-API-budgeted pilot -- next phase's first task, not this one's)",
        },
    }

    out = REPORTS / "phaseC_feasibility.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
