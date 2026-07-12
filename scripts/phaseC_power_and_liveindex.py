"""Phase C: power targets + live-index quantification (ADR-0030 input). Analysis only.

Two questions, both reusing ground truth already computed by prior phases:

1. POWER — pairs needed to (a) gate the Phase 2 fine-tune's product task (reuse
   phase2a_corpus_feasibility.py's power_calc method, applied to ADR-0027's actual product-task
   results, not the W3-era numbers phase2a used); (b) measure the k8s LIVE retriever's
   product-task recall@5 to a workable CI (single-proportion precision calc — there is no
   existing point estimate against the live index, so this is not a delta-power calc).

2. LIVE-INDEX — the live-serving retriever (`dup_index_{slug}_bge`, BAAI/bge-base-en-v1.5,
   loaded by src/triage_iq/api/loader.py) is an OFF-THE-SHELF embedding model: it is never
   trained on gold pairs (only the separate, unshipped `bge_finetuned_*_v2` artifact is). This
   means the w3-retry train/val/test split labels -- designed to prevent leakage for the
   FINE-TUNED model's training -- carry no leakage risk for the live model. Every existing
   product-stratum pair whose query+target both fall in the live index's number set is USABLE
   for measuring the live retriever RIGHT NOW, not just the pairs the w3-retry split happened to
   assign to "test". This script quantifies exactly how many that is per repo, and compares
   against the power targets from (1).

Output: reports/phaseC_power_and_liveindex.json
Reproduce: python scripts/phaseC_power_and_liveindex.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

GOLD_PATH = Path("data/gold_related_v2.parquet")
SPLIT_PATH = Path("data/w3_split_v2.parquet")
MODELS_DIR = Path("data/models")
W3_T5_RESULTS = Path("reports/w3_t5_eval_results_v2.json")
OUTPUT_PATH = Path("reports/phaseC_power_and_liveindex.json")

# ── ADR-0027 product-task results (reports/w3_t5_eval_results_v2.json), the actual held
#    fine-tune's gating target -- NOT the W3-era numbers phase2a_corpus_feasibility.py used. ──
PRODUCT_RESULTS = {
    "kubernetes_kubernetes": {"n_test": 57, "delta": 0.03508771929824561,
                               "ci_lo": -0.03508771929824561, "ci_hi": 0.10526315789473684},
    "microsoft_vscode": {"n_test": 281, "delta": 0.03202846975088966,
                          "ci_lo": -0.0035587188612099642, "ci_hi": 0.06761565836298933},
}


def power_calc_finetune_gate(repo: str, r: dict, n_split_kept: int) -> dict:
    """Same method as scripts/phase2a_corpus_feasibility.py::power_calc, applied to the
    ADR-0027 product-task result (the actual held gate) instead of the W3-era number."""
    d = r["delta"]
    se_now = (r["ci_hi"] - r["ci_lo"]) / (2 * 1.959964)
    sigma1 = se_now * np.sqrt(r["n_test"])
    discordance = sigma1**2 + d**2
    z_a = stats.norm.ppf(0.975)

    def n_for_power(power: float) -> int:
        z_b = stats.norm.ppf(power)
        return int(np.ceil(((z_a + z_b) * sigma1 / d) ** 2))

    test_frac = r["n_test"] / n_split_kept
    return {
        "observed": r,
        "n_split_kept_current": n_split_kept,
        "se_per_obs": round(float(sigma1), 4),
        "implied_discordance_rate": round(float(discordance), 4),
        "n_test_needed": {
            "power_80pct": n_for_power(0.80),
            "power_90pct": n_for_power(0.90),
        },
        "test_fraction_observed": round(test_frac, 4),
        "total_pairs_needed_at_observed_test_frac": {
            "power_80pct": int(np.ceil(n_for_power(0.80) / test_frac)),
            "power_90pct": int(np.ceil(n_for_power(0.90) / test_frac)),
        },
        "sensitivity_if_true_effect_half": {
            "n_test_80pct_power": int(
                np.ceil(((z_a + stats.norm.ppf(0.80)) * sigma1 / (d / 2)) ** 2)
            ),
        },
        "point_estimate_near_zero_flag": bool(abs(d) < 0.03),
        "caveat": "test_fraction_observed is NOT a fixed design ratio -- ADR-0027's split "
        "correction lets product pairs 'ride along' with the gate-stratum chronological walk, "
        "so k8s's 7.7% and vscode's 60.7% test fractions are algorithm artifacts, not targets. "
        "Projecting total-pairs-needed by dividing through this ratio assumes future mining "
        "splits the same way -- flagged, not assumed silently.",
    }


def single_proportion_n_needed(margin: float, p_values: list[float]) -> dict:
    z = stats.norm.ppf(0.975)
    return {
        f"p={p:.2f}": int(np.ceil((z**2) * p * (1 - p) / margin**2)) for p in p_values
    }


def live_index_numbers(repo: str) -> set[int]:
    meta = joblib.load(str(MODELS_DIR / f"dup_index_{repo}_bge" / "meta.pkl"))
    return {int(n) for n in meta["issue_numbers"]}


def measurability_now(repo: str, gold: pd.DataFrame, live_set: set[int]) -> dict:
    prod = gold[(gold["repo"] == repo) & (gold["stratum"] == "product")]
    in_live = prod.apply(
        lambda r: int(r["query_number"]) in live_set and int(r["original_number"]) in live_set,
        axis=1,
    )
    return {
        "product_stratum_total": int(len(prod)),
        "usable_now_against_live_index": int(in_live.sum()),
        "note": "the live-serving index is an off-the-shelf pretrained embedder "
        "(BAAI/bge-base-en-v1.5, src/triage_iq/models/similar_issues.py), never trained on "
        "gold pairs -- the w3-retry train/val/test split (which exists to prevent leakage for "
        "the FINE-TUNED variant's training) carries zero leakage risk here. Every in-range "
        "product pair is usable for measurement regardless of its split label.",
    }


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    split = pd.read_parquet(SPLIT_PATH)
    report: dict = {"generated_by": "scripts/phaseC_power_and_liveindex.py", "repos": {}}

    for repo in ("kubernetes_kubernetes", "microsoft_vscode"):
        sp = split[(split["repo"] == repo) & (split["stratum"] == "product")]
        n_split_kept = int(len(sp))
        log.info("[%s] fine-tune gate power calc (n_split_kept=%d) ...", repo, n_split_kept)
        gate_power = power_calc_finetune_gate(repo, PRODUCT_RESULTS[repo], n_split_kept)

        live_set = live_index_numbers(repo)
        log.info("[%s] live-index measurability-now ...", repo)
        meas_now = measurability_now(repo, gold, live_set)

        entry = {
            "finetune_gate_power": gate_power,
            "live_index_measurability_now": meas_now,
            "live_index_size": len(live_set),
        }

        if repo == "kubernetes_kubernetes":
            margin_targets = single_proportion_n_needed(0.05, [0.25, 0.30, 0.50])
            entry["measure_live_recall_power"] = {
                "target_ci_halfwidth": 0.05,
                "n_needed_by_prior_p": margin_targets,
                "n_available_now": meas_now["usable_now_against_live_index"],
                "gap_at_p0.30": max(
                    0, margin_targets["p=0.30"] - meas_now["usable_now_against_live_index"]
                ),
                "gap_at_p0.50_conservative": max(
                    0, margin_targets["p=0.50"] - meas_now["usable_now_against_live_index"]
                ),
                "note": "k8s ADR-0028 flagged this UNMEASURABLE (0 test-split pairs). It is in "
                "fact near-measurable NOW: 277 product pairs already fall in the live index's "
                "number range and carry no leakage risk (see live_index_measurability_now). "
                "p priors are informed by the v2-index product baseline R@5 (0.228, a smaller/"
                "easier live index plausibly measures somewhat higher per ADR-0027's index-size "
                "note) -- true live p is exactly what's unmeasured, hence a range.",
            }

        report["repos"][repo] = entry
        log.info(
            "[%s] gate: %d test pairs needed (80%%) | live-measure: %d available now",
            repo, gate_power["n_test_needed"]["power_80pct"], meas_now["usable_now_against_live_index"],
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote %s", OUTPUT_PATH)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
