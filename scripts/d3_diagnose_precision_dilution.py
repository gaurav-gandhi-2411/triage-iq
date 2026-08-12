"""D3 k8s regression diagnostic, candidate A (precision dilution), zero-cost check.

ADR-0048 found k8s_related's D3 fine-tune to be a confirmed, CI-excludes-zero regression
(-15.15pp R@5 on the clean 66-pair VALID subset) and named two undisentangled candidate
mechanisms. This script runs the cheapest of the two tests GG specified: no retraining, just
re-scoring the ALREADY-TRAINED k8s model on the 84 pairs the strict rubric excluded (label !=
"VALID" in reports/track2_k8s_clean_eval.json) and comparing that delta to the already-measured
delta on the 66 VALID pairs.

If the fine-tuned model improved (or held flat) on the EXCLUDE population while degrading on
VALID, that's near-direct confirmation of the precision-dilution mechanism: the MNRL loss pulled
the embedding space toward exactly the invalid pairs it was (mistakenly) trained to treat as
positive, at the expense of pairs a good retriever should actually surface.

Same harness discipline as d3_eval_finetuned.py: current live-serving-matching baseline index,
full untruncated query text from the processed corpus, leakage guard re-asserted before scoring.

Reads:  reports/track2_k8s_clean_eval.json  (150 pairs, VALID/EXCLUDE_* labels)
        reports/d1_eval_set_k8s_related.json (150 pairs, query text)
        data/models/dup_index_kubernetes_kubernetes_bge  (baseline)
        data/models/d3_finetuned_k8s_related              (fine-tuned)
Writes: reports/d3_diagnose_precision_dilution_k8s_related.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from d3_assert_leakage_guard import assert_task_disjoint
from d3_eval_finetuned import (
    K8S_CLEAN_EVAL,
    REPORTS,
    build_finetuned_index,
    hit_vectors,
    load_baseline,
    load_body_lookup,
    score_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TASK = "k8s_related"
REPO = "kubernetes_kubernetes"
MODEL_DIR = "data/models/d3_finetuned_k8s_related"


def main() -> None:
    assert_task_disjoint(TASK)

    all_pairs = json.loads((REPORTS / "d1_eval_set_k8s_related.json").read_text(encoding="utf-8"))
    clean = json.loads(K8S_CLEAN_EVAL.read_text(encoding="utf-8"))["pairs"]
    label_by_key = {(int(p["query_number"]), int(p["target_number"])): p["label"] for p in clean}

    def key(p: dict) -> tuple[int, int]:
        return (int(p["query_number"]), int(p["original_number"]))

    valid_pairs = [p for p in all_pairs if label_by_key.get(key(p)) == "VALID"]
    exclude_pairs = [p for p in all_pairs if label_by_key.get(key(p), "").startswith("EXCLUDE")]
    unlabeled = [p for p in all_pairs if key(p) not in label_by_key]
    if unlabeled:
        raise SystemExit(
            f"{len(unlabeled)} eval pairs have no track2 label -- d1_eval_set_k8s_related.json "
            f"and track2_k8s_clean_eval.json disagree on population. Refusing to proceed."
        )
    logger.info(
        "population split: %d VALID, %d EXCLUDE (of %d total)",
        len(valid_pairs), len(exclude_pairs), len(all_pairs),
    )

    by_reason: dict[str, list[dict]] = {}
    for p in exclude_pairs:
        by_reason.setdefault(label_by_key[key(p)], []).append(p)

    body_by_num = load_body_lookup(REPO)
    baseline = load_baseline(REPO)
    logger.info("building fine-tuned index from %s", MODEL_DIR)
    trained = build_finetuned_index(REPO, MODEL_DIR)

    result: dict = {"task": TASK, "repo": REPO, "model_dir": MODEL_DIR}

    for pop_name, pop_pairs in [("valid_66", valid_pairs), ("exclude_84", exclude_pairs)]:
        base_vecs = hit_vectors(baseline, pop_pairs, body_by_num)
        trained_vecs = hit_vectors(trained, pop_pairs, body_by_num)
        score_pairs(pop_name, result, base_vecs, trained_vecs)

    result["exclude_by_reason"] = {}
    for reason, pop_pairs in by_reason.items():
        base_vecs = hit_vectors(baseline, pop_pairs, body_by_num)
        trained_vecs = hit_vectors(trained, pop_pairs, body_by_num)
        sub: dict = {}
        score_pairs(reason, sub, base_vecs, trained_vecs)
        result["exclude_by_reason"][reason] = sub[reason]

    out_path = REPORTS / f"d3_diagnose_precision_dilution_{TASK}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    v5, e5 = result["valid_66"]["recall_at_5"], result["exclude_84"]["recall_at_5"]
    logger.info(
        "R@5 delta -- VALID: %+.4f (CI %s, excludes_zero=%s)  |  EXCLUDE: %+.4f (CI %s, "
        "excludes_zero=%s)",
        v5["delta"], v5["delta_ci95_paired"], v5["excludes_zero"],
        e5["delta"], e5["delta_ci95_paired"], e5["excludes_zero"],
    )
    if e5["delta"] > 0 and v5["delta"] < 0:
        logger.info(
            "DIRECTIONAL SUPPORT for precision dilution: trained model improved on EXCLUDE "
            "while degrading on VALID."
        )
    elif e5["delta"] <= 0 and v5["delta"] < 0:
        logger.info(
            "NO directional support for precision dilution from this signal alone: trained "
            "model degraded on EXCLUDE too (not selectively pulled toward invalid pairs)."
        )


if __name__ == "__main__":
    sys.exit(main())
