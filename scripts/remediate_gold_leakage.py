"""Execute gold-set decontamination (phases 1-2 of gold-set-leakage remediation).

Per docs/investigations/gold-set-leakage.md:

  1. DROP every gold row whose issue number appears in classifier_train,
     temporal_train, or the W3 retrieval-train pair set (recomputed live from the
     split parquets — not read from a stale report), plus the explicit near-dup
     drop decision k8s #14398 (BGE cosine 0.907 to classifier_train #14399;
     re-filed duplicate content that ID-level checks cannot see).
  2. RECONCILE the post-drop gold against the in-flight eval/eval_set.jsonl:
     expected relationship is eval_set == gold + {k8s #14398} (the eval set was
     frozen before the #14398 drop decision).
  3. ENUMERATE vscode backfill candidates from the held-out eval splits with the
     dual admission checks (ID-disjoint from all three training sources AND max
     BGE cosine vs classifier_train ∪ temporal_train < NEAR_DUP_COSINE_MAX=0.90),
     excluding rows already rejected by GG in the W5 labeling round. Candidates
     are written for HUMAN labeling — nothing is ingested into gold here; gold
     admission stays behind scripts/w5_ingest_labeled.py's accept/reject flow.

Idempotent: re-running after a successful run drops zero rows and regenerates
the same candidate file. Default is dry-run; pass --write to modify
data/gold_triage_plans.parquet.

Usage (from repo root):
    python scripts/remediate_gold_leakage.py            # dry-run
    python scripts/remediate_gold_leakage.py --write
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("remediate_gold_leakage")

GOLD_PATH = ROOT / "data" / "gold_triage_plans.parquet"
W3_SPLIT_PATH = ROOT / "data" / "w3_split.parquet"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"
LABELED_CSV = ROOT / "data" / "gold_expansion_candidates_labeled.csv"
BACKFILL_CSV = ROOT / "data" / "gold_backfill_candidates_vscode.csv"
OUT_JSON = ROOT / "reports" / "gold_remediation.json"

REPOS = {"microsoft/vscode": "microsoft_vscode", "kubernetes/kubernetes": "kubernetes_kubernetes"}
EVAL_SPLITS = ["temporal_val", "temporal_test", "classifier_val", "classifier_test"]

# Near-dup admission threshold. Measured background (non-dup) max-cosine tops out
# ~0.85-0.89 while confirmed re-filed dups sit at 0.907-1.0 — bands don't overlap.
# Mirrors eval/test_invariants.py::NEAR_DUP_COSINE_MAX.
NEAR_DUP_COSINE_MAX = 0.90

# Explicit drop decision (2026-07-06): near-dup of classifier_train #14399 at
# cosine 0.907. Not ID-contaminated, so it is not caught by the recomputed union.
NEAR_DUP_DROPS: set[tuple[str, int]] = {("kubernetes/kubernetes", 14398)}


def training_id_sets(slug: str) -> dict[str, set[int]]:
    """Issue-number sets of the three training sources for one repo."""
    sets: dict[str, set[int]] = {}
    for split in ("classifier_train", "temporal_train"):
        path = ROOT / "data" / "processed" / f"{slug}_{split}.parquet"
        sets[split] = set(pd.read_parquet(path, columns=["number"])["number"].astype(int))
    if W3_SPLIT_PATH.exists():
        w3 = pd.read_parquet(W3_SPLIT_PATH)
        train = w3[(w3["repo"] == slug) & (w3["split"] == "train")]
        sets["retrieval_train_w3"] = set(train["query_number"].astype(int)) | set(
            train["original_number"].astype(int)
        )
    else:
        logger.warning("w3_split.parquet missing — retrieval-train set empty")
        sets["retrieval_train_w3"] = set()
    return sets


def load_bge_vectors(slug: str) -> tuple[np.ndarray, dict[int, int]]:
    """Stored L2-normalized BGE vectors + number->row map from the saved index."""
    idx_dir = ROOT / "data" / "models" / f"dup_index_{slug}_bge"
    meta = joblib.load(str(idx_dir / "meta.pkl"))
    index = faiss.read_index(str(idx_dir / "index.faiss"))
    vecs = index.reconstruct_n(0, index.ntotal)
    return vecs, {int(n): i for i, n in enumerate(meta["issue_numbers"])}


def cohort_of(v: Any) -> str:
    return "w5_added" if isinstance(v, (list, np.ndarray)) else "original_60"


def drop_contaminated(gold: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove ID-contaminated rows (recomputed) and the explicit near-dup drops."""
    gold = gold.copy()
    gold["number"] = gold["number"].astype(int)
    gold["cohort"] = (
        gold["related_issue_numbers"].apply(cohort_of)
        if ("related_issue_numbers" in gold.columns)
        else "original_60"
    )

    drop_mask = pd.Series(False, index=gold.index)
    stats: dict[str, Any] = {"per_repo": {}}

    for repo, slug in REPOS.items():
        id_sets = training_id_sets(slug)
        union = (
            id_sets["classifier_train"] | id_sets["temporal_train"] | id_sets["retrieval_train_w3"]
        )
        repo_mask = gold["repo"] == repo
        id_hit = repo_mask & gold["number"].isin(union)
        near_hit = repo_mask & gold.apply(
            lambda r: (r["repo"], int(r["number"])) in NEAR_DUP_DROPS, axis=1
        )
        drop_mask |= id_hit | near_hit

        before = gold[repo_mask]
        stats["per_repo"][repo] = {
            "before": int(repo_mask.sum()),
            "before_by_cohort": before["cohort"].value_counts().to_dict(),
            "dropped_id_contaminated": int(id_hit.sum()),
            "dropped_near_dup": int(near_hit.sum()),
            "dropped_numbers": sorted(int(n) for n in gold[id_hit | near_hit]["number"]),
        }

    clean = gold[~drop_mask].drop(columns=["cohort"]).reset_index(drop=True)
    for repo in REPOS:
        after = clean[clean["repo"] == repo]
        stats["per_repo"][repo]["after"] = int(len(after))
    stats["total_before"] = int(len(gold))
    stats["total_dropped"] = int(drop_mask.sum())
    stats["total_after"] = int(len(clean))
    return clean, stats


def reconcile_with_eval_set(clean: pd.DataFrame) -> dict[str, Any]:
    """Verify eval_set.jsonl == clean gold + NEAR_DUP_DROPS, explicitly."""
    if not EVAL_SET_PATH.exists():
        return {"checked": False, "reason": "eval_set.jsonl missing"}
    eval_keys: set[tuple[str, int]] = set()
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                eval_keys.add((rec["repo"], int(rec["number"])))
    gold_keys = set(zip(clean["repo"], clean["number"].astype(int), strict=True))
    return {
        "checked": True,
        "n_eval_set": len(eval_keys),
        "n_clean_gold": len(gold_keys),
        "eval_minus_gold": sorted(map(str, eval_keys - gold_keys)),
        "gold_minus_eval": sorted(map(str, gold_keys - eval_keys)),
        "reconciles_as_expected": (eval_keys - gold_keys) == {(r, n) for r, n in NEAR_DUP_DROPS}
        and not (gold_keys - eval_keys),
    }


def enumerate_vscode_backfill(clean_gold: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """All eligible vscode candidates passing the dual admission checks.

    NOT ingested into gold — written to BACKFILL_CSV for GG labeling via the
    standard w5_ingest_labeled.py accept/reject flow.
    """
    repo, slug = "microsoft/vscode", "microsoft_vscode"
    parts = [
        pd.read_parquet(ROOT / "data" / "processed" / f"{slug}_{s}.parquet")
        for s in EVAL_SPLITS
        if (ROOT / "data" / "processed" / f"{slug}_{s}.parquet").exists()
    ]
    pool = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["number"])
    pool["number"] = pool["number"].astype(int)
    n_pool = len(pool)

    pool = pool[pool["component"].notna() & (pool["resolution_hours"] > 0)]
    n_labeled_ok = len(pool)

    gold_numbers = set(clean_gold[clean_gold["repo"] == repo]["number"].astype(int))
    pool = pool[~pool["number"].isin(gold_numbers)]

    # Respect GG's W5 rejections — a rejected candidate stays rejected.
    rejected: set[int] = set()
    if LABELED_CSV.exists():
        lab = pd.read_csv(LABELED_CSV)
        rej = lab[(lab["repo"] == repo) & (lab["label_decision"] == "reject")]
        rejected = set(rej["number"].astype(int))
    pool = pool[~pool["number"].isin(rejected)]

    id_sets = training_id_sets(slug)
    train_union = (
        id_sets["classifier_train"] | id_sets["temporal_train"] | id_sets["retrieval_train_w3"]
    )
    n_before_id = len(pool)
    pool = pool[~pool["number"].isin(train_union)]
    n_after_id = len(pool)

    # Near-dup check vs classifier_train ∪ temporal_train using stored vectors
    vecs, num_to_row = load_bge_vectors(slug)
    text_train = id_sets["classifier_train"] | id_sets["temporal_train"]
    train_rows = [num_to_row[n] for n in text_train if n in num_to_row]
    cand_nums = [n for n in pool["number"] if n in num_to_row]
    missing_from_index = sorted(set(pool["number"]) - set(cand_nums))
    cand_rows = [num_to_row[n] for n in cand_nums]

    sims = vecs[cand_rows] @ vecs[train_rows].T
    max_sim = sims.max(axis=1)
    ok_numbers = {n for n, s in zip(cand_nums, max_sim, strict=True) if s < NEAR_DUP_COSINE_MAX}
    n_near_dup_dropped = len(cand_nums) - len(ok_numbers)
    pool = pool[pool["number"].isin(ok_numbers)].reset_index(drop=True)

    pool["repo"] = repo
    pool["body_excerpt"] = pool["body_clean"].fillna("").str[:300]
    pool["label_status"] = "pending"
    pool["label_decision"] = ""
    pool["label_rejection_code"] = ""
    pool["corrected_component"] = ""
    pool["labeler_notes"] = ""

    stats = {
        "pool_unique": n_pool,
        "with_component_and_resolution": n_labeled_ok,
        "excluded_already_in_clean_gold": len(gold_numbers),
        "excluded_w5_rejected": len(rejected),
        "dropped_id_train_overlap": n_before_id - n_after_id,
        "dropped_near_dup_cosine_ge_0.90": n_near_dup_dropped,
        "missing_from_bge_index": missing_from_index,
        "eligible_for_labeling": int(len(pool)),
        "near_dup_threshold": NEAR_DUP_COSINE_MAX,
    }
    return pool, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write", action="store_true", help="Write gold parquet + backfill CSV (default: dry-run)"
    )
    args = parser.parse_args()

    gold = pd.read_parquet(GOLD_PATH)
    clean, drop_stats = drop_contaminated(gold)
    for repo, s in drop_stats["per_repo"].items():
        logger.info(
            "%s: %d -> %d (dropped %d ID-contaminated + %d near-dup)",
            repo,
            s["before"],
            s["after"],
            s["dropped_id_contaminated"],
            s["dropped_near_dup"],
        )

    recon = reconcile_with_eval_set(clean)
    logger.info("Reconciliation vs eval_set.jsonl: %s", recon)

    backfill, backfill_stats = enumerate_vscode_backfill(clean)
    logger.info("vscode backfill candidates eligible for labeling: %d", len(backfill))

    report = {
        "near_dup_threshold": NEAR_DUP_COSINE_MAX,
        "explicit_near_dup_drops": sorted(map(str, NEAR_DUP_DROPS)),
        "drop": drop_stats,
        "reconciliation": recon,
        "vscode_backfill": backfill_stats,
        "written": bool(args.write),
    }

    if args.write:
        clean.to_parquet(GOLD_PATH, index=False)
        logger.info("Wrote %s (%d rows)", GOLD_PATH, len(clean))
        backfill.to_csv(BACKFILL_CSV, index=False)
        logger.info("Wrote %s (%d candidates for GG labeling)", BACKFILL_CSV, len(backfill))
    else:
        logger.info("[DRY-RUN] No files written. Pass --write to apply.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote %s", OUT_JSON)

    print("\n=== REMEDIATION SUMMARY ===")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
