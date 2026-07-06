"""Gold-set / training-data overlap audit (leakage investigation, 2026-07).

Quantifies contamination between the triage gold set (data/gold_triage_plans.parquet,
n=119) and every training-data source, three ways:

  1. ID overlap        — exact (repo, number) intersection with classifier_train,
                         temporal_train, and the W3 retrieval-train pair numbers.
  2. Content-hash      — sha256 of normalized title+body; catches identical text
                         re-filed under a different issue number.
  3. Embedding near-dup — max BGE cosine similarity of each gold issue against all
                         training issues (vectors reconstructed from the saved
                         dup_index_{repo}_bge FAISS index; no re-encoding).

Gold rows are split into cohorts: original_60 (curated by 10_curate_triage_gold.py,
never disjointness-checked) vs w5_added (passed assert_gold_disjoint_from_train at
ingest). The whole point of the audit is that these two cohorts had different
admission criteria.

Read-only: writes reports/gold_leakage_overlap.json + console summary, touches nothing
else.

Usage (from repo root):
    python scripts/verify_gold_train_overlap.py
    python scripts/verify_gold_train_overlap.py --near-dup-thresholds 0.90 0.95
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
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
logger = logging.getLogger("verify_gold_train_overlap")

GOLD_PATH = ROOT / "data" / "gold_triage_plans.parquet"
W3_SPLIT_PATH = ROOT / "data" / "w3_split.parquet"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"
OUT_JSON = ROOT / "reports" / "gold_leakage_overlap.json"

REPOS = {"microsoft/vscode": "microsoft_vscode", "kubernetes/kubernetes": "kubernetes_kubernetes"}

BGE_INDEX_DIRS = {
    "microsoft/vscode": ROOT / "data" / "models" / "dup_index_microsoft_vscode_bge",
    "kubernetes/kubernetes": ROOT / "data" / "models" / "dup_index_kubernetes_kubernetes_bge",
}


def normalize_text(title: str, body: str) -> str:
    """Lowercase, collapse whitespace. Same normalization for gold and train sides."""
    combined = f"{title} {body}".lower()
    return re.sub(r"\s+", " ", combined).strip()


def content_hash(title: str, body: str) -> str:
    return hashlib.sha256(normalize_text(title, body).encode("utf-8")).hexdigest()


def load_split_numbers(repo_slug: str, suffix: str) -> set[int]:
    path = ROOT / "data" / "processed" / f"{repo_slug}_{suffix}.parquet"
    if not path.exists():
        logger.warning("%s missing — treating as empty set", path)
        return set()
    return set(pd.read_parquet(path, columns=["number"])["number"].astype(int))


def load_retrieval_train_numbers(repo_slug: str) -> set[int]:
    """Union of query/original numbers in the W3 retrieval fine-tune train split."""
    if not W3_SPLIT_PATH.exists():
        logger.warning("w3_split.parquet missing — retrieval-train set empty")
        return set()
    df = pd.read_parquet(W3_SPLIT_PATH)
    train = df[(df["repo"] == repo_slug) & (df["split"] == "train")]
    return set(train["query_number"].astype(int)) | set(train["original_number"].astype(int))


def load_cqr_calibration_numbers(repo_slug: str, cal_frac: float = 0.30) -> set[int]:
    """Reproduce scripts/10_calibrate_cqr.py's calibration slice.

    cal = first int(cal_frac * n) rows of temporal_test, resolution_hours > 0,
    sorted by created_at ascending. Deployed config uses cal_frac=0.30 for both
    repos (data/models/cqr_conformal_adjustments.json).
    """
    path = ROOT / "data" / "processed" / f"{repo_slug}_temporal_test.parquet"
    if not path.exists():
        return set()
    test = pd.read_parquet(path)
    test = test[test["resolution_hours"] > 0].sort_values("created_at").reset_index(drop=True)
    n_cal = int(len(test) * cal_frac)
    return set(test.iloc[:n_cal]["number"].astype(int))


def load_train_texts(repo_slug: str, numbers: set[int]) -> pd.DataFrame:
    """Rows of the full issues parquet restricted to the given training numbers."""
    path = ROOT / "data" / "processed" / f"issues_{repo_slug}.parquet"
    df = pd.read_parquet(path, columns=["number", "title", "body_clean"])
    df["number"] = df["number"].astype(int)
    return df[df["number"].isin(numbers)].reset_index(drop=True)


def load_bge_vectors(repo: str) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct all stored BGE vectors + issue numbers from the saved FAISS index.

    Vectors were L2-normalized at build time, so inner product == cosine.
    """
    idx_dir = BGE_INDEX_DIRS[repo]
    meta = joblib.load(str(idx_dir / "meta.pkl"))
    index = faiss.read_index(str(idx_dir / "index.faiss"))
    vecs = index.reconstruct_n(0, index.ntotal)
    numbers = np.asarray(meta["issue_numbers"], dtype=np.int64)
    assert len(numbers) == vecs.shape[0], "index/meta length mismatch"
    return vecs, numbers


def assign_cohort(gold: pd.DataFrame) -> pd.Series:
    """original_60 rows predate W5 ingest and lack related_issue_numbers."""
    if "related_issue_numbers" not in gold.columns:
        return pd.Series(["original_60"] * len(gold), index=gold.index)
    has_related_col = gold["related_issue_numbers"].apply(
        lambda v: isinstance(v, (list, np.ndarray))
    )
    return has_related_col.map({True: "w5_added", False: "original_60"})


def id_overlap(
    gold_repo: pd.DataFrame, train_sets: dict[str, set[int]]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, numbers in train_sets.items():
        hits = gold_repo[gold_repo["number"].isin(numbers)]
        out[name] = {
            "n_overlap": int(len(hits)),
            "by_cohort": hits["cohort"].value_counts().to_dict(),
            "numbers": sorted(int(n) for n in hits["number"]),
        }
    return out


def hash_overlap(
    gold_repo: pd.DataFrame, train_texts: pd.DataFrame, train_name: str
) -> dict[str, Any]:
    """Exact normalized-text matches between gold and a training set.

    Split into same-number (subset of ID overlap) and cross-number (re-filed dup).
    """
    train_hashes: dict[str, list[int]] = {}
    for _, row in train_texts.iterrows():
        h = content_hash(str(row["title"]), str(row["body_clean"]))
        train_hashes.setdefault(h, []).append(int(row["number"]))

    same_number: list[dict[str, Any]] = []
    cross_number: list[dict[str, Any]] = []
    for _, row in gold_repo.iterrows():
        h = content_hash(str(row["title"]), str(row["body_clean"]))
        matches = train_hashes.get(h, [])
        for m in matches:
            rec = {"gold_number": int(row["number"]), "train_number": m, "cohort": row["cohort"]}
            (same_number if m == int(row["number"]) else cross_number).append(rec)

    return {
        "train_set": train_name,
        "n_same_number": len(same_number),
        "n_cross_number": len(cross_number),
        "cross_number_pairs": cross_number,
    }


def near_dup_scan(
    gold_repo: pd.DataFrame,
    train_numbers: set[int],
    vecs: np.ndarray,
    vec_numbers: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    """Max cosine similarity of each gold issue vs all training issues (BGE vectors).

    Same-number pairs are excluded — those are ID overlap, counted separately.
    """
    num_to_row = {int(n): i for i, n in enumerate(vec_numbers)}

    gold_rows, gold_missing = [], []
    for n in gold_repo["number"].astype(int):
        (gold_rows.append(num_to_row[n]) if n in num_to_row else gold_missing.append(n))

    train_rows = [num_to_row[n] for n in train_numbers if n in num_to_row]
    train_missing = len(train_numbers) - len(train_rows)
    if gold_missing:
        logger.warning("%d gold issues not in BGE index: %s", len(gold_missing), gold_missing)
    if train_missing:
        logger.warning("%d training issues not in BGE index", train_missing)

    gold_vecs = vecs[gold_rows]  # (g, d), L2-normalized
    train_vecs = vecs[train_rows]  # (t, d)
    train_nums = vec_numbers[train_rows]
    sims = gold_vecs @ train_vecs.T  # cosine, (g, t)

    # Mask self-pairs (same issue number) so they don't count as near-dups
    gold_nums = (
        gold_repo["number"]
        .astype(int)
        .to_numpy()[
            [i for i, n in enumerate(gold_repo["number"].astype(int)) if int(n) in num_to_row]
        ]
    )
    for gi, gnum in enumerate(gold_nums):
        self_cols = np.where(train_nums == gnum)[0]
        sims[gi, self_cols] = -1.0

    max_idx = sims.argmax(axis=1)
    max_sim = sims.max(axis=1)

    per_gold = [
        {
            "gold_number": int(gnum),
            "cohort": str(gold_repo.iloc[gi]["cohort"]),
            "max_train_cosine": round(float(max_sim[gi]), 4),
            "nearest_train_number": int(train_nums[max_idx[gi]]),
        }
        for gi, gnum in enumerate(gold_nums)
    ]

    threshold_counts = {
        str(t): {
            "n_gold_over_threshold": int((max_sim >= t).sum()),
            "by_cohort": pd.Series([p["cohort"] for p in per_gold if p["max_train_cosine"] >= t])
            .value_counts()
            .to_dict(),
        }
        for t in thresholds
    }

    return {
        "n_gold_scanned": len(gold_nums),
        "n_gold_missing_from_index": len(gold_missing),
        "n_train_vectors": len(train_rows),
        "n_train_missing_from_index": train_missing,
        "max_cosine_distribution": {
            "p50": round(float(np.percentile(max_sim, 50)), 4),
            "p90": round(float(np.percentile(max_sim, 90)), 4),
            "max": round(float(max_sim.max()), 4),
        },
        "threshold_counts": threshold_counts,
        "pairs_over_lowest_threshold": sorted(
            (p for p in per_gold if p["max_train_cosine"] >= min(thresholds)),
            key=lambda p: -p["max_train_cosine"],
        ),
    }


def check_eval_set_matches_gold(gold: pd.DataFrame) -> dict[str, Any]:
    """Confirm eval/eval_set.jsonl rows are exactly the gold parquet rows."""
    if not EVAL_SET_PATH.exists():
        return {"checked": False, "reason": "eval_set.jsonl missing"}
    eval_keys = set()
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                eval_keys.add((rec.get("repo"), int(rec.get("number", -1))))
    gold_keys = set(zip(gold["repo"], gold["number"].astype(int), strict=True))
    return {
        "checked": True,
        "n_eval_set": len(eval_keys),
        "n_gold": len(gold_keys),
        "eval_minus_gold": sorted(map(str, eval_keys - gold_keys)),
        "gold_minus_eval": sorted(map(str, gold_keys - eval_keys)),
        "identical": eval_keys == gold_keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--near-dup-thresholds", nargs="+", type=float, default=[0.90, 0.95])
    args = parser.parse_args()
    thresholds = sorted(args.near_dup_thresholds)

    gold = pd.read_parquet(GOLD_PATH)
    gold["number"] = gold["number"].astype(int)
    gold["cohort"] = assign_cohort(gold)
    logger.info(
        "Gold set: n=%d  cohorts=%s  repos=%s",
        len(gold),
        gold["cohort"].value_counts().to_dict(),
        gold["repo"].value_counts().to_dict(),
    )

    report: dict[str, Any] = {
        "gold_total": int(len(gold)),
        "cohort_counts": gold["cohort"].value_counts().to_dict(),
        "eval_set_vs_gold": check_eval_set_matches_gold(gold),
        "near_dup_thresholds": thresholds,
        "repos": {},
    }

    for repo, slug in REPOS.items():
        logger.info("=== %s ===", repo)
        gold_repo = gold[gold["repo"] == repo].reset_index(drop=True)

        train_sets = {
            "classifier_train": load_split_numbers(slug, "classifier_train"),
            "temporal_train": load_split_numbers(slug, "temporal_train"),
            "retrieval_train_w3": load_retrieval_train_numbers(slug),
            # Secondary: calibration/model-selection sets, reported for completeness
            "classifier_val": load_split_numbers(slug, "classifier_val"),
            "temporal_val": load_split_numbers(slug, "temporal_val"),
            "cqr_calibration_slice": load_cqr_calibration_numbers(slug),
        }
        for name, s in train_sets.items():
            logger.info("  %s: n=%d", name, len(s))

        ids = id_overlap(gold_repo, train_sets)

        hashes = {}
        for name in ("classifier_train", "temporal_train"):
            train_texts = load_train_texts(slug, train_sets[name])
            hashes[name] = hash_overlap(gold_repo, train_texts, name)

        vecs, vec_numbers = load_bge_vectors(repo)
        near_dups = {
            name: near_dup_scan(gold_repo, train_sets[name], vecs, vec_numbers, thresholds)
            for name in ("classifier_train", "temporal_train")
        }

        report["repos"][repo] = {
            "n_gold": int(len(gold_repo)),
            "gold_cohorts": gold_repo["cohort"].value_counts().to_dict(),
            "train_set_sizes": {k: len(v) for k, v in train_sets.items()},
            "id_overlap": ids,
            "hash_overlap": hashes,
            "near_dup": near_dups,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote %s", OUT_JSON)

    print("\n" + "=" * 72)
    print("GOLD/TRAIN OVERLAP SUMMARY")
    print("=" * 72)
    ev = report["eval_set_vs_gold"]
    if ev.get("checked"):
        print(
            f"eval_set.jsonl == gold parquet: {ev['identical']} "
            f"(eval n={ev['n_eval_set']}, gold n={ev['n_gold']})"
        )
    for repo, r in report["repos"].items():
        print(f"\n{repo}  (gold n={r['n_gold']}, cohorts {r['gold_cohorts']})")
        for name, o in r["id_overlap"].items():
            tag = " <-- LEAK" if o["n_overlap"] and name.endswith("_train") else ""
            tag = " <-- LEAK" if o["n_overlap"] and "train" in name else tag
            print(
                f"  ID overlap vs {name:22s} (n={r['train_set_sizes'][name]:6d}): "
                f"{o['n_overlap']:3d}  {o['by_cohort']}{tag}"
            )
        for name, h in r["hash_overlap"].items():
            print(
                f"  Hash overlap vs {name:20s}: same-number={h['n_same_number']}, "
                f"cross-number={h['n_cross_number']}"
            )
        for name, nd in r["near_dup"].items():
            counts = {t: c["n_gold_over_threshold"] for t, c in nd["threshold_counts"].items()}
            print(
                f"  Near-dup vs {name:24s}: scanned={nd['n_gold_scanned']}, "
                f"max-cos dist p50/p90/max = {nd['max_cosine_distribution']['p50']}/"
                f"{nd['max_cosine_distribution']['p90']}/{nd['max_cosine_distribution']['max']}, "
                f"over-threshold={counts}"
            )
    print()


if __name__ == "__main__":
    main()
