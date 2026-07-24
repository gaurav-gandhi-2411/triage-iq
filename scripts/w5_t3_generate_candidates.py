"""Generate labeling candidate pool for W5 gold-set expansion (n=60 -> n=150).

Produces ~120 candidates (60 per repo, 12 per resolution bucket × 5 buckets) from
held-out eval splits, with TF-IDF and BGE systems 1+2 output pre-computed.

Outputs:
  data/gold_expansion_candidates.parquet  — full data (all columns)
  data/gold_expansion_candidates.csv      — same columns as parquet; includes both
                                             body_excerpt (300-char skim column) and the
                                             full body_clean (required standalone by
                                             scripts/w5_ingest_labeled.py)
  reports/w5_gold_audit.json              — current gold audit + candidate pool stats
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is on path so triage_iq modules resolve
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from triage_iq.models.component_classifier import load_classifier
from triage_iq.models.similar_issues import SimilarIssueRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("w5_t3_generate_candidates")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
CANDIDATES_PER_BUCKET = 12
MAX_PER_COMPONENT_PER_BUCKET = 3

REPOS = {
    "microsoft_vscode": "microsoft/vscode",
    "kubernetes_kubernetes": "kubernetes/kubernetes",
}

REPO_ALIAS = {
    "microsoft/vscode": "vsc",
    "kubernetes/kubernetes": "k8s",
}

EVAL_SPLITS = ["temporal_val", "temporal_test", "classifier_val", "classifier_test"]

CLASSIFIER_PATHS = {
    "microsoft/vscode": REPO_ROOT / "data/models/component_classifier_microsoft_vscode.pkl",
    "kubernetes/kubernetes": REPO_ROOT / "data/models/component_classifier_kubernetes_kubernetes.pkl",
}

BGE_INDEX_PATHS = {
    "microsoft/vscode": REPO_ROOT / "data/models/dup_index_microsoft_vscode_bge",
    "kubernetes/kubernetes": REPO_ROOT / "data/models/dup_index_kubernetes_kubernetes_bge",
}

GOLD_PATH = REPO_ROOT / "data/gold_triage_plans.parquet"
W3_SPLIT_PATH = REPO_ROOT / "data/w3_split.parquet"
OUTPUT_PARQUET = REPO_ROOT / "data/gold_expansion_candidates.parquet"
OUTPUT_CSV = REPO_ROOT / "data/gold_expansion_candidates.csv"
AUDIT_JSON = REPO_ROOT / "reports/w5_gold_audit.json"


# ---------------------------------------------------------------------------
# Resolution bucket helper
# ---------------------------------------------------------------------------

BUCKET_ORDER = ["hours", "days", "weeks", "months", "long"]


def assign_res_bucket(hours: float) -> str:
    """Assign a resolution bucket label based on resolution_hours."""
    if hours < 24:
        return "hours"
    elif hours < 7 * 24:
        return "days"
    elif hours < 30 * 24:
        return "weeks"
    elif hours < 180 * 24:
        return "months"
    else:
        return "long"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_pool(repo_key: str) -> pd.DataFrame:
    """Union of all 4 held-out eval splits, deduped on number."""
    dfs: list[pd.DataFrame] = []
    for split in EVAL_SPLITS:
        path = REPO_ROOT / "data/processed" / f"{repo_key}_{split}.parquet"
        df = pd.read_parquet(path)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["number"])
    logger.info(
        "%s eval pool: %d unique issues before filter", repo_key, len(combined)
    )
    return combined


def load_retrieval_train_numbers(repo_key: str) -> set[int]:
    """Load the retrieval fine-tune's train-split issue numbers for a repo.

    Reads data/w3_split.parquet (the ADR-0016 retrieval fine-tune's train/val/test
    split assignments), filters to ``split == "train"`` and the given repo_key
    (e.g. "kubernetes_kubernetes"), and returns the union of the ``query_number``
    and ``original_number`` columns as an issue-number set. Candidates matching
    any of these numbers must be excluded from the labeling pool so that no gold
    issue can later collide with a future retrieval-train split (disjointness
    guarantee carried forward from the current gold set's discipline).
    """
    if not W3_SPLIT_PATH.exists():
        logger.warning(
            "w3_split.parquet not found at %s — skipping retrieval-train disjointness filter",
            W3_SPLIT_PATH,
        )
        return set()
    split_df = pd.read_parquet(W3_SPLIT_PATH)
    train_df = split_df[(split_df["repo"] == repo_key) & (split_df["split"] == "train")]
    numbers = set(train_df["query_number"].tolist()) | set(train_df["original_number"].tolist())
    return numbers


def load_classifier_train_numbers(repo_key: str) -> set[int]:
    """Load the component classifier's train-split issue numbers for a repo.

    Reads ``data/processed/{repo_key}_classifier_train.parquet`` and returns its
    ``number`` column as an issue-number set. Candidates matching any of these numbers
    must be excluded from the labeling pool — a gold issue that leaks into the
    classifier's train split would silently inflate classifier accuracy evaluated
    against gold (mirrors scripts/w5_ingest_labeled.py's
    ``assert_gold_disjoint_from_train`` check #1).
    """
    path = REPO_ROOT / "data/processed" / f"{repo_key}_classifier_train.parquet"
    if not path.exists():
        logger.warning(
            "%s not found — skipping classifier-train disjointness filter", path
        )
        return set()
    df = pd.read_parquet(path, columns=["number"])
    return set(df["number"].astype(int))


def load_temporal_train_numbers(repo_key: str) -> set[int]:
    """Load the resolution-time model's temporal train-split issue numbers for a repo.

    Reads ``data/processed/{repo_key}_temporal_train.parquet`` and returns its
    ``number`` column as an issue-number set. Mirrors
    scripts/w5_ingest_labeled.py's ``assert_gold_disjoint_from_train`` check #2.
    """
    path = REPO_ROOT / "data/processed" / f"{repo_key}_temporal_train.parquet"
    if not path.exists():
        logger.warning(
            "%s not found — skipping temporal-train disjointness filter", path
        )
        return set()
    df = pd.read_parquet(path, columns=["number"])
    return set(df["number"].astype(int))


def filter_pool(
    df: pd.DataFrame,
    gold_numbers: set[int],
    retrieval_train_numbers: set[int],
    classifier_train_numbers: set[int],
    temporal_train_numbers: set[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep: component not-null, resolution_hours > 0, not in current gold, and not in
    any of the three training-data issue-number sets (retrieval-train, classifier-train,
    temporal-train).

    The three training-set exclusions are each measured independently against the same
    post-gold-filter pool (not sequentially), so every filter's reported drop count
    reflects its own true overlap rather than an order-dependent residual after earlier
    filters already removed some of the same issues. The actual removal applied to the
    pool is the UNION of the three masks — an issue present in more than one training
    set is only removed once, so ``dropped_training_union`` is <= the sum of the three
    individual counts (equality only when the three sets don't overlap).

    Returns the filtered DataFrame plus a stats dict with keys:
    ``dropped_already_in_current_gold``, ``dropped_retrieval_train_overlap``,
    ``dropped_classifier_train_overlap``, ``dropped_temporal_train_overlap``,
    ``dropped_training_union``.
    """
    mask = df["component"].notna() & (df["resolution_hours"] > 0)
    df = df[mask].copy()

    before_gold = len(df)
    df = df[~df["number"].isin(gold_numbers)]
    dropped_gold = before_gold - len(df)
    logger.info(
        "After current-gold filter: %d issues (removed %d already in current gold)",
        len(df),
        dropped_gold,
    )

    in_retrain = df["number"].isin(retrieval_train_numbers)
    in_classifier = df["number"].isin(classifier_train_numbers)
    in_temporal = df["number"].isin(temporal_train_numbers)

    dropped_retrain = int(in_retrain.sum())
    dropped_classifier = int(in_classifier.sum())
    dropped_temporal = int(in_temporal.sum())

    union_mask = in_retrain | in_classifier | in_temporal
    dropped_union = int(union_mask.sum())
    sum_individual = dropped_retrain + dropped_classifier + dropped_temporal

    logger.info(
        "Training-disjointness exclusions (independent counts against post-gold pool of "
        "%d): retrieval_train=%d, classifier_train=%d, temporal_train=%d, union=%d "
        "(sum=%d, overcounted by %d due to overlap across the three sets)",
        len(df),
        dropped_retrain,
        dropped_classifier,
        dropped_temporal,
        dropped_union,
        sum_individual,
        sum_individual - dropped_union,
    )

    df = df[~union_mask]
    logger.info("After training-disjointness filters: %d issues remain", len(df))

    stats = {
        "dropped_already_in_current_gold": dropped_gold,
        "dropped_retrieval_train_overlap": dropped_retrain,
        "dropped_classifier_train_overlap": dropped_classifier,
        "dropped_temporal_train_overlap": dropped_temporal,
        "dropped_training_union": dropped_union,
    }
    return df, stats


# ---------------------------------------------------------------------------
# Stratified sampling with component diversity
# ---------------------------------------------------------------------------

def sample_with_diversity(
    bucket_df: pd.DataFrame,
    n: int,
    gold_component_counts: Counter,
    max_per_component: int = MAX_PER_COMPONENT_PER_BUCKET,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample n issues from a bucket, maximising component diversity.

    Algorithm:
    1. Rank components by ascending gold frequency (prefer under-represented).
    2. Greedily draw 1 issue per component in that order, cycling until n reached.
    3. Cap any single component at max_per_component within this bucket.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    if len(bucket_df) <= n:
        # Not enough candidates — return all
        logger.warning("Only %d candidates in bucket (need %d); using all", len(bucket_df), n)
        return bucket_df.copy()

    # Build component -> available indices map (shuffled for randomness)
    comp_to_rows: dict[str, list[int]] = {}
    for comp, grp in bucket_df.groupby("component"):
        idxs = grp.index.tolist()
        rng.shuffle(idxs)
        comp_to_rows[str(comp)] = idxs

    # Sort components by ascending gold count (under-represented first)
    sorted_comps = sorted(
        comp_to_rows.keys(),
        key=lambda c: (gold_component_counts.get(c, 0), c),
    )

    selected: list[int] = []
    component_drawn: Counter = Counter()

    # Round-robin until we have n or exhausted everything
    while len(selected) < n:
        made_progress = False
        for comp in sorted_comps:
            if len(selected) >= n:
                break
            if component_drawn[comp] >= max_per_component:
                continue
            pool = comp_to_rows[comp]
            if not pool:
                continue
            idx = pool.pop(0)
            selected.append(idx)
            component_drawn[comp] += 1
            made_progress = True
        if not made_progress:
            break  # All components exhausted or capped

    return bucket_df.loc[selected].copy()


def stratified_sample(
    df: pd.DataFrame,
    gold_component_counts: Counter,
    repo_alias: str,
) -> pd.DataFrame:
    """Sample CANDIDATES_PER_BUCKET × 5 buckets with component diversity."""
    df = df.copy()
    df["res_bucket"] = df["resolution_hours"].apply(assign_res_bucket)

    rng = np.random.default_rng(RANDOM_STATE)
    parts: list[pd.DataFrame] = []

    for bucket in BUCKET_ORDER:
        bdf = df[df["res_bucket"] == bucket]
        sampled = sample_with_diversity(
            bdf, CANDIDATES_PER_BUCKET, gold_component_counts, rng=rng
        )
        parts.append(sampled)
        logger.info(
            "  bucket=%s  available=%d  sampled=%d  components=%s",
            bucket,
            len(bdf),
            len(sampled),
            dict(Counter(sampled["component"].tolist())),
        )

    result = pd.concat(parts, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# System 1: TF-IDF component predictions
# ---------------------------------------------------------------------------

def add_tfidf_predictions(
    df: pd.DataFrame,
    repo: str,
) -> pd.DataFrame:
    """Add tfidf_top1..3 and tfidf_top1_conf..3_conf columns."""
    clf_path = CLASSIFIER_PATHS[repo]
    if not clf_path.exists():
        logger.warning("Classifier not found at %s — skipping TF-IDF columns", clf_path)
        for i in [1, 2, 3]:
            df[f"tfidf_top{i}"] = None
            df[f"tfidf_top{i}_conf"] = None
        return df

    # load_classifier() dispatches on the pkl's model_kind marker (ADR-0036).
    clf = load_classifier(clf_path.parent, repo.replace("/", "_"))
    classes = clf.classes_()

    texts = (
        df["title"].fillna("").str.strip()
        + ". "
        + df["body_clean"].fillna("").str.strip()
    )
    proba = clf.predict_proba(texts)  # (n, num_classes)

    top3_idx = np.argsort(proba, axis=1)[:, ::-1][:, :3]
    for rank, col_suffix in enumerate([1, 2, 3]):
        df[f"tfidf_top{col_suffix}"] = classes[top3_idx[:, rank]]
        df[f"tfidf_top{col_suffix}_conf"] = proba[
            np.arange(len(proba)), top3_idx[:, rank]
        ].round(4)

    logger.info("TF-IDF predictions added for %d issues (%s)", len(df), repo)
    return df


# ---------------------------------------------------------------------------
# System 2: BGE similar-issue retrieval
# ---------------------------------------------------------------------------

def add_bge_similar(
    df: pd.DataFrame,
    repo: str,
) -> pd.DataFrame:
    """Add bge_similar_1..3 and bge_similar_1_score..3_score columns."""
    idx_path = BGE_INDEX_PATHS[repo]
    if not idx_path.exists():
        logger.warning("BGE index not found at %s — skipping BGE columns", idx_path)
        for i in [1, 2, 3]:
            df[f"bge_similar_{i}"] = None
            df[f"bge_similar_{i}_score"] = None
        return df

    retriever = SimilarIssueRetriever.load(str(idx_path))

    similar_rows: list[dict] = []
    for _, row in df.iterrows():
        text = f"{str(row['title']).strip()}. {str(row['body_clean']).strip()}"
        hits = retriever.retrieve(text, k=3, exclude_number=int(row["number"]))
        # Pad to 3 if fewer hits returned
        while len(hits) < 3:
            hits.append({"number": None, "score": None})
        similar_rows.append({
            "bge_similar_1": hits[0]["number"],
            "bge_similar_1_score": round(hits[0]["score"], 4) if hits[0]["score"] is not None else None,
            "bge_similar_2": hits[1]["number"],
            "bge_similar_2_score": round(hits[1]["score"], 4) if hits[1]["score"] is not None else None,
            "bge_similar_3": hits[2]["number"],
            "bge_similar_3_score": round(hits[2]["score"], 4) if hits[2]["score"] is not None else None,
        })

    bge_df = pd.DataFrame(similar_rows, index=df.index)
    df = pd.concat([df, bge_df], axis=1)
    logger.info("BGE similar-issue columns added for %d issues (%s)", len(df), repo)
    return df


# ---------------------------------------------------------------------------
# Stratum label
# ---------------------------------------------------------------------------

def build_stratum(row: pd.Series) -> str:
    """Build stratum label: e.g. 'vsc-hours-debug'."""
    alias = REPO_ALIAS.get(str(row.get("repo", "")), "unk")
    bucket = str(row.get("res_bucket", "unk"))
    comp = str(row.get("component", "unk")).replace("/", "-").replace(" ", "_")
    return f"{alias}-{bucket}-{comp}"


# ---------------------------------------------------------------------------
# Gold audit helpers
# ---------------------------------------------------------------------------

def build_gold_audit(gold: pd.DataFrame) -> dict:
    """Build the current_gold section of the audit JSON."""
    gold = gold.copy()
    gold["bucket"] = (gold["actual_resolution_days"] * 24).apply(assign_res_bucket)
    gold["year"] = pd.to_datetime(gold["created_at"]).dt.year

    return {
        "total": len(gold),
        "per_repo": gold["repo"].value_counts().to_dict(),
        "era_distribution": {str(k): int(v) for k, v in gold["year"].value_counts().sort_index().items()},
        "component_counts": gold["gold_component"].value_counts().to_dict(),
        "res_bucket_counts": gold["bucket"].value_counts().to_dict(),
        "priority_counts": gold["gold_priority"].value_counts().to_dict(),
        "unique_components": gold["gold_component"].nunique(),
        "gaps": [
            "95% issues from 2014-2016 — zero 2017-2025 representation",
            "Resolution bucket '>30d' is too coarse — months (30-180d) has only 8 issues vs long (180d+) 12 issues",
            "debug component is 13% of gold (8/60) — overrepresented vs corpus",
            "medium priority only 10% (6/60) — inferred from resolution speed, not label",
            "13 vscode components, 16 k8s components — many tail components absent",
        ],
    }


def build_candidate_audit(
    candidates: pd.DataFrame,
    gold_components: set[str],
) -> dict:
    """Build the candidate_pool section of the audit JSON."""
    new_components = sorted(
        set(candidates["component"].dropna().unique()) - gold_components
    )
    return {
        "total": len(candidates),
        "per_repo": candidates["repo"].value_counts().to_dict(),
        "res_bucket_counts": candidates["res_bucket"].value_counts().to_dict(),
        "component_counts": candidates["component"].value_counts().to_dict(),
        "new_components_introduced": new_components,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    # --- Load gold ---
    gold = pd.read_parquet(GOLD_PATH)
    gold_components: set[str] = set(gold["gold_component"].dropna().unique())
    gold_component_counts = Counter(gold["gold_component"].dropna().tolist())

    all_candidates: list[pd.DataFrame] = []
    filter_stats: dict[str, dict[str, int]] = {}

    for repo_key, repo_name in REPOS.items():
        logger.info("=== Processing %s ===", repo_name)
        gold_numbers_repo = set(gold[gold["repo"] == repo_name]["number"].tolist())
        retrieval_train_numbers = load_retrieval_train_numbers(repo_key)
        classifier_train_numbers = load_classifier_train_numbers(repo_key)
        temporal_train_numbers = load_temporal_train_numbers(repo_key)

        pool = load_eval_pool(repo_key)
        pool, stats = filter_pool(
            pool,
            gold_numbers_repo,
            retrieval_train_numbers,
            classifier_train_numbers,
            temporal_train_numbers,
        )
        filter_stats[repo_name] = stats
        pool["repo"] = repo_name

        # Stratified sample with diversity
        sampled = stratified_sample(
            pool,
            gold_component_counts,
            repo_alias=REPO_ALIAS[repo_name],
        )
        logger.info("%s: %d candidates after stratified sampling", repo_name, len(sampled))

        # TF-IDF predictions
        sampled = add_tfidf_predictions(sampled, repo_name)

        # BGE similar issues — encode per issue (slow but correct)
        logger.info("Running BGE retrieval for %s (%d issues)…", repo_name, len(sampled))
        sampled = add_bge_similar(sampled, repo_name)

        all_candidates.append(sampled)

    # --- Combine ---
    candidates = pd.concat(all_candidates, ignore_index=True)

    # --- Add derived columns ---
    candidates["stratum"] = candidates.apply(build_stratum, axis=1)
    candidates["label_status"] = "pending"
    candidates["body_excerpt"] = candidates["body_clean"].fillna("").str[:300]

    # --- Save outputs ---
    REPO_ROOT.joinpath("data").mkdir(parents=True, exist_ok=True)
    REPO_ROOT.joinpath("reports").mkdir(parents=True, exist_ok=True)

    candidates.to_parquet(OUTPUT_PARQUET, index=False)
    logger.info("Saved parquet: %s", OUTPUT_PARQUET)

    # CSV keeps both body_excerpt (300-char skim column, for human review) and the
    # full body_clean (required by scripts/w5_ingest_labeled.py's validate_accepted_rows
    # / extract_related_issue_numbers — ingestion reads the labeled CSV standalone and
    # must not depend on re-joining against the parquet). Kept as two distinct columns
    # (not deduped) so the excerpt stays skimmable while body_clean remains available.
    candidates.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved CSV: %s", OUTPUT_CSV)

    # --- Audit JSON ---
    gold_audit = build_gold_audit(gold)
    candidate_audit = build_candidate_audit(candidates, gold_components)

    audit = {
        "current_gold": gold_audit,
        "candidate_pool": candidate_audit,
        "pool_filter_stats": filter_stats,
        "sampling_methodology": (
            "Stratified 12-per-bucket × 5 buckets from held-out eval splits "
            "(temporal_val + temporal_test + classifier_val + classifier_test). "
            "Component diversity constraint: prefer underrepresented components, "
            "max 3 per component per bucket. Seed 42. Pool additionally excludes any "
            "issue number already in the current gold set, plus the union of three "
            "training-data issue-number sets: (1) data/w3_split.parquet's train split "
            "(query_number or original_number, per repo — retrieval-train disjointness), "
            "(2) data/processed/{repo}_classifier_train.parquet's number column "
            "(classifier-train disjointness), and (3) "
            "data/processed/{repo}_temporal_train.parquet's number column "
            "(temporal-train disjointness). Mirrors scripts/w5_ingest_labeled.py's "
            "assert_gold_disjoint_from_train three-way check."
        ),
    }

    with open(AUDIT_JSON, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    logger.info("Saved audit JSON: %s", AUDIT_JSON)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("CANDIDATE POOL SUMMARY")
    print("=" * 60)
    print(f"Total candidates: {len(candidates)}")
    print()
    for repo_name in REPOS.values():
        repo_df = candidates[candidates["repo"] == repo_name]
        print(f"{repo_name}: {len(repo_df)} candidates")
        bucket_counts = repo_df["res_bucket"].value_counts()
        for b in BUCKET_ORDER:
            print(f"  {b:8s}: {bucket_counts.get(b, 0)}")
        stats = filter_stats.get(repo_name, {})
        print(f"  dropped (already in current gold):       {stats.get('dropped_already_in_current_gold', 0)}")
        print(f"  dropped (retrieval-train overlap):       {stats.get('dropped_retrieval_train_overlap', 0)}")
        print(f"  dropped (classifier-train overlap):      {stats.get('dropped_classifier_train_overlap', 0)}")
        print(f"  dropped (temporal-train overlap):        {stats.get('dropped_temporal_train_overlap', 0)}")
        print(f"  dropped (training union, net):           {stats.get('dropped_training_union', 0)}")
    print()

    new_comps = candidate_audit["new_components_introduced"]
    print(f"New unique components vs current gold: {len(new_comps)}")
    if new_comps:
        for c in sorted(new_comps):
            print(f"  + {c}")
    print()
    print(f"Output files:")
    print(f"  {OUTPUT_PARQUET}")
    print(f"  {OUTPUT_CSV}")
    print(f"  {AUDIT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
