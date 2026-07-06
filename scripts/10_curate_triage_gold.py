"""Curate gold-standard triage evaluation set.

DEPRECATED — DO NOT regenerate gold with this script. Historically its sampling
pool was the union of val+test from TWO INDEPENDENT split schemes (temporal and
stratified-classifier) with no exclusion against either scheme's train split,
which produced the 2026-07 gold contamination (54/60 original rows in training
data — see docs/investigations/gold-set-leakage.md). That cross-scheme gap is
now closed in load_eval_splits (ADR-0018), but the script stays deprecated:
gold admission goes through scripts/w5_ingest_labeled.py (three-way ID
disjointness, hard-fail) and is enforced repo-wide by eval/test_invariants.py's
whole-gold invariants (ID-level + BGE near-dup < 0.90). Kept for provenance of
the original 60 rows.

Selects 30 closed issues per repo (60 total) with known outcomes:
- component label (from normalized label set)
- priority (inferred from label or position)
- actual resolution days

Selection criteria:
- Must be closed (resolution_hours known)
- Must have component + type labels (so gold component is unambiguous)
- Stratified: 10 issues per resolution bucket (<7d, 7–30d, >30d)
- Sampled from the union of val+test of both split schemes (NOT train-disjoint
  across schemes — see deprecation note above)

Output: data/gold_triage_plans.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]
ISSUES_PER_REPO = 30
BUCKETS = [(0, 7), (7, 30), (30, 99999)]
ISSUES_PER_BUCKET = 10
RANDOM_SEED = 42


def load_train_numbers(repo: str, suffix: str) -> set[int]:
    """Load a train split's issue numbers for cross-split disjointness checks.

    ADR-0018: the temporal split and the classifier split are computed
    independently (different logic, different label). Membership in one
    split's held-out portion says nothing about membership in the OTHER
    split's train set — an issue held out by the temporal split can still be
    a training example for the classifier, and vice versa. `load_eval_splits`
    must exclude both train sets, not just the held-out portions of each.
    """
    path = ROOT / "data" / "processed" / f"{repo}_{suffix}.parquet"
    if not path.exists():
        logger.warning("%s not found — skipping %s disjointness filter", path, suffix)
        return set()
    df = pd.read_parquet(path, columns=["number"])
    return set(df["number"].astype(int))


def load_eval_splits(repo: str) -> pd.DataFrame:
    """Load val + test splits, excluding any issue in EITHER split's train set.

    The temporal test split for some repos is entirely in one resolution bucket
    due to distribution shift. Combining val + test gives better stratification
    for the gold evaluation set without leaking training data.

    ADR-0018: temporal_val/temporal_test and classifier_val/classifier_test are
    two independently-computed splits over the same corpus. An issue held out
    by one split can simultaneously be a training example in the OTHER split —
    this was not checked prior to ADR-0018 and caused a component_match
    train-contamination that went undetected from the gold set's original
    curation (2026-04-29) until W5 built the first disjointness guard. The
    exclusion below closes that gap; historical fallout is documented in
    docs/investigations/gold-set-leakage.md.
    """
    parts = []
    for suffix in ["temporal_val", "temporal_test", "classifier_val", "classifier_test"]:
        path = ROOT / "data" / "processed" / f"{repo}_{suffix}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path))
    if not parts:
        raise FileNotFoundError(f"No eval splits found for {repo} in data/processed/")
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["number"]).reset_index(drop=True)

    classifier_train_numbers = load_train_numbers(repo, "classifier_train")
    temporal_train_numbers = load_train_numbers(repo, "temporal_train")
    train_numbers = classifier_train_numbers | temporal_train_numbers

    before = len(df)
    df = df[~df["number"].isin(train_numbers)].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "%s: dropped %d issues held out by one split but present in the OTHER "
            "split's train set (cross-split disjointness, ADR-0018)",
            repo,
            dropped,
        )
    return df


def infer_priority(row: pd.Series) -> str:
    """Infer priority label from available metadata."""
    if pd.notna(row.get("priority")):
        p = str(row["priority"]).lower()
        if any(k in p for k in ["critical", "p0", "urgent", "blocker"]):
            return "high"
        if any(k in p for k in ["p1", "important", "high"]):
            return "high"
        if any(k in p for k in ["p2", "medium", "normal"]):
            return "medium"
        return "low"
    # Fall back to resolution speed as priority signal
    hrs = row.get("resolution_hours", np.nan)
    if pd.isna(hrs):
        return "medium"
    if hrs < 24:
        return "high"
    if hrs < 7 * 24:
        return "medium"
    return "low"


def stratified_sample(df: pd.DataFrame, n_per_bucket: int, seed: int) -> pd.DataFrame:
    """Sample n_per_bucket issues from each resolution bucket."""
    rng = np.random.default_rng(seed)
    parts = []
    days = df["resolution_hours"] / 24.0

    for lo, hi in BUCKETS:
        mask = (days >= lo) & (days < hi) & df["component"].notna()
        bucket = df[mask]
        k = min(n_per_bucket, len(bucket))
        if k == 0:
            logger.warning("Bucket [%d, %d) empty — skipping", lo, hi)
            continue
        idx = rng.choice(len(bucket), size=k, replace=False)
        parts.append(bucket.iloc[idx])

    if not parts:
        return pd.DataFrame()
    sample = pd.concat(parts, ignore_index=True)
    # Trim to ISSUES_PER_REPO if overcollected
    if len(sample) > ISSUES_PER_REPO:
        sample = sample.sample(n=ISSUES_PER_REPO, random_state=seed)
    return sample.reset_index(drop=True)


def curate_repo(repo: str) -> pd.DataFrame:
    logger.info("Curating gold set for %s", repo)
    df = load_eval_splits(repo)

    required = ["number", "title", "body_clean", "component", "resolution_hours"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {repo}: {missing}")

    df = df[df["resolution_hours"].notna() & df["component"].notna()].copy()
    sample = stratified_sample(df, ISSUES_PER_BUCKET, RANDOM_SEED)

    sample["repo"] = repo.replace("_", "/", 1)
    sample["actual_resolution_days"] = sample["resolution_hours"] / 24.0
    sample["gold_priority"] = sample.apply(infer_priority, axis=1)

    keep = [
        "repo", "number", "title", "body_clean",
        "component", "type", "gold_priority",
        "actual_resolution_days", "created_at",
    ]
    keep = [c for c in keep if c in sample.columns]
    gold = sample[keep].rename(columns={"component": "gold_component"})

    logger.info(
        "[%s] Curated %d issues: buckets %s",
        repo,
        len(gold),
        gold["actual_resolution_days"]
        .apply(lambda d: "<7d" if d < 7 else ("7-30d" if d < 30 else ">30d"))
        .value_counts()
        .to_dict(),
    )
    return gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=REPOS)
    parser.add_argument("--n", type=int, default=ISSUES_PER_REPO,
                        help="Issues per repo")
    args = parser.parse_args()

    parts = []
    for repo in args.repos:
        try:
            gold = curate_repo(repo)
            parts.append(gold)
        except Exception as exc:
            logger.error("Failed for %s: %s", repo, exc)

    if not parts:
        logger.error("No gold data collected — exiting")
        sys.exit(1)

    out = pd.concat(parts, ignore_index=True)
    out_path = ROOT / "data" / "gold_triage_plans.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info("Saved %d gold issues to %s", len(out), out_path)

    # Summary
    print("\n=== Gold Standard Summary ===")
    for repo in out["repo"].unique():
        sub = out[out["repo"] == repo]
        print(f"\n{repo}: {len(sub)} issues")
        print("  Priority:", sub["gold_priority"].value_counts().to_dict())
        print("  Resolution bucket:",
              sub["actual_resolution_days"]
              .apply(lambda d: "<7d" if d < 7 else ("7-30d" if d < 30 else ">30d"))
              .value_counts()
              .to_dict())
        print("  Components (top-5):", sub["gold_component"].value_counts().head(5).to_dict())


if __name__ == "__main__":
    main()
