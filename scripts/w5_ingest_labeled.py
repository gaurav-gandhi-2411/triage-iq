"""W5: Ingest GG-labeled candidate CSV into the canonical gold set.

Reads the labeled output of data/gold_expansion_candidates_labeled.csv,
validates accepted rows, merges them into the existing gold set, and
writes the expanded gold to data/gold_triage_plans.parquet.

Usage:
    python scripts/w5_ingest_labeled.py --labeled data/gold_expansion_candidates_labeled.csv
    python scripts/w5_ingest_labeled.py --labeled ... --write          # actually write
    python scripts/w5_ingest_labeled.py --labeled ... --write --output data/gold_v2.parquet

Default is dry-run (prints what would happen, writes nothing).
The --write flag must be passed explicitly to modify the canonical gold file.

Labeled CSV contract (GG fills exactly these columns):
  label_decision       REQUIRED  "accept" | "reject"
  label_rejection_code REQUIRED when reject  one of the REJECTION_CODES
  corrected_component  optional  override component if the model label was wrong
  labeler_notes        optional  free text

All other columns are read-only context; do not modify them.
"""
from __future__ import annotations

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

CANONICAL_GOLD = ROOT / "data" / "gold_triage_plans.parquet"

VALID_REPOS = {"microsoft/vscode", "kubernetes/kubernetes"}
VALID_DECISIONS = {"accept", "reject"}
VALID_RES_BUCKETS = {"hours", "days", "weeks", "months", "long"}
REJECTION_CODES = {
    "empty-body",
    "bot",
    "non-english",
    "mislabeled",
    "trivial",
    "duplicate-theme",
    "wontfix",
    "other",
}

# W5 stratification targets: 9 per bucket per repo
BUCKET_TARGETS: dict[str, dict[str, int]] = {
    "microsoft/vscode": {b: 9 for b in VALID_RES_BUCKETS},
    "kubernetes/kubernetes": {b: 9 for b in VALID_RES_BUCKETS},
}


def infer_priority(row: pd.Series) -> str:
    """Mirror the priority-inference logic in scripts/10_curate_triage_gold.py."""
    raw = row.get("priority")
    if pd.notna(raw) and raw:
        p = str(raw).lower()
        if any(k in p for k in ("critical", "p0", "urgent", "blocker", "important-soon")):
            return "high"
        if any(k in p for k in ("p1", "important-soon", "important", "high")):
            return "high"
        if any(k in p for k in ("p2", "medium", "normal")):
            return "medium"
        # "backlog", "awaiting-more-evidence", etc. fall through to resolution speed
    hrs = row.get("resolution_hours", np.nan)
    if pd.isna(hrs):
        return "medium"
    if float(hrs) < 24:
        return "high"
    if float(hrs) < 7 * 24:
        return "medium"
    return "low"


def validate_labeled_csv(df: pd.DataFrame) -> None:
    """Raise ValueError if the required labeling columns are missing."""
    if "label_decision" not in df.columns:
        raise ValueError(
            "Missing required column 'label_decision'. "
            "Add it to the CSV and fill with 'accept' or 'reject' for every row."
        )
    bad_decisions = df[~df["label_decision"].isin(VALID_DECISIONS)]["label_decision"].unique()
    if len(bad_decisions):
        raise ValueError(
            f"Invalid label_decision values: {bad_decisions}. "
            f"Must be one of: {VALID_DECISIONS}."
        )


def _is_null(val) -> bool:
    """True for None, NaN, empty string, or the string 'nan'/'none'."""
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    return str(val).strip().lower() in ("", "nan", "none")


def validate_accepted_rows(df: pd.DataFrame, existing_gold: pd.DataFrame) -> list[str]:
    """Return list of validation error messages (one per bad row)."""
    errors: list[str] = []
    existing_keys = set(zip(existing_gold["repo"], existing_gold["number"]))

    for i, row in df.iterrows():
        prefix = f"Row {i} (#{row.get('number')}, {row.get('repo')})"

        for field in ("number", "repo", "title", "body_clean", "resolution_hours"):
            val = row.get(field)
            if _is_null(val):
                errors.append(f"{prefix}: missing required field '{field}'")

        if row.get("repo") not in VALID_REPOS:
            errors.append(f"{prefix}: repo '{row.get('repo')}' not in {VALID_REPOS}")

        # corrected_component overrides component; either must be non-null
        comp = row.get("corrected_component")
        if _is_null(comp):
            comp = row.get("component")
        if _is_null(comp):
            errors.append(f"{prefix}: component is null — fill 'corrected_component' if needed")

        try:
            rh = float(row.get("resolution_hours", 0))
            if rh <= 0:
                errors.append(f"{prefix}: resolution_hours must be > 0, got {rh}")
        except (ValueError, TypeError):
            errors.append(f"{prefix}: resolution_hours is not numeric")

        key = (row.get("repo"), row.get("number"))
        if key in existing_keys:
            errors.append(f"{prefix}: already in existing gold set — duplicate")

    return errors


def build_gold_rows(accepted: pd.DataFrame) -> pd.DataFrame:
    """Transform accepted candidate rows into the canonical gold schema."""
    rows = []
    for _, row in accepted.iterrows():
        comp = row.get("corrected_component")
        if _is_null(comp):
            comp = row["component"]
        rows.append({
            "repo": row["repo"],
            "number": int(row["number"]),
            "title": str(row["title"]),
            "body_clean": str(row["body_clean"]),
            "gold_component": str(comp),
            "type": str(row.get("type", "")) if pd.notna(row.get("type")) else "",
            "gold_priority": infer_priority(row),
            "actual_resolution_days": float(row["resolution_hours"]) / 24.0,
            "created_at": row["created_at"],
        })
    return pd.DataFrame(rows)


def print_composition_report(combined: pd.DataFrame, label: str = "Combined gold") -> None:
    """Print per-repo, per-component, per-bucket counts vs W5 targets."""
    print(f"\n{'='*60}")
    print(f"{label}  (n={len(combined)})")
    print(f"{'='*60}")

    for repo in VALID_REPOS:
        sub = combined[combined["repo"] == repo].copy()
        if sub.empty:
            continue
        alias = "vsc" if "vscode" in repo else "k8s"
        print(f"\n--- {repo} (n={len(sub)}) ---")

        sub["res_bucket"] = sub["actual_resolution_days"].apply(_days_to_bucket)
        bucket_counts = sub["res_bucket"].value_counts()
        targets = BUCKET_TARGETS[repo]
        print("  Resolution buckets (count vs target=9 new each):")
        for b in sorted(VALID_RES_BUCKETS):
            cnt = int(bucket_counts.get(b, 0))
            t = targets.get(b, 9)
            flag = "" if cnt >= t else f"  <- under target ({t})"
            print(f"    {b:8s}: {cnt:3d}{flag}")

        comp_counts = sub["gold_component"].value_counts()
        print(f"  Unique components: {sub['gold_component'].nunique()}")
        print("  Top 10 components:")
        for comp, cnt in comp_counts.head(10).items():
            print(f"    {comp:30s}: {cnt}")

        prio_counts = sub["gold_priority"].value_counts()
        print(f"  Priority: {dict(prio_counts)}")

    print()


def _days_to_bucket(days: float) -> str:
    if days * 24 < 24:
        return "hours"
    if days < 7:
        return "days"
    if days < 30:
        return "weeks"
    if days < 180:
        return "months"
    return "long"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest labeled W5 candidates into gold set")
    parser.add_argument("--labeled", required=True, help="Path to labeled CSV")
    parser.add_argument("--write", action="store_true",
                        help="Actually write the merged gold set (default: dry-run only)")
    parser.add_argument("--output", default=str(CANONICAL_GOLD),
                        help=f"Output parquet path (default: {CANONICAL_GOLD})")
    parser.add_argument("--gold", default=str(CANONICAL_GOLD),
                        help=f"Existing gold parquet to extend (default: {CANONICAL_GOLD})")
    args = parser.parse_args()

    # --- Load inputs ---
    labeled_path = Path(args.labeled)
    if not labeled_path.exists():
        logger.error("Labeled CSV not found: %s", labeled_path)
        sys.exit(1)

    df_labeled = pd.read_csv(labeled_path)
    logger.info("Loaded %d rows from %s", len(df_labeled), labeled_path)

    gold_path = Path(args.gold)
    if not gold_path.exists():
        logger.error("Existing gold set not found: %s", gold_path)
        sys.exit(1)
    existing_gold = pd.read_parquet(gold_path)
    logger.info("Existing gold set: %d issues", len(existing_gold))

    # --- Validate column contract ---
    try:
        validate_labeled_csv(df_labeled)
    except ValueError as e:
        logger.error("Schema error in labeled CSV: %s", e)
        sys.exit(1)

    # --- Split accepts / rejects ---
    accepted = df_labeled[df_labeled["label_decision"] == "accept"].copy()
    rejected = df_labeled[df_labeled["label_decision"] == "reject"].copy()
    pending = df_labeled[~df_labeled["label_decision"].isin({"accept", "reject"})]

    print(f"\n=== Label decision summary ===")
    print(f"  accept:  {len(accepted)}")
    print(f"  reject:  {len(rejected)}")
    if len(pending):
        print(f"  pending: {len(pending)}  <- these will NOT be ingested")

    # Rejection code breakdown
    if len(rejected):
        print("\n  Rejection codes:")
        code_col = "label_rejection_code" if "label_rejection_code" in rejected.columns else None
        if code_col:
            codes = rejected[code_col].fillna("unspecified").value_counts()
            for code, cnt in codes.items():
                print(f"    {code:30s}: {cnt}")
        else:
            print("    (label_rejection_code column absent — codes unknown)")

    if len(accepted) == 0:
        logger.warning("No accepted rows — nothing to ingest.")
        sys.exit(0)

    # --- Validate accepted rows ---
    errors = validate_accepted_rows(accepted, existing_gold)
    if errors:
        print("\n=== VALIDATION ERRORS (must fix before --write) ===")
        for e in errors:
            print(f"  ERROR: {e}")
        logger.error("%d validation error(s). Fix before re-running with --write.", len(errors))
        sys.exit(1)

    # --- Build gold-schema rows ---
    new_gold_rows = build_gold_rows(accepted)
    logger.info("Built %d new gold rows", len(new_gold_rows))

    # --- Merge ---
    combined = pd.concat([existing_gold, new_gold_rows], ignore_index=True)
    logger.info("Merged: %d existing + %d new = %d total", len(existing_gold), len(new_gold_rows), len(combined))

    # --- Composition report ---
    print_composition_report(combined, label=f"Merged gold set (dry-run={not args.write})")

    # --- Write (gated on --write) ---
    if args.write:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, index=False)
        logger.info("Wrote merged gold set (%d issues) -> %s", len(combined), out_path)
        print(f"\n[WRITTEN] {out_path}  ({len(combined)} issues)")
    else:
        print(f"\n[DRY-RUN] Not written. Pass --write to update {args.output}")


if __name__ == "__main__":
    main()
