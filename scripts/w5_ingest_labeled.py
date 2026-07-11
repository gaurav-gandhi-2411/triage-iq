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

Before any merge, every accepted row is checked for THREE-WAY disjointness from
training data (classifier_train, temporal_train, retrieval-train from w3_split.parquet)
— mirrors scripts/w3_t5_eval.py's assert_eval_disjoint_from_train pattern — PLUS a
BGE-embedding near-duplicate check (cosine >= 0.90) against the union of those three
sets, so a re-filed near-identical bug that survives exact-ID matching also hard-fails
(see docs/investigations/gold-set-leakage.md — gold #14398 vs classifier_train #14399,
cosine 0.907, is exactly the case this closes). A gold issue that leaks into training
data would silently inflate every downstream metric, so this hard-fails (raises,
non-zero exit) rather than warning.

Each accepted row also gets a `related_issue_numbers` ground-truth column, extracted
using ONLY the body_ref pattern strategy from scripts/07_extract_related_pairs.py
(duplicate/dup-of/same-as/closing-as-dup-of #N — high-confidence explicit cross-refs).
The body_related patterns (Closes/Fixes/See #N) and title-similarity are deliberately
excluded (unreliable / circular per ADR-0007) — see docs/eval/gold_labeling_protocol.md.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_GOLD = ROOT / "data" / "gold_triage_plans.parquet"
W3_SPLIT_PATH = ROOT / "data" / "w3_split.parquet"

# Ported verbatim from scripts/07_extract_related_pairs.py's BODY_PATTERNS (the "high
# confidence" explicit cross-reference strategy only). Deliberately excludes that
# script's body_related patterns (See/Closes/Fixes #N — unreliable, ADR-0007) and
# title-similarity (circular w.r.t. the retrieval model under evaluation).
BODY_REF_PATTERNS = [
    r"[Dd]uplicate[sd]?(?: of)? #?(\d+)",
    r"[Dd]up(?:licate)? of #?(\d+)",
    r"[Ss]ame as #(\d+)",
    r"[Cc]losing as dup(?:licate)? of #?(\d+)",
]

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


def load_repo_train_numbers(repo: str) -> tuple[set[int], set[int], set[int]]:
    """Load the three training-data issue-number sets a gold issue must avoid.

    Returns (classifier_train_numbers, temporal_train_numbers, retrieval_train_numbers)
    for the given repo (slash form, e.g. "microsoft/vscode"). retrieval_train_numbers is
    the union of query_number/original_number from data/w3_split.parquet's train split.
    """
    repo_slug = repo.replace("/", "_")

    clf_path = ROOT / "data" / "processed" / f"{repo_slug}_classifier_train.parquet"
    classifier_train = set(
        pd.read_parquet(clf_path, columns=["number"])["number"].astype(int)
    )

    temporal_path = ROOT / "data" / "processed" / f"{repo_slug}_temporal_train.parquet"
    temporal_train = set(
        pd.read_parquet(temporal_path, columns=["number"])["number"].astype(int)
    )

    if W3_SPLIT_PATH.exists():
        split_df = pd.read_parquet(W3_SPLIT_PATH)
        train_df = split_df[(split_df["repo"] == repo_slug) & (split_df["split"] == "train")]
        retrieval_train = set(train_df["query_number"].astype(int)) | set(
            train_df["original_number"].astype(int)
        )
    else:
        logger.warning(
            "w3_split.parquet not found at %s — retrieval-train check will be vacuous", W3_SPLIT_PATH
        )
        retrieval_train = set()

    return classifier_train, temporal_train, retrieval_train


NEAR_DUP_THRESHOLD = 0.90  # matches scripts/verify_gold_train_overlap.py's lower audit threshold


def load_repo_bge_vectors(repo: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Reconstruct BGE vectors + issue numbers from the saved FAISS dup-index.

    Mirrors scripts/verify_gold_train_overlap.py::load_bge_vectors (vectors are
    L2-normalized at build time, so inner product == cosine similarity). Returns None
    (not raises) when the index is absent — the near-dup check degrades to vacuous with
    a warning, same tolerance-of-missing-artifact pattern as retrieval_train below.
    """
    import faiss
    import joblib

    repo_slug = repo.replace("/", "_")
    idx_dir = ROOT / "data" / "models" / f"dup_index_{repo_slug}_bge"
    if not (idx_dir / "index.faiss").exists():
        logger.warning(
            "BGE dup-index not found for %s at %s — near-duplicate check will be "
            "vacuous for this repo (ID-based disjointness checks still apply)",
            repo, idx_dir,
        )
        return None
    meta = joblib.load(str(idx_dir / "meta.pkl"))
    index = faiss.read_index(str(idx_dir / "index.faiss"))
    vecs = index.reconstruct_n(0, index.ntotal)
    numbers = np.asarray(meta["issue_numbers"], dtype=np.int64)
    return vecs, numbers


def check_near_duplicates(
    accepted: pd.DataFrame,
    train_numbers_by_repo: dict[str, tuple[set[int], set[int], set[int]]],
    bge_vectors_by_repo: dict[str, tuple[np.ndarray, np.ndarray] | None] | None = None,
    threshold: float = NEAR_DUP_THRESHOLD,
) -> list[dict]:
    """Embedding-similarity near-dup check across classifier/temporal/retrieval train.

    Closes the ADR-0018 gap: exact-ID disjointness alone missed gold #14398 (k8s), which
    is 0.907 cosine to classifier_train #14399 — the same bug re-filed under a new
    issue number. ``bge_vectors_by_repo`` allows callers (e.g. tests) to inject
    {repo: (vecs, numbers) | None} directly instead of reading the FAISS index from disk.
    Returns a list of violation dicts (empty if none, or if no index is available).
    """
    violations: list[dict] = []
    for repo in accepted["repo"].unique():
        clf_train, temp_train, retr_train = train_numbers_by_repo[repo]
        train_numbers = clf_train | temp_train | retr_train
        if not train_numbers:
            continue

        if bge_vectors_by_repo is not None:
            loaded = bge_vectors_by_repo.get(repo)
        else:
            loaded = load_repo_bge_vectors(repo)
        if loaded is None:
            continue
        vecs, vec_numbers = loaded

        num_to_row = {int(n): i for i, n in enumerate(vec_numbers)}
        train_rows = [num_to_row[n] for n in train_numbers if n in num_to_row]
        if not train_rows:
            continue
        train_vecs = vecs[train_rows]
        train_nums = vec_numbers[train_rows]

        for _, row in accepted[accepted["repo"] == repo].iterrows():
            gnum = int(row["number"])
            grow = num_to_row.get(gnum)
            if grow is None:
                continue  # candidate not embedded yet — cannot near-dup check it
            sims = train_vecs @ vecs[grow]
            sims = np.where(train_nums == gnum, -1.0, sims)  # exclude self-number match
            max_idx = int(sims.argmax())
            max_sim = float(sims[max_idx])
            if max_sim >= threshold:
                violations.append({
                    "repo": repo,
                    "gold_number": gnum,
                    "nearest_train_number": int(train_nums[max_idx]),
                    "cosine": round(max_sim, 4),
                })
    return violations


def assert_gold_disjoint_from_train(
    accepted: pd.DataFrame,
    train_numbers_by_repo: dict[str, tuple[set[int], set[int], set[int]]] | None = None,
    bge_vectors_by_repo: dict[str, tuple[np.ndarray, np.ndarray] | None] | None = None,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
) -> None:
    """Assert accepted (repo, number) pairs are disjoint from all 3 training sources.

    Mirrors scripts/w3_t5_eval.py's assert_eval_disjoint_from_train pattern: frozenset
    intersection per check, raise with count + first-5 examples on any violation. Runs
    BEFORE any merge into the canonical gold set — a gold issue leaking into training
    data would silently inflate every downstream metric (classifier accuracy, resolution
    MAE, or retrieval recall@k).

    Checks (each reported PASS/FAIL to stdout, not just asserted):
      1. Not in data/processed/{repo_slug}_classifier_train.parquet["number"]
      2. Not in data/processed/{repo_slug}_temporal_train.parquet["number"]
      3. Not in the retrieval-train issue-number set from data/w3_split.parquet
      4. Not >= near_dup_threshold BGE cosine similarity to any issue in 1-3 (catches a
         re-filed near-duplicate that survives exact-ID matching — the gold #14398 gap)

    ``train_numbers_by_repo`` allows callers (e.g. tests) to inject
    {repo: (classifier_train, temporal_train, retrieval_train)} directly instead of
    reading from disk; defaults to loading via load_repo_train_numbers per repo.
    ``bge_vectors_by_repo`` similarly allows injecting {repo: (vecs, numbers) | None}
    for check 4; defaults to loading the FAISS dup-index from disk per repo.
    """
    accepted_keys = frozenset(zip(accepted["repo"], accepted["number"].astype(int)))

    check_keys: dict[str, set[tuple[str, int]]] = {
        "classifier_train": set(),
        "temporal_train": set(),
        "retrieval_train": set(),
    }
    resolved_train_numbers: dict[str, tuple[set[int], set[int], set[int]]] = {}
    for repo in accepted["repo"].unique():
        if train_numbers_by_repo is not None:
            clf_train, temp_train, retr_train = train_numbers_by_repo[repo]
        else:
            clf_train, temp_train, retr_train = load_repo_train_numbers(repo)
        resolved_train_numbers[repo] = (clf_train, temp_train, retr_train)
        check_keys["classifier_train"] |= {(repo, n) for n in clf_train}
        check_keys["temporal_train"] |= {(repo, n) for n in temp_train}
        check_keys["retrieval_train"] |= {(repo, n) for n in retr_train}

    print("\n=== Disjointness checks (accepted gold vs training data) ===")
    violations: dict[str, list] = {}
    for name, train_keys in check_keys.items():
        overlap = accepted_keys & frozenset(train_keys)
        status = "PASS" if not overlap else "FAIL"
        print(f"  {name:20s}: {status}  (overlap={len(overlap)})")
        if overlap:
            violations[name] = sorted(overlap)[:5]

    near_dup_violations = check_near_duplicates(
        accepted, resolved_train_numbers, bge_vectors_by_repo, near_dup_threshold
    )
    status = "PASS" if not near_dup_violations else "FAIL"
    print(f"  {'near_duplicate':20s}: {status}  (overlap={len(near_dup_violations)})")
    if near_dup_violations:
        violations["near_duplicate"] = near_dup_violations[:5]

    if violations:
        parts = []
        for name, sample in violations.items():
            if name == "near_duplicate":
                parts.append(f"near_duplicate: {len(sample)} overlap, first 5={sample}")
            else:
                parts.append(
                    f"{name}: {len(check_keys[name] & accepted_keys)} overlap, first 5={sample}"
                )
        raise AssertionError(
            "GOLD/TRAIN LEAK: accepted gold issues found in training data. "
            + " | ".join(parts)
            + ". Gold issues MUST be disjoint (exact-ID and near-duplicate) from "
            "classifier, temporal, and retrieval training data (see spec.md hard rules / "
            "scripts/w3_t5_eval.py assert_eval_disjoint_from_train)."
        )
    logger.info(
        "Disjointness checks PASSED: 0 overlap across all 3 training sources, "
        "0 near-duplicates >= %.2f cosine.", near_dup_threshold
    )


def load_issue_created_at_map(repo: str) -> dict[int, pd.Timestamp]:
    """Load {number: created_at} for a repo's full issues parquet.

    Used to validate related-issue references: the reference must exist in the repo's
    issue corpus, and must predate the candidate (mirrors the "original must predate
    query" rule in scripts/07_extract_related_pairs.py).
    """
    repo_slug = repo.replace("/", "_")
    path = ROOT / "data" / "processed" / f"issues_{repo_slug}.parquet"
    df = pd.read_parquet(path, columns=["number", "created_at"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return dict(zip(df["number"].astype(int), df["created_at"]))


def extract_related_issue_numbers(
    title: str,
    body_clean: str,
    created_at,
    issue_created_at_map: dict[int, pd.Timestamp],
) -> list[int]:
    """Extract ground-truth related-issue numbers from body_ref patterns only.

    Applies BODY_REF_PATTERNS (ported from scripts/07_extract_related_pairs.py's
    BODY_PATTERNS) to ``title + " " + body_clean``, case-insensitive. A match is kept
    only if: (a) the referenced issue number exists in the repo's issues parquet, and
    (b) its created_at predates ``created_at`` (the candidate's creation time). Returns
    a list of ints — an empty list is a valid, non-ambiguous label meaning "no
    documented related issue found" (not "not checked").
    """
    combined = f"{title} {body_clean}"
    created_at = pd.to_datetime(created_at, utc=True)
    found: list[int] = []
    for pat in BODY_REF_PATTERNS:
        for m in re.finditer(pat, combined, re.IGNORECASE):
            ref = int(m.group(1))
            if ref in found:
                continue
            ref_created_at = issue_created_at_map.get(ref)
            if ref_created_at is None:
                continue  # referenced issue must exist in this repo's corpus
            if ref_created_at >= created_at:
                continue  # original must predate the candidate (query)
            found.append(ref)
    return found


def build_gold_rows(
    accepted: pd.DataFrame,
    issue_created_at_maps: dict[str, dict[int, pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """Transform accepted candidate rows into the canonical gold schema.

    ``issue_created_at_maps`` allows callers (e.g. tests) to inject
    {repo: {number: created_at}} directly instead of reading from disk; defaults to
    lazily loading via load_issue_created_at_map per repo encountered.
    """
    rows = []
    issue_maps: dict[str, dict[int, pd.Timestamp]] = (
        dict(issue_created_at_maps) if issue_created_at_maps is not None else {}
    )
    for _, row in accepted.iterrows():
        comp = row.get("corrected_component")
        if _is_null(comp):
            comp = row["component"]

        repo = row["repo"]
        if repo not in issue_maps:
            issue_maps[repo] = load_issue_created_at_map(repo)
        related = extract_related_issue_numbers(
            str(row["title"]), str(row["body_clean"]), row["created_at"], issue_maps[repo]
        )

        rows.append({
            "repo": repo,
            "number": int(row["number"]),
            "title": str(row["title"]),
            "body_clean": str(row["body_clean"]),
            "gold_component": str(comp),
            "type": str(row.get("type", "")) if pd.notna(row.get("type")) else "",
            "gold_priority": infer_priority(row),
            "actual_resolution_days": float(row["resolution_hours"]) / 24.0,
            # CSV-sourced created_at is a plain string (pd.read_csv does not parse
            # dates); must coerce to Timestamp here or the merged frame ends up with
            # a mixed str/Timestamp object column that pyarrow rejects on to_parquet.
            "created_at": pd.to_datetime(row["created_at"], utc=True),
            "related_issue_numbers": related,
            "related_issue_needs_spot_check": len(related) > 0,
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

    # --- Disjointness from training data (hard-fail, BEFORE any merge) ---
    assert_gold_disjoint_from_train(accepted)

    # --- Build gold-schema rows ---
    new_gold_rows = build_gold_rows(accepted)
    logger.info("Built %d new gold rows", len(new_gold_rows))
    n_spot_check = int(new_gold_rows["related_issue_needs_spot_check"].sum())
    logger.info(
        "related_issue_numbers: %d/%d new rows have a body_ref match needing spot-check",
        n_spot_check,
        len(new_gold_rows),
    )

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
