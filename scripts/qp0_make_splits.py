"""Temporal train / validation / test split of issues with computed (never silent) leakage flags.

PRE-REGISTERED DESIGN (spec section 4.3, 2026-10-20; do not change after seeing results)
 1. Pure temporal split on created_at (UTC): train < 2025-07-01T00:00:00Z (i.e. <= 2025-06-30
    23:59:59), validation 2025-07-01 .. 2025-12-31, test >= 2026-01-01. Issues only (PRs are
    not split). Every issue gets exactly one split. Nothing is dropped.
 2. Leakage flags for validation/test issues (train issues: False by definition):
    leak_texthash = text_hash equals that of some issue in an EARLIER split (train for
    validation; train or validation for test); leak_dup_pair = the issue is the source or the
    canonical of a duplicate pair whose other end lies in an earlier split (both ends must be
    issues in the table). classifier_eligible = not (leak_texthash or leak_dup_pair): the rule
    "remove near-duplicate leakage across splits" for the CLASSIFIER and RESOLUTION tasks only.
 3. Retrieval keeps cross-split duplicate pairs: dup_task_pairs = pairs whose source (the query)
    is in validation or test and whose canonical was created strictly before the source. A
    canonical outside the issues table (PR, or older than the scrape window) cannot be dated, is
    kept and marked canonical_in_table False. exact_text_dup = equal text_hash. Duplicate
    groups inside one split (same text_hash) are counted, not flagged.
 4. Report = counts, hashes, rules; no issue ids. sha256 of each split's sorted newline-joined
    issue numbers (and of the eligible subsets), of each input file, and the commit of this script.
No label is read or computed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

FROZEN_DATE = "2026-10-20"
DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].split("No label", 1)[0].strip()
VAL_START = pd.Timestamp("2025-07-01T00:00:00Z")
TEST_START = pd.Timestamp("2026-01-01T00:00:00Z")
SPLITS = ("train", "validation", "test")
RANK = {s: i for i, s in enumerate(SPLITS)}
COLS = ["number", "created_at", "split", "closed_at", "state", "classifier_eligible"]
COLS += ["leak_texthash", "leak_dup_pair"]


def assign_split(created_at: pd.Series) -> pd.Series:
    """Split name per created_at; naive timestamps are rejected rather than guessed."""
    if created_at.dt.tz is None:
        raise ValueError("created_at must be timezone-aware (UTC)")
    t = created_at.dt.tz_convert("UTC")
    return pd.Series(
        ["train" if x < VAL_START else "validation" if x < TEST_START else "test" for x in t],
        index=created_at.index,
    )


def make_splits(
    issues: pd.DataFrame, pairs: pd.DataFrame, component: Callable[[str], bool] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(issues with split + leak flags, dup_task_pairs). Pure function of its inputs.

    `component` (optional; repos without one label family, e.g. vscode ADR-0070): predicate for
    "this label names a product component". classifier_eligible then also requires >= 1 such label
    in labels_final and the helper column `has_component_label` is added. None (k8s) changes nothing.
    """
    df = issues.sort_values("number").reset_index(drop=True).copy()
    df["split"] = assign_split(df["created_at"])
    rank = df["split"].map(RANK)
    # earliest split in which each text appears: a later-split issue leaks if it is earlier
    first_rank = df.assign(r=rank).groupby("text_hash")["r"].transform("min")
    df["leak_texthash"] = first_rank < rank
    by_num = df.set_index("number")
    leak_nums: set[int] = set()
    for s, c in zip(pairs["source"], pairs["canonical"], strict=True):
        if s in by_num.index and c in by_num.index:
            rs, rc = RANK[by_num.at[s, "split"]], RANK[by_num.at[c, "split"]]
            if rc < rs:
                leak_nums.add(int(s))
            if rs < rc:
                leak_nums.add(int(c))
    df["leak_dup_pair"] = df["number"].isin(leak_nums)
    df["classifier_eligible"] = ~(df["leak_texthash"] | df["leak_dup_pair"])
    if component is not None:
        df["has_component_label"] = [any(component(x) for x in ls) for ls in df["labels_final"]]
        df["classifier_eligible"] &= df["has_component_label"]
    rows = []
    for s, c in sorted(set(zip(pairs["source"], pairs["canonical"], strict=True))):
        if s not in by_num.index or RANK[by_num.at[s, "split"]] == 0:
            continue
        known = c in by_num.index
        if known and not by_num.at[c, "created_at"] < by_num.at[s, "created_at"]:
            continue
        rows.append(
            {
                "source": int(s), "canonical": int(c),
                "split_of_source": by_num.at[s, "split"],
                "split_of_canonical": by_num.at[c, "split"] if known else None,
                "canonical_in_table": known,
                "exact_text_dup": bool(by_num.at[s, "text_hash"] == by_num.at[c, "text_hash"])
                if known else None,
            }
        )  # fmt: skip
    cols = ["source", "canonical", "split_of_source", "split_of_canonical"]
    out = pd.DataFrame(rows, columns=cols + ["canonical_in_table", "exact_text_dup"])
    # explicit nullable dtypes: a bare None in an object column becomes NaN under pandas 3, so
    # "undatable canonical" would not survive a parquet round trip as a proper missing value
    out = out.astype({"split_of_canonical": "string", "exact_text_dup": "boolean"})
    return df, out


def digest_numbers(numbers: pd.Series) -> str:
    return hashlib.sha256("\n".join(map(str, sorted(map(int, numbers)))).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def script_commit() -> str | None:
    """Last commit that touched this script (stable across later report commits)."""
    here = Path(__file__).resolve()
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "log", "-1", "--format=%H", "--", here.name],  # noqa: S607
            cwd=here.parent, capture_output=True, text=True, check=True,
        )  # fmt: skip
    except (subprocess.CalledProcessError, OSError):
        return None
    return out.stdout.strip() or None


def summarise(df: pd.DataFrame, dup: pd.DataFrame) -> dict[str, Any]:
    """Counts and hashes per split; deterministic, no issue ids."""
    out: dict[str, Any] = {}
    for s in SPLITS:
        d = df[df["split"] == s]
        grp = d.groupby("text_hash").size()
        out[s] = {
            "n": len(d), "by_year_month": d["created_at"].dt.strftime("%Y-%m").value_counts().sort_index().to_dict(),
            "state": d["state"].value_counts().sort_index().to_dict(),
            "leak_texthash": int(d["leak_texthash"].sum()),
            "leak_dup_pair": int(d["leak_dup_pair"].sum()),
            "leak_either": int((d["leak_texthash"] | d["leak_dup_pair"]).sum()),
            "classifier_eligible": int(d["classifier_eligible"].sum()),
            **({"has_component_label": int(d["has_component_label"].sum())}
               if "has_component_label" in d else {}),
            "within_split_text_dup_groups": int((grp > 1).sum()),
            "within_split_issues_in_dup_groups": int(grp[grp > 1].sum()),
            "created_at_min": str(d["created_at"].min()), "created_at_max": str(d["created_at"].max()),
            "sha256_numbers": digest_numbers(d["number"]),
            "sha256_numbers_eligible": digest_numbers(d.loc[d["classifier_eligible"], "number"]),
        }  # fmt: skip
    out["dup_task_pairs"] = {
        "n": len(dup),
        "by_split_of_source": dup["split_of_source"].value_counts().sort_index().to_dict(),
        "canonical_not_in_table": int((~dup["canonical_in_table"]).sum()),
        "exact_text_dup": int((dup["exact_text_dup"] == True).sum()),  # noqa: E712
    }
    return out


ADR_0070_CLASS_LIST_SHA256 = "95129c95809c21574ed34d41a46358149322680a7ccb0e42e90f4b62e7b761ea"


def taxonomy_block(df: pd.DataFrame, tx: Any) -> dict[str, Any]:
    """ADR-0070 provenance: exclusion-spec hash and the TRAIN-only parent class list + its hash."""
    train = df[df["split"] == "train"]
    classes = tx.family(tx.window_counts(train, tx.parent_of))
    sha = hashlib.sha256("\n".join(classes).encode()).hexdigest()
    return {
        "adr": "0070", "script": "scripts/qp2_vscode_taxonomy.py",
        "classifier_eligible_rule": "no text-hash / duplicate-pair leakage AND >= 1 non-excluded "
        "component label in labels_final (an unlabelled issue is outside the evaluated population)",
        "exclusion_spec_sha256": tx.SPEC_SHA256, "n_parent_classes_train": len(classes),
        "class_list_sha256": sha, "adr_0070_class_list_sha256": ADR_0070_CLASS_LIST_SHA256,
        "class_list_matches_adr": sha == ADR_0070_CLASS_LIST_SHA256,
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Temporal splits with leakage flags")
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--derived-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--taxonomy", choices=("none", "vscode"), default="none",
                    help="vscode: classifier_eligible also needs a non-excluded component label (ADR-0070)")  # fmt: skip
    a = ap.parse_args(argv)
    slug = a.repo.replace("/", "__")
    derived = a.derived_dir or Path(r"D:\triageiq-data") / "derived" / slug
    out_dir = a.out_dir or Path(r"D:\triageiq-data") / "splits" / slug
    report = a.report or Path(f"reports/qp0_split_manifest_{slug}_2026-10.json")
    files = {n: derived / f"{n}.parquet" for n in ("issues", "duplicate_pairs")}
    component = None
    if a.taxonomy == "vscode":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import qp2_vscode_taxonomy as tx  # noqa: PLC0415  (only needed for this taxonomy)

        component = tx.not_excluded
    cols = ["number", "created_at", "closed_at", "state", "text_hash"]
    cols += ["labels_final"] if component else []
    df, dup = make_splits(
        pd.read_parquet(files["issues"], columns=cols),
        pd.read_parquet(files["duplicate_pairs"]),
        component,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        df.loc[df["split"] == s, COLS].to_parquet(out_dir / f"{s}.parquet", index=False)
    dup.to_parquet(out_dir / "dup_task_pairs.parquet", index=False)
    man = {
        "repo": a.repo, "frozen_date": FROZEN_DATE, "rules": DESIGN,
        "boundaries": {"validation_start": str(VAL_START), "test_start": str(TEST_START)},
        "generated_with": "scripts/qp0_make_splits.py", "script_commit": script_commit(),
        "input_sha256": {n: sha256_file(p) for n, p in files.items()},
        "n_issues": len(df), **summarise(df, dup),
    }  # fmt: skip
    if a.taxonomy == "vscode":
        man["taxonomy"] = taxonomy_block(df, tx)
    text = json.dumps(man, indent=1, sort_keys=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    (out_dir / "manifest.json").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
