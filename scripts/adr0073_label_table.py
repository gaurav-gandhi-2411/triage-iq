"""Regenerate (or check) the ADR-0073 vscode label decision table.

The counts and the two long label lists (already excluded by ADR-0070, KEEP) are rebuilt from the
scraped vscode data; the hand-made decisions (label, decision, reason) are data in
``adr0073_label_decisions.json`` next to this file and are never derived.

    python scripts/adr0073_label_table.py            # rewrite the committed table
    python scripts/adr0073_label_table.py --check    # exit 1 if the committed table differs
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp2_vscode_taxonomy as tax  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "docs/architecture/adr/0073-vscode-label-decisions.md"
DECISIONS_JSON = Path(__file__).resolve().parent / "adr0073_label_decisions.json"
DATA_DIR = Path(r"D:\triageiq-data\derived_v2\microsoft__vscode")

FROZEN_REASON = "already excluded by the frozen ADR-0070 list (taxonomy.is_excluded)"
KEEP_REASON = (
    "names a feature, UI area, language, extension, platform or subsystem; none of tests 1-5 apply"
)
ORDER = {"EXCLUDE": 0, "BORDERLINE-EXCLUDE": 1, "BORDERLINE-KEEP": 2}

# One row per distinct label: (label, n_final, n_event_issues, decision, new, reason).
Row = tuple[str, int, int, str, str, str]


def load_decisions(path: Path = DECISIONS_JSON) -> dict[str, tuple[str, str]]:
    """Return label -> (decision, reason) from the hand-made decisions file."""
    out: dict[str, tuple[str, str]] = {}
    for grp in json.loads(path.read_text(encoding="utf-8")):
        for lab in grp["labels"]:
            assert lab not in out, f"duplicate decision for {lab!r}"
            out[lab] = (grp["decision"], grp["reason"])
    return out


def label_counts(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Per-label final-state issue count, event-issue count and frozen (ADR-0070) flag."""
    iss = pd.read_parquet(
        data_dir / "issues.parquet", columns=["number", "labels_final", "labels_t72h"]
    )
    ev = pd.read_parquet(data_dir / "label_events.parquet")
    fin = Counter(x for ls in iss["labels_final"] for x in set(ls))
    t72 = Counter(x for ls in iss["labels_t72h"] for x in set(ls))
    evn = ev.groupby("label")["number"].nunique().to_dict()
    allv = sorted(set(fin) | set(t72) | set(evn), key=lambda x: (-fin.get(x, 0), x))
    return pd.DataFrame(
        [
            {
                "label": x,
                "n_final": fin.get(x, 0),
                "n_event_issues": evn.get(x, 0),
                "frozen_excluded": tax.is_excluded(x),
            }
            for x in allv
        ]
    )


def build_rows(counts: pd.DataFrame, decisions: dict[str, tuple[str, str]]) -> list[Row]:
    """Join counts (in the given order) with decisions; undecided non-frozen labels are KEEP."""
    rows: list[Row] = []
    for r in counts.itertuples():
        if bool(r.frozen_excluded):
            dec, reason, new = "EXCLUDE", FROZEN_REASON, "no"
        elif r.label in decisions:
            dec, reason = decisions[r.label]
            new = "yes" if dec.endswith("EXCLUDE") else "no"
        else:
            dec, reason, new = "KEEP", KEEP_REASON, "no"
        assert "|" not in r.label and "`" not in r.label, r.label
        rows.append((r.label, int(r.n_final), int(r.n_event_issues), dec, new, reason))
    missing = set(decisions) - set(counts["label"])
    assert not missing, missing
    frozen = set(counts.loc[counts["frozen_excluded"], "label"])
    assert not set(decisions) & frozen, "decision for a label ADR-0070 already excludes"
    return rows


def render(rows: list[Row]) -> str:
    """Render the decision file text (pure; no I/O)."""
    cnt = Counter(r[3] for r in rows)
    new_ex = [r[0] for r in rows if r[4] == "yes"]
    out: list[str] = []
    w = out.append
    w("# ADR-0073 addendum: decisions for every distinct vscode label\n\n")
    w(
        "Generated 2026-10-29 for GG decision D93 from `D:\\triageiq-data\\derived_v2\\microsoft__vscode`"
        " (`issues.parquet` labels_final and labels_t72h, `label_events.parquet`). The rule and its tests are in"
        " `0073-vscode-retrain-meta-label-exclusions.md`.\n"
    )
    w(
        "Counts are issues over the whole scraped history (not the retrain window): `final` = issues carrying the label at their"
        " final state, `events` = issues with at least one label event for it. No model score, prediction or accuracy was looked at"
        " to make these decisions; the scraped data holds no GitHub label descriptions, so the decisions rest on the label name and counts.\n\n"
    )
    w(
        f"Distinct labels: {len(rows)}. Decisions: "
        + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items()))
        + ".\n"
    )
    w(
        f"NEW exclusions for retraining (EXCLUDE or BORDERLINE-EXCLUDE and not already excluded by ADR-0070): {len(new_ex)}.\n"
    )
    w(
        "The `NEW` column of the first table is the contract: `RETRAIN_META_EXCLUDE` in `scripts/vscode_retrain_lib.py` equals the labels with"
        " NEW = yes (lower-cased); `tests/test_vscode_retrain.py` parses this file to check it, and also that every label in the two lists"
        " below is, or is not, excluded by ADR-0070 as stated.\n\n"
    )
    w("## New exclusions and borderline calls\n\n")
    w("| label | final | events | decision | NEW | reason |\n|---|---:|---:|---|---|---|\n")
    for lab, nf, ne, dec, new, reason in sorted(
        (r for r in rows if r[4] == "yes" or r[3] == "BORDERLINE-KEEP"),
        key=lambda x: (ORDER[x[3]], -x[1], x[0]),
    ):
        w(f"| {lab} | {nf} | {ne} | {dec} | {new} | {reason} |\n")
    for title, sel, why in (
        (
            "Already excluded by ADR-0070 (EXCLUDE, NEW = no)",
            lambda r: r[3] == "EXCLUDE" and r[4] == "no",
            "taxonomy.is_excluded is true; not re-decided here",
        ),
        ("KEEP", lambda r: r[3] == "KEEP", KEEP_REASON),
    ):
        sub = [r for r in rows if sel(r)]
        assert all("; " not in r[0] for r in sub)
        w(
            f"\n## {title}: {len(sub)}\n\nReason for all: {why}."
            " Entries are `label (final-state issue count)`.\n\n"
        )
        w("; ".join(f"{r[0]} ({r[1]})" for r in sub) + "\n")
    return "".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed table differs")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--table", type=Path, default=TABLE)
    args: Any = ap.parse_args(argv)
    text = render(build_rows(label_counts(args.data_dir), load_decisions()))
    # Compare with newlines normalised: the index holds LF, a Windows checkout may hold CRLF.
    current = args.table.read_bytes().decode("utf-8").replace("\r\n", "\n")
    if args.check:
        if text == current:
            print(f"identical: {args.table} ({len(text)} chars)")
            return 0
        diff = difflib.unified_diff(
            current.splitlines(), text.splitlines(), "committed", "regenerated", lineterm="", n=0
        )
        print("\n".join(list(diff)[:60]))
        print(f"DIFFERS: {args.table}")
        return 1
    args.table.write_bytes(text.encode("utf-8"))
    print(f"wrote {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
