"""Project the vscode retrain input tables to the release-asset columns (ADR-0072).

The scheduled retrain (`.github/workflows/vscode-retrain.yml`) reads `issues.parquet` and
`label_events.parquet` from the `vscode-retrain-data` GitHub release. Those files hold public issue
fields only: no author, user login, assignee, actor, email or any other account field. Every refresh of
the release MUST be produced by this script from the derived_v2 scrape, never by uploading the scrape
files, so that the column lists below stay the single definition of what is published.

Usage: python scripts/vscode_retrain_release_assets.py --src <derived_v2/microsoft__vscode> --out <dir>
Upload the two files in <dir> (plus champion.joblib / champion.manifest_key) with `gh release`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Exactly the columns `scripts/vscode_retrain.py all` reads; changing a list is a privacy-relevant edit.
SPEC: dict[str, list[str]] = {
    "issues.parquet": [
        "number",
        "created_at",
        "title",
        "body",
        "labels_final",
        "labels_t72h",
        "text_hash",
    ],
    "label_events.parquet": ["number", "ts", "action", "label"],
}


def project_table(table: pa.Table, columns: list[str], name: str) -> pa.Table:
    """`table` reduced to exactly `columns`, in that order, without schema metadata.

    Raises ValueError if a required column is missing (a renamed upstream column must stop the
    refresh, not publish a file the retrain cannot read). Schema metadata is dropped because pandas
    metadata embeds the original column names and index information, including dropped columns.
    """
    missing = [c for c in columns if c not in table.column_names]
    if missing:
        raise ValueError(f"{name}: required columns missing from the source: {missing}")
    return table.select(columns).replace_schema_metadata(None)


def project_dir(src: Path, out: Path) -> dict[str, int]:
    """Write the projected release files for every entry of SPEC; returns {file: rows}."""
    out.mkdir(parents=True, exist_ok=True)
    rows: dict[str, int] = {}
    for name, columns in SPEC.items():
        table = project_table(pq.read_table(src / name), columns, name)
        pq.write_table(table, out / name, compression="zstd")
        rows[name] = table.num_rows
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--src", type=Path, required=True, help="derived_v2/microsoft__vscode directory"
    )
    ap.add_argument("--out", type=Path, required=True, help="output directory (created)")
    args = ap.parse_args()
    for name, n in project_dir(args.src, args.out).items():
        print(name, n, SPEC[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
