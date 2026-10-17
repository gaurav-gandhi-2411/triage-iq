"""Majority-class baseline for the 47-class component classifier, on the same fresh test rows as the paired report.

For each repo: the label of every test row is `component` (as scripts/measure_classifier_old_vs_new_paired.py reads it);
the baseline predicts the most frequent TRAIN label for top-1, and the three most frequent TRAIN labels for top-3.
Writes reports/classifier_majority_baseline.json (Wilson 95% intervals, same formula as the other reports) and checks
that the regenerated test split has the same size as the paired report's test rows.

The classifier split parquets are not committed. They are regenerated deterministically from data/raw with
scripts/02_preprocess.py + scripts/03_split.py into a scratch directory (seed 42, `stratified_classifier_split`);
pass --processed-dir to reuse an already regenerated one.

Usage: python scripts/classifier_majority_baseline.py [--processed-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRED = ROOT / "reports" / "classifier_old_vs_new_paired_2026-09-23.json"
OUT = ROOT / "reports" / "classifier_majority_baseline.json"
REPOS = ("microsoft_vscode", "kubernetes_kubernetes")


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def regenerate(out_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "02_preprocess.py"), "--out-dir", str(out_dir)],
        check=True,
        cwd=ROOT,
    )  # noqa: S603
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "03_split.py"), "--processed-dir", str(out_dir)],
        check=True,
        cwd=ROOT,
    )  # noqa: S603


def build(processed: Path) -> dict:
    paired = json.loads(PAIRED.read_text(encoding="utf-8"))["per_repo"]
    out: dict = {}
    for repo in REPOS:
        train = pd.read_parquet(processed / f"{repo}_classifier_train.parquet")
        test = pd.read_parquet(processed / f"{repo}_classifier_test.parquet")
        n = len(test)
        assert n == paired[repo]["test_rows"], (repo, n, paired[repo]["test_rows"])
        top = train["component"].value_counts().index.tolist()
        y = test["component"]
        k1 = int((y == top[0]).sum())
        k3 = int(y.isin(top[:3]).sum())
        out[repo] = {
            "test_rows": n,
            "n_classes_train": len(top),
            "majority_label": top[0],
            "top3_labels": top[:3],
            "top1": round(k1 / n, 4),
            "top1_ci95": wilson(k1, n),
            "top3": round(k3 / n, 4),
            "top3_ci95": wilson(k3, n),
        }
    return {
        "provenance": "scripts/classifier_majority_baseline.py: most frequent TRAIN label(s) predicted for every row of the regenerated fresh test split; test sizes match reports/classifier_old_vs_new_paired_2026-09-23.json",
        "per_repo": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir")
    a = ap.parse_args()
    if a.processed_dir:
        rep = build(Path(a.processed_dir))
    else:
        with tempfile.TemporaryDirectory() as td:
            regenerate(Path(td))
            rep = build(Path(td))
    OUT.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(rep["per_repo"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
