"""Gold v2 re-score of the kubernetes sig/* TF-IDF+LR model (quality push, 2026-10).

NOT a new test evaluation of a new model: the model, its classes and its SAVED validation / test
probabilities are untouched; only the gold changes. Old gold = the primary label computed by the
pre-amendment `per_issue` (git ref passed as --old-gold-ref) on the OLD derived tables (embedded,
incomplete timelines). New gold ("v2") = the current `qp0_ceiling_labels.per_issue` (a label with no
event counts as present from creation) on tables derived from the complete DIRECT timelines. Both
are scored with the same functions as the original report (qp1_eval.evaluate, same bootstrap).
Populations: classifier_eligible issues with >= 1 sig/* label under that gold, restricted to
issues that have saved probabilities; newly labelled issues (no probabilities) are only counted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp0_ceiling_labels as cl  # noqa: E402
import qp1_eval as ev  # noqa: E402

SPLITS = ("train", "validation", "test")
METRICS = ("top1", "top3", "anyvalid_top1", "anyvalid_top3", "macro_f1", "nll", "ece")
PerIssue = Callable[[pd.DataFrame, pd.DataFrame, Callable[[str], bool]], pd.DataFrame]


def sig(label: str) -> bool:
    return label.startswith("sig/")


def load_old_per_issue(script: Path) -> PerIssue:
    """`per_issue` from a saved copy of the pre-amendment qp0_ceiling_labels.py."""
    spec = importlib.util.spec_from_file_location("qp0_ceiling_labels_old", script)
    mod: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.per_issue


def git_show(ref: str, path: str, dest: Path) -> Path:
    """Write `git show <ref>:<path>` to dest (scratch copy of a historical script)."""
    out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, check=True)  # noqa: S603, S607
    dest.write_bytes(out.stdout)
    return dest


def gold_table(derived: Path, splits_dir: Path, per_issue: PerIssue) -> pd.DataFrame:
    """Labelled issues (>= 1 sig/* final label) of every split with primary / final sets."""
    sp = pd.concat(
        [
            pd.read_parquet(
                splits_dir / f"{s}.parquet", columns=["number", "split", "classifier_eligible"]
            )
            for s in SPLITS
        ]
    )
    issues = pd.read_parquet(derived / "issues.parquet")
    issues = (
        issues[issues["number"].isin(sp["number"])].sort_values("number").reset_index(drop=True)
    )
    events = pd.read_parquet(derived / "label_events.parquet")
    d = per_issue(issues, events[events["number"].isin(issues["number"])], sig)
    d = d.merge(sp, on="number")
    return d[d["final"].map(bool).astype(bool)].reset_index(drop=True)


def to_class(label: str, modelled: set[str]) -> str:
    return label if label in modelled else "other"


def compare_gold(old: pd.DataFrame, new: pd.DataFrame, modelled: set[str]) -> dict[str, Any]:
    """Per split, eligible issues only: how the labelled population, label sets and primaries moved."""
    out: dict[str, Any] = {}
    for s in SPLITS:
        o, n = (
            t[t["split"].eq(s) & t["classifier_eligible"]].set_index("number") for t in (old, new)
        )
        both = o.index.intersection(n.index)
        prim_raw = (o.loc[both, "primary"] != n.loc[both, "primary"]).sum()
        prim_cls = sum(
            to_class(o.at[i, "primary"], modelled) != to_class(n.at[i, "primary"], modelled)
            for i in both
        )
        set_chg = (o.loc[both, "final"] != n.loc[both, "final"]).sum()
        out[s] = {
            "n_old_labelled": len(o), "n_new_labelled": len(n), "n_both": len(both),
            "newly_labelled": len(n.index.difference(o.index)),
            "unlabelled": len(o.index.difference(n.index)),
            "label_set_changed": int(set_chg), "primary_changed": int(prim_raw),
            "primary_changed_modelled_class": int(prim_cls),
            "share_primary_changed": float(prim_raw / len(both)) if len(both) else None,
        }  # fmt: skip
    return out


def class_counts(t: pd.DataFrame, modelled: set[str]) -> dict[str, dict[str, int]]:
    """Primary-class counts per split (eligible only), modelled classes + other."""
    return {
        s: dict(
            Counter(
                to_class(p, modelled)
                for p in t[t["split"].eq(s) & t["classifier_eligible"]]["primary"]
            )
        )
        for s in SPLITS
    }


def score(
    t: pd.DataFrame, probs: pd.DataFrame, classes: list[str], split: str, numbers: set[int] | None
) -> dict[str, Any]:
    """qp1_eval.evaluate (bootstrap CIs) of saved probabilities against gold ``t``."""
    modelled = set(classes[:-1])
    d = t[t["split"].eq(split) & t["classifier_eligible"] & t["number"].isin(probs["number"])]
    if numbers is not None:
        d = d[d["number"].isin(numbers)]
    d = d.sort_values("number").reset_index(drop=True)
    p = probs.set_index("number").loc[d["number"], classes].to_numpy()
    y = [to_class(x, modelled) for x in d["primary"]]
    sets = [frozenset(to_class(x, modelled) for x in s) for s in d["final"]]
    res = ev.evaluate(split, True, y, sets, p, classes)  # final=True: re-score of the tested model
    return {k: res[k] for k in (*METRICS, "n")}


def rescore(
    old: pd.DataFrame, new: pd.DataFrame, models_dir: Path, grid: dict[str, Any]
) -> dict[str, Any]:
    classes, out = grid["classes"], {}
    for s in ("validation", "test"):
        probs = pd.read_parquet(models_dir / "tfidf_lr" / f"{s}.parquet")
        both = set(old[old["split"].eq(s) & old["classifier_eligible"]]["number"]) & set(
            new[new["split"].eq(s) & new["classifier_eligible"]]["number"]
        )
        out[s] = {
            "old_gold": score(old, probs, classes, s, None),
            "gold_v2": score(new, probs, classes, s, None),
            "common_issues_old_gold": score(old, probs, classes, s, both),
            "common_issues_gold_v2": score(new, probs, classes, s, both),
            "n_saved_probability_rows": len(probs),
            "n_new_labelled_without_probabilities": int(
                len(new[new["split"].eq(s) & new["classifier_eligible"] & ~new["number"].isin(probs["number"])])
            ),
        }  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gold v2 re-score of the saved k8s TF-IDF+LR model")
    ap.add_argument("--derived-old", type=Path, required=True)
    ap.add_argument("--derived-v2", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--old-gold-ref", required=True, help="git ref with the pre-amendment script")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        old_script = git_show(
            a.old_gold_ref, "scripts/qp0_ceiling_labels.py", Path(tmp) / "old_cl.py"
        )
        old_pi = load_old_per_issue(old_script)
    old = gold_table(a.derived_old, a.splits_dir, old_pi)
    new = gold_table(a.derived_v2, a.splits_dir, cl.per_issue)
    grid = json.loads((a.models_dir / "tfidf_lr" / "grid.json").read_text(encoding="utf-8"))
    modelled = set(grid["classes"][:-1])
    carried = Counter(x for s in new[new["split"].eq("train")]["final"] for x in s)
    v2_classes = sorted(k for k, v in carried.items() if v >= 30)
    report = {
        "title": "gold v2 re-score",
        "note": "re-scoring of the already-tested tfidf_lr_v1 model (saved probabilities, no "
        "retraining) against gold recomputed from complete direct timelines; not a new test "
        "evaluation of a new model",
        "old_gold_ref": a.old_gold_ref, "derived_old": str(a.derived_old), "derived_v2": str(a.derived_v2),
        "classes": grid["classes"],
        "v2_train_classes_ge30": v2_classes, "class_definition_would_change": v2_classes != sorted(modelled),
        "gold_comparison": compare_gold(old, new, modelled),
        "class_counts_primary": {"old_gold": class_counts(old, modelled), "gold_v2": class_counts(new, modelled)},
        "metrics": rescore(old, new, a.models_dir, grid),
        "seed": ev.SEED, "n_boot": ev.N_BOOT,
    }  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(report["gold_comparison"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
