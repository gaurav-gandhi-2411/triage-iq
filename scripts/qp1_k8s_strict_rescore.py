"""Secondary STRICT re-scoring of the SAVED k8s Phase 1 test probabilities (tfidf_lr_v1 / c1).

Not a new test evaluation: no refit, no new model id; reads <data-root>/models/<slug>/tfidf_lr/test.parquet.
Strict convention (as vscode, ADR-0070): gold `other` is always a miss and a predicted `other` never
earns a hit. Lenient (published headline): gold other + predicted other = hit; recomputed here from
the same probabilities and asserted equal to the published report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_eval as ev  # noqa: E402
import qp1_tfidf_lr as t1  # noqa: E402

OTHER = "other"
SLUG = "kubernetes__kubernetes"
ROOT = Path(r"D:\triageiq-data")
REPORTS = Path(__file__).resolve().parents[1] / "reports"
STATEMENT = (
    "Secondary re-scoring of SAVED Phase 1 test probabilities of tfidf_lr_v1 (c1) under the strict "
    "convention; not a new test evaluation, no refit, no new model id, published headline unchanged."
)


def hits(y: np.ndarray, p: np.ndarray, other: int) -> dict[str, np.ndarray]:
    """Per-issue booleans: lenient/strict top-1/top-3 and gold-in-family (gold != other)."""
    order = ev.ranked(p)[:, :3]
    top1, top3 = order[:, 0] == y, (order == y[:, None]).any(axis=1)
    fam = y != other
    # strict: gold other is a miss; a predicted other can only match a gold other, so it never hits
    return {
        "lenient_top1": top1,
        "lenient_top3": top3,
        "strict_top1": fam & top1,
        "strict_top3": fam & top3,
        "gold_in_family": fam,
    }


def _rate(a: np.ndarray, n_boot: int) -> dict[str, Any]:
    out = ev.bootstrap(lambda i: float(a[i].mean()), len(a), n_boot)
    out["wilson95"] = list(ev.wilson(int(a.sum()), len(a)))
    out["k"] = int(a.sum())
    return out


def rescore(y: np.ndarray, p: np.ndarray, other: int, n_boot: int = ev.N_BOOT) -> dict[str, Any]:
    """All secondary numbers for gold indices `y` and probabilities `p`."""
    h = hits(y, p, other)
    fam = h["gold_in_family"]
    out: dict[str, Any] = {"n": len(y), "n_gold_other": int((~fam).sum())}
    for k in ("strict_top1", "strict_top3", "lenient_top1", "lenient_top3"):
        out[k] = _rate(h[k], n_boot)
    out["gold_other_share"] = {"rate": float((~fam).mean()), "k": int((~fam).sum()), "n": len(y)}
    out["strict_coverage_bound"] = float(fam.mean())  # strict top-3 cannot exceed this
    out["n_in_family"] = int(fam.sum())
    for k in ("top1", "top3"):  # in-family accuracy (strict hits over gold-in-family issues)
        out[f"in_family_{k}"] = _rate(h[f"strict_{k}"][fam], n_boot)
    return out


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S607


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=REPORTS / "qp1_k8s_strict_rescore_2026-10.json")
    a = ap.parse_args(argv)
    published = json.loads((REPORTS / f"qp1_tfidf_lr_{SLUG}_2026-10.json").read_text("utf-8"))
    pub = published["models"][t1.MODEL_ID]["test"]
    classes = published["classes"]
    pfile = a.root / "models" / SLUG / "tfidf_lr" / "test.parquet"
    pr = pd.read_parquet(pfile)
    te = t1.load_split(a.root / "derived" / SLUG, a.root / "splits" / SLUG, ("test",))
    te = te[te["classifier_eligible"]].reset_index(drop=True)
    assert te["number"].tolist() == pr["number"].tolist(), "issue order differs from saved probs"
    y_lab, _ = t1.gold(te, classes[:-1])
    y = ev.to_idx(y_lab, classes)
    res = rescore(y, pr[classes].to_numpy(), classes.index(OTHER))
    assert res["n"] == pub["n"] == 1384, (res["n"], pub["n"])
    for mine, key in (("lenient_top1", "top1"), ("lenient_top3", "top3")):
        assert res[mine]["point"] == pub[key]["point"], (
            mine,
            res[mine]["point"],
            pub[key]["point"],
        )
        assert res[mine]["ci95"] == pub[key]["ci95"], (mine, res[mine]["ci95"], pub[key]["ci95"])
    assert res["lenient_top1"]["wilson95"] == pub["top1_wilson95"]
    report = {
        "statement": STATEMENT,
        "git_sha": _git("rev-parse", "HEAD"),
        "inputs": {
            "probabilities": str(pfile),
            "probabilities_sha256": ev.file_sha256(pfile),
            "published_report": f"reports/qp1_tfidf_lr_{SLUG}_2026-10.json",
            "model_id": t1.MODEL_ID,
            "bootstrap": {"n_boot": ev.N_BOOT, "seed": ev.SEED},
        },
        "lenient_matches_published_exactly": True,  # asserted above
        "results": res,
    }
    a.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in res if k != "n"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
