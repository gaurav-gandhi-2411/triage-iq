"""One-off investigation script (2026-08-11 session, Track 2 follow-up).

Joins the blind pair-validity labels (produced by 5 independent reviewers, each blind to
retrieval outcome -- see reports/track2_k8s_pair_labels.json and the investigation doc for the
rubric) against the already-computed hit/miss results (reports/track2_k8s_miss_analysis.json),
and computes:
  - the clean-subset (VALID-only) R@5 with a bootstrap CI
  - the excluded-vs-valid hit-rate split, as a direct check against selection bias: exclusion
    labels were assigned with zero visibility into hit/miss, so a large gap here reflects the
    exclusion criteria's validity/difficulty correlation, not outcome-driven cherry-picking

Reads:  reports/track2_k8s_150_blind.json
        reports/track2_k8s_pair_labels.json
        reports/track2_k8s_miss_analysis.json
Writes: reports/track2_k8s_clean_eval.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 42
N_BOOTSTRAP = 2000
REPORTS = Path("reports")


def main() -> None:
    blind = json.loads((REPORTS / "track2_k8s_150_blind.json").read_text(encoding="utf-8"))
    labels = json.loads((REPORTS / "track2_k8s_pair_labels.json").read_text(encoding="utf-8"))
    results = json.loads((REPORTS / "track2_k8s_miss_analysis.json").read_text(encoding="utf-8"))[
        "results"
    ]

    pair_ids = sorted(entry["pair_id"] for entry in labels)
    assert pair_ids == list(range(150)), "expected all 150 pair_ids exactly once"

    label_by_id = {entry["pair_id"]: entry for entry in labels}
    hit_by_pair = {(r["query_number"], r["target_number"]): r["hit_at_5"] for r in results}

    merged = []
    for b in blind:
        lab = label_by_id[b["pair_id"]]
        hit = hit_by_pair.get((b["query_number"], b["target_number"]))
        assert hit is not None, f"missing hit/miss data for pair {b['pair_id']}"
        merged.append(
            {
                "pair_id": b["pair_id"],
                "query_number": b["query_number"],
                "target_number": b["target_number"],
                "label": lab["label"],
                "reason": lab["reason"],
                "hit_at_5": hit,
            }
        )

    n_total = len(merged)
    valid = [m for m in merged if m["label"] == "VALID"]
    excluded = [m for m in merged if m["label"] != "VALID"]
    overall_hits = sum(1 for m in merged if m["hit_at_5"])
    val_hits = sum(1 for m in valid if m["hit_at_5"])
    exc_hits = sum(1 for m in excluded if m["hit_at_5"])

    hits_arr = np.array([1.0 if m["hit_at_5"] else 0.0 for m in valid])
    rng = np.random.default_rng(SEED)
    n = len(hits_arr)
    boot = np.array([hits_arr[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    clean_r5 = float(hits_arr.mean())

    label_counts = dict(Counter(m["label"] for m in merged))
    by_reason = {}
    for reason in ["EXCLUDE_UMBRELLA", "EXCLUDE_CAUSAL_ONLY", "EXCLUDE_OTHER"]:
        sub = [m for m in merged if m["label"] == reason]
        h = sum(1 for m in sub if m["hit_at_5"])
        by_reason[reason] = {"n": len(sub), "hits": h, "hit_rate": h / len(sub) if sub else None}

    out = {
        "n_total": n_total,
        "n_valid": len(valid),
        "n_excluded": len(excluded),
        "label_counts": label_counts,
        "overall_unfiltered_r5": overall_hits / n_total,
        "excluded_hit_rate": exc_hits / len(excluded),
        "valid_hit_rate": val_hits / len(valid),
        "by_exclusion_reason": by_reason,
        "clean_r5": clean_r5,
        "clean_r5_ci95": [float(lo), float(hi)],
        "pairs": merged,
    }
    (REPORTS / "track2_k8s_clean_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Overall (unfiltered): {overall_hits}/{n_total} = {overall_hits / n_total:.4f}")
    print(f"Excluded pairs: {exc_hits}/{len(excluded)} = {exc_hits / len(excluded):.4f} hit rate")
    print(f"Valid pairs:    {val_hits}/{len(valid)} = {val_hits / len(valid):.4f} hit rate")
    print(f"CLEAN SUBSET: n={n}, R@5={clean_r5:.4f} [{lo:.4f},{hi:.4f}] ({val_hits}/{n} hits)")
    print("Wrote reports/track2_k8s_clean_eval.json")


if __name__ == "__main__":
    main()
