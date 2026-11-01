"""Assemble reports/vscode_index_v2_prereg_results_2026-11.json from the per-step outputs.

Pre-registration: docs/QP_VSCODE_INDEX_V2_PREREG_2026-11.md. Takes the raw outputs of
vscode_v2_refreeze_set_copy.py, check_prompt_neighbour_leakage.py, vscode_index_v2_prereg_scoring.py and
vscode_index_v2_serving_budget.py and evaluates each gated criterion mechanically (no judgement): the
pass/fail booleans below are the prereg thresholds applied to the measured numbers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PEAK_RSS_LIMIT_GIB, P50_LIMIT_MS = 3.4, 300.0


def load(p: Path) -> dict[str, Any]:
    """Read one JSON file."""
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scratch", type=Path, required=True, help="dir holding refreeze/leak_v2/scoring/serving_*/load_* json")
    ap.add_argument("--leakage-exit-code", type=int, required=True, help="exit code of check_prompt_neighbour_leakage.py")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    s = a.scratch
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()  # noqa: E731,S603,S607
    refreeze, leak, score = load(s / "refreeze.json"), load(s / "leak_v2.json"), load(s / "scoring.json")
    v2, served = load(s / "serving_v2.json"), load(s / "serving_served.json")
    c2 = score["criterion_2_answerable_share"]["paired_bootstrap_v2_minus_served"]
    c3 = score["criterion_3_r5_over_all_queries"]["paired_bootstrap_v2_minus_served"]
    gates = {
        "1_leakage": a.leakage_exit_code == 0 and leak["n_violations"] == 0 and leak["n_unknown"] == 0
        and not leak["prompt_vs_signals_mismatches"] and not leak["issues_without_prompt"],
        "2_answerable_share_ci_lower_gt_0": c2["ci95"][0] > 0,
        "3_r5_all_ci_lower_gt_0": c3["ci95"][0] > 0,
        "4a_peak_rss_le_3.4_gib": v2["mem_after_20_triage"]["peak_rss_gib"] <= PEAK_RSS_LIMIT_GIB,
        "4b_cloud_run_utilisation_lt_85pct": "PENDING (measured later on the tagged candidate revision)",
        "5_vscode_retrieval_p50_le_300_ms": v2["latency_serving_detector"]["p50_ms"] <= P50_LIMIT_MS,
    }
    out = {
        "prereg": "docs/QP_VSCODE_INDEX_V2_PREREG_2026-11.md",
        "scripts_commit": git("rev-parse", "HEAD"),
        "gates": gates,
        "all_measurable_gates_pass": all(v is True for v in gates.values() if isinstance(v, bool)),
        "criterion_1_leakage_3b": {
            "exit_code": a.leakage_exit_code,
            "n_violations": leak["n_violations"],
            "n_unknown": leak["n_unknown"],
            "per_repo_per_type_counts": leak["per_repo"],
            "evidence_items_removed_by_as_of_filter": leak["evidence_items_removed_by_as_of_filter"],
            "date_source_counts": leak["date_source_counts"],
            "date_source_disagreements": leak["date_source_disagreements"],
            "prompt_vs_signals_mismatches": leak["prompt_vs_signals_mismatches"],
            "refreeze": {k: v for k, v in refreeze.items() if k != "rows"},
        },
        "scoring_3c": score,
        "serving_budget": {"v2": v2, "served": served,
                           "index_cold_load": {"v2": load(s / "load_v2.json"), "served": load(s / "load_served.json")}},
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"gates": gates, "all_measurable_gates_pass": out["all_measurable_gates_pass"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
