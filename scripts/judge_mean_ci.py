"""Bootstrap 95% interval for the pooled judge mean (/15) on the committed eval cassette.

Reads the recorded judge responses (no network, no model), sums the six rubric dimensions per issue exactly as
`TriageScore.total()` does, checks the mean against reports/eval_baseline.json, and writes
reports/judge_mean_ci.json: n, mean, percentile bootstrap 95% CI (2000 resamples, seed 42 as elsewhere in the repo).
The interval is over the 64 issues; it does not include judge noise or synthesis sampling noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASSETTE = ROOT / "eval" / "cassettes" / "eval_cassette.json"
BASELINE = ROOT / "reports" / "eval_baseline.json"
OUT = ROOT / "reports" / "judge_mean_ci.json"
DIMENSIONS = (
    "component_match",
    "similar_issues_relevance",
    "resolution_estimate_reasonableness",
    "priority_alignment",
    "next_steps_actionability",
    "overall_quality",
)
N_RESAMPLES = 2000
SEED = 42


def totals() -> list[int]:
    entries = json.loads(CASSETTE.read_text(encoding="utf-8"))["entries"]
    out = []
    for e in entries.values():
        if "GOLD STANDARD:" not in e["request_messages"][-1]["content"]:
            continue
        content = e["response"]["content"]
        resp = json.loads(content[content.index("{") : content.rindex("}") + 1])
        out.append(sum(int(resp[d]) for d in DIMENSIONS))
    return out


def build() -> dict:
    t = np.array(totals(), dtype=float)
    base = json.loads(BASELINE.read_text(encoding="utf-8"))["overall"]
    assert len(t) == base["n"], (len(t), base["n"])
    assert abs(t.mean() - base["mean"]) < 1e-3, (t.mean(), base["mean"])
    rng = np.random.default_rng(SEED)
    means = np.array([rng.choice(t, size=len(t), replace=True).mean() for _ in range(N_RESAMPLES)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "provenance": "scripts/judge_mean_ci.py over eval/cassettes/eval_cassette.json (judge qwen3:8b), checked against reports/eval_baseline.json overall",
        "n": len(t),
        "mean": round(float(t.mean()), 4),
        "ci95": [round(float(lo), 2), round(float(hi), 2)],
        "method": "percentile bootstrap over issues",
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "scale_max": 15,
    }


def main() -> int:
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(OUT.relative_to(ROOT), json.loads(OUT.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
