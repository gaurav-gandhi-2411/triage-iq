"""Per-repo results for the microsoft/vscode expansion (63 issues) pooled with the committed 11.

Reads recorded judge requests/responses only (no network, no model): each judge entry carries the gold
block, the plan that was judged (predicted_component, grounding_status) and the six rubric scores, so
every arm is computed by the same code path. Issues are matched to their repo through the eval sets by
title. Writes reports/vscode_expansion_results_2026-10.json.

Arms: committed11 (eval/cassettes, repo microsoft/vscode), expansion63 (reports/vscode_expansion_2026-10-12),
pooled74 (union). Intervals are percentile bootstrap over issues (2000 resamples, seed 42) and do not
include judge noise or synthesis sampling noise. Nothing here changes a gate threshold or baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMITTED_SET = ROOT / "eval" / "eval_set.jsonl"
COMMITTED_CASSETTE = ROOT / "eval" / "cassettes" / "eval_cassette.json"
EXPANSION_SET = ROOT / "eval" / "expansions" / "vscode_2026-10-12" / "eval_set.jsonl"
EXPANSION_DIR = ROOT / "reports" / "vscode_expansion_2026-10-12"
EXPANSION_CASSETTE = EXPANSION_DIR / "eval_cassette.json"
EXPANSION_CHECKPOINT = EXPANSION_DIR / "recording_checkpoint.json"
BASELINE = ROOT / "reports" / "eval_baseline.json"
OUT = ROOT / "reports" / "vscode_expansion_results_2026-10.json"
VSCODE = "microsoft/vscode"
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
# Measured per-issue judge jitter of the committed baseline (reports/eval_baseline.json threshold block).
JITTER_STD = 0.748


def _rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _norm(title: str) -> str:
    return " ".join(title.strip().strip('"').split())


def _judged(cassette: Path, rows: list[dict]) -> list[dict]:
    """One record per judge entry whose issue title belongs to a row of `rows`."""
    by_title = {_norm(r["title"]): r for r in rows}
    entries = json.loads(cassette.read_text(encoding="utf-8"))["entries"]
    out = []
    for e in entries.values():
        req = e["request_messages"][-1]["content"]
        if "GOLD STANDARD:" not in req:
            continue
        title_line = req.split("Title:", 1)[1].split("\n", 1)[0]
        row = by_title.get(_norm(title_line))
        if row is None:
            continue
        content = e["response"]["content"]
        score = json.loads(content[content.index("{") : content.rindex("}") + 1])
        plan, _ = json.JSONDecoder().raw_decode(req.split("TRIAGE PLAN TO EVALUATE:\n", 1)[1])
        dims = {d: int(score[d]) for d in DIMENSIONS}
        out.append(
            {
                "id": row["id"],
                "dims": dims,
                "total": sum(dims.values()),
                "gold": str(row["gold_component"]).strip().lower(),
                "pred": str(plan["predicted_component"]).strip().lower(),
                "grounded": bool(plan["grounding_status"]["all_grounded"]),
            }
        )
    return out


def _ci(totals: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    means = np.array(
        [rng.choice(totals, size=len(totals), replace=True).mean() for _ in range(N_RESAMPLES)]
    )
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(lo), 2), round(float(hi), 2)]


def _summary(recs: list[dict]) -> dict:
    t = np.array([r["total"] for r in recs], dtype=float)
    n = len(recs)
    floor = [
        r["dims"]["component_match"] == 0 or r["dims"]["similar_issues_relevance"] == 0
        for r in recs
    ]
    return {
        "n": n,
        "mean": round(float(t.mean()), 4),
        "ci95": _ci(t),
        "std_per_issue": round(float(t.std(ddof=1)), 3),
        "dimensions": {
            d: round(float(np.mean([r["dims"][d] for r in recs])), 4) for d in DIMENSIONS
        },
        "floor_fail_rate": round(float(np.mean(floor)), 4),
        "fabrication_rate": round(float(np.mean([not r["grounded"] for r in recs])), 4),
        "component_exact_match": round(float(np.mean([r["gold"] == r["pred"] for r in recs])), 4),
        "component_exact_match_count": int(sum(r["gold"] == r["pred"] for r in recs)),
        "judge_component_match_full_marks": round(
            float(np.mean([r["dims"]["component_match"] == 2 for r in recs])), 4
        ),
        "band_if_gated_at_this_n": round(2 * JITTER_STD / n**0.5, 3),
    }


def build() -> dict:
    committed = [r for r in _rows(COMMITTED_SET) if r["repo"] == VSCODE]
    expansion = _rows(EXPANSION_SET)
    a = _judged(COMMITTED_CASSETTE, committed)
    b = _judged(EXPANSION_CASSETTE, expansion)
    base = json.loads(BASELINE.read_text(encoding="utf-8"))["per_repo"][VSCODE]
    assert len(a) == base["n"] == 11, (len(a), base["n"])
    assert abs(np.mean([r["total"] for r in a]) - base["mean"]) < 1e-3
    assert len(b) == len(expansion) == 63, (len(b), len(expansion))
    cp = json.loads(EXPANSION_CHECKPOINT.read_text(encoding="utf-8"))["done"]
    assert len(cp) == 63 and all(r.get("judge_score") for r in cp.values())
    assert not ({r["id"] for r in a} & {r["id"] for r in b})
    ta = np.array([r["total"] for r in a], dtype=float)
    tb = np.array([r["total"] for r in b], dtype=float)
    rng = np.random.default_rng(SEED)
    diffs = np.array(
        [
            rng.choice(tb, len(tb)).mean() - rng.choice(ta, len(ta)).mean()
            for _ in range(N_RESAMPLES)
        ]
    )
    pooled = _summary(a + b)
    return {
        "provenance": (
            "scripts/vscode_expansion_results.py over eval/cassettes/eval_cassette.json (committed 11) and "
            "reports/vscode_expansion_2026-10-12/eval_cassette.json (63 recorded 2026-10-16..19, synthesis "
            "openai/gpt-oss-120b, judge qwen3:8b); committed-11 mean checked against reports/eval_baseline.json"
        ),
        "committed11": _summary(a),
        "expansion63": _summary(b),
        "pooled74": pooled,
        "expansion63_minus_committed11": {
            "mean_diff": round(float(tb.mean() - ta.mean()), 4),
            "ci95": [round(float(x), 2) for x in np.percentile(diffs, [2.5, 97.5])],
            "method": "percentile bootstrap of the difference of arm means",
        },
        "baseline_vscode_mean_n11": base["mean"],
        "baseline_vscode_band_n11": 0.45,
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "scale_max": 15,
        "caveat": "Intervals are over issues only (no judge or synthesis sampling noise). Component exact match "
        "compares the judged plan's predicted_component with the real repo component label.",
    }


def main() -> int:
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(OUT.relative_to(ROOT))
    print(OUT.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
