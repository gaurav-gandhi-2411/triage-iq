"""Bootstrap 95% interval for the pooled judge mean (/15) on a recorded eval cassette.

Reads the recorded judge responses (no network, no model), sums the six rubric dimensions per issue exactly as
`TriageScore.total()` does and writes reports/judge_mean_ci.json: n, mean, percentile bootstrap 95% CI
(2000 resamples, seed 42 as elsewhere in the repo). The interval is over the issues; it does not include
judge noise or synthesis sampling noise.

Which judge entries count: those whose parent synthesis key (judge_provenance) belongs to a record in the
recording checkpoint, not every judge entry in the cassette (a cassette keeps superseded entries).

Defaults reproduce the committed 64-issue report byte for byte (checked against reports/eval_baseline.json).
Other sets: --cassette <set>/cassettes/eval_cassette.json [--baseline ...] [--out ...]. --detail adds
per-repo intervals and, when every eval-set row has meta.weight, the weight-weighted mean with its Kish
effective sample size (sum(w)^2 / sum(w^2)); the unweighted headline stays first and unchanged.
"""

from __future__ import annotations

import argparse
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


def _entry_total(entry: dict) -> int:
    content = entry["response"]["content"]
    resp = json.loads(content[content.index("{") : content.rindex("}") + 1])
    return sum(int(resp[d]) for d in DIMENSIONS)


def judged(cassette: Path, checkpoint: Path) -> list[tuple[str, int]]:
    """(issue_id, judge total) in cassette order for every checkpoint record's judge entry.

    Fails loudly if a checkpoint record has no judge entry, or one issue has two records: a mean over
    a partial or ambiguous set is a wrong number, not a smaller one.
    """
    entries = json.loads(cassette.read_text(encoding="utf-8"))["entries"]
    done = json.loads(checkpoint.read_text(encoding="utf-8"))["done"]
    by_synthesis: dict[str, str] = {}
    for rec in done.values():
        key, issue_id = rec["synthesis_cache_key"], rec["issue_id"]
        if issue_id in by_synthesis.values():
            raise SystemExit(f"checkpoint has more than one record for {issue_id}: ambiguous")
        by_synthesis[key] = issue_id
    out: list[tuple[str, int]] = []
    for e in entries.values():
        if "GOLD STANDARD:" not in e["request_messages"][-1]["content"]:
            continue
        parent = (e.get("judge_provenance") or {}).get("parent_synthesis_key")
        if parent in by_synthesis:
            out.append((by_synthesis[parent], _entry_total(e)))
    missing = sorted(set(by_synthesis.values()) - {i for i, _ in out})
    if missing:
        raise SystemExit(
            f"{len(missing)} checkpoint records have no judge entry, e.g. {missing[:5]}"
        )
    return out


def _boot_ci(values: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the (weighted) mean, resampling issues, seed 42."""
    rng = np.random.default_rng(SEED)
    n = len(values)
    if weights is None:
        means = np.array(
            [rng.choice(values, size=n, replace=True).mean() for _ in range(N_RESAMPLES)]
        )
    else:
        idx = [rng.integers(0, n, size=n) for _ in range(N_RESAMPLES)]
        means = np.array([np.average(values[i], weights=weights[i]) for i in idx])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 2), round(float(hi), 2)


def detail(rows: list[tuple[str, int]], eval_set: Path) -> dict:
    """Per-repo bootstrap CIs and (if every row carries meta.weight) the weighted mean with Kish n_eff."""
    meta = {}
    for line in eval_set.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            meta[r["id"]] = (r["repo"], (r.get("meta") or {}).get("weight"))
    ids = [i for i, _ in rows]
    t = np.array([v for _, v in rows], dtype=float)
    repos = np.array([meta[i][0] for i in ids])
    per_repo = {}
    for repo in sorted(set(repos)):
        sel = t[repos == repo]
        lo, hi = _boot_ci(sel)
        per_repo[repo] = {"n": len(sel), "mean": round(float(sel.mean()), 4), "ci95": [lo, hi]}
    out: dict = {"per_repo": per_repo}
    w_list = [meta[i][1] for i in ids]
    if all(w is not None for w in w_list):
        w = np.array(w_list, dtype=float)
        lo, hi = _boot_ci(t, w)
        out["weighted"] = {
            "mean": round(float(np.average(t, weights=w)), 4),
            "ci95": [lo, hi],
            "kish_n_eff": round(float(w.sum() ** 2 / (w**2).sum()), 2),
            "n": len(t),
            "weights": "eval-set meta.weight; bootstrap resamples issues with their weights",
        }
    return out


def build(
    cassette: Path = CASSETTE,
    checkpoint: Path | None = None,
    baseline: Path | None = None,
    eval_set: Path | None = None,
    with_detail: bool = False,
) -> dict:
    checkpoint = checkpoint or cassette.parent / "recording_checkpoint.json"
    if baseline is None and cassette == CASSETTE:  # the committed 64-set is always checked
        baseline = BASELINE
    rows = judged(cassette, checkpoint)
    t = np.array([v for _, v in rows], dtype=float)
    if baseline is not None:
        base = json.loads(baseline.read_text(encoding="utf-8"))["overall"]
        assert len(t) == base["n"], (len(t), base["n"])
        assert abs(t.mean() - base["mean"]) < 1e-3, (t.mean(), base["mean"])
    lo, hi = _boot_ci(t)
    what = "eval/cassettes/eval_cassette.json" if cassette == CASSETTE else cassette.as_posix()
    out = {
        "provenance": f"scripts/judge_mean_ci.py over {what} (judge qwen3:8b)"
        + (", checked against reports/eval_baseline.json overall" if baseline is not None else ""),
        "n": len(t),
        "mean": round(float(t.mean()), 4),
        "ci95": [lo, hi],
        "method": "percentile bootstrap over issues",
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "scale_max": 15,
    }
    if with_detail:
        out.update(detail(rows, eval_set or cassette.parent.parent / "eval_set.jsonl"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cassette", type=Path, default=CASSETTE)
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: next to the cassette")
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="assert n and mean against this baseline (default: reports/eval_baseline.json for "
        "the default cassette, none otherwise)",
    )
    ap.add_argument(
        "--eval-set", type=Path, default=None, help="default: <cassette dir>/../eval_set.jsonl"
    )
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--detail", action="store_true", help="add per-repo CIs and the weighted mean")
    args = ap.parse_args(argv)
    result = build(args.cassette, args.checkpoint, args.baseline, args.eval_set, args.detail)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(args.out, json.loads(args.out.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
