from __future__ import annotations
"""Score every (issue, arm) row that has plan content saved, using the local qwen3:8b
judge (ADR-0019, zero cost). Reads one or more result JSONL files, skips rows without a
'plan' key (the old diagnostic-only rows), scores the rest, and prints per-arm summary
stats plus judge-score-per-completion-token."""
import json
import sys

sys.path.insert(0, "src")

from triage_iq.evaluation.triage_eval import TriageJudge  # noqa: E402

judge = TriageJudge(provider="ollama", model="qwen3:8b", ollama_seed=42)

files = sys.argv[1:]
rows = []
for fpath in files:
    for line in open(fpath, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "plan" in r:
            rows.append(r)

print(f"Scoring {len(rows)} rows with plan content...", flush=True)

results = []
for i, r in enumerate(rows):
    plan_json = json.dumps(r["plan"], ensure_ascii=False)
    try:
        score = judge.score(r["issue_title"], r["issue_body"], plan_json, r["gold"])
        total = (
            score.component_match + score.similar_issues_relevance
            + score.resolution_estimate_reasonableness + score.priority_alignment
            + score.next_steps_actionability + score.overall_quality
        )
        comp_tokens = r["usage"]["completion_tokens"]
        results.append({
            "arm": r["arm"], "number": r["number"], "total": total,
            "completion_tokens": comp_tokens, "per_1k_tokens": round(total / comp_tokens * 1000, 3),
        })
        print(f"[{i+1}/{len(rows)}] {r['arm']} #{r['number']}: total={total}/15 comp_tokens={comp_tokens}", flush=True)
    except Exception as exc:
        print(f"[{i+1}/{len(rows)}] {r['arm']} #{r['number']}: JUDGE ERROR {type(exc).__name__}: {exc}", flush=True)

print("\n=== SUMMARY ===")
by_arm: dict[str, list[dict]] = {}
for res in results:
    by_arm.setdefault(res["arm"], []).append(res)
for arm, items in sorted(by_arm.items()):
    totals = [it["total"] for it in items]
    per_tok = [it["per_1k_tokens"] for it in items]
    mean = sum(totals) / len(totals)
    mean_per_tok = sum(per_tok) / len(per_tok)
    print(f"Arm {arm}: n={len(items)} judge_mean={mean:.2f}/15 mean_per_1k_completion_tokens={mean_per_tok:.3f}")

print("\n=== RAW JSON ===")
print(json.dumps(results, indent=2))
