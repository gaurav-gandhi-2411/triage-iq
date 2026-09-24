from __future__ import annotations
"""Phase 3b (ADR-0057 follow-on): measure the token-budget cost of turning
TRIAGE_PROMPT_INCLUDE_ATTRIBUTION on, offline, zero live calls.

Builds the actual messages list _call_llm_verbose would send, both with and without the
attribution flag, for every issue in the eval set, using the SAME estimator the real
per-request token-budget guard uses (_estimate_prompt_tokens). Reports the delta against
the guard's own ceiling (_GROQ_TPM_LIMIT=8000, safety margin=200) so the cost is measured
against the actual constraint, not an abstract token count.

Usage:
    .venv/Scripts/python.exe scripts/measure_attribution_token_cost.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "eval")

ROOT = Path(__file__).parent.parent
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"


def _load_eval_set(path: Path) -> list[dict]:
    issues: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                issues.append(json.loads(line))
    return issues


def main() -> None:
    from triage_iq.models.triage import (
        _estimate_prompt_tokens,
        _GROQ_TPM_LIMIT,
        _PROMPT_SIZE_SAFETY_MARGIN,
        _MIN_VIABLE_COMPLETION_TOKENS,
    )
    from triage_iq.prompts.triage_prompt import (
        SYSTEM_PROMPT_LEGACY,
        SYSTEM_PROMPT_PROSE,
        build_few_shot_examples,
        build_few_shot_examples_legacy,
        build_triage_prompt,
    )

    issues = _load_eval_set(EVAL_SET_PATH)

    off_shots = build_few_shot_examples_legacy()
    on_shots = build_few_shot_examples()

    deltas = []
    ceiling_off_violations = 0
    ceiling_on_violations = 0
    max_tokens_default = 2048  # self.max_tokens default, matches TriageAssistant

    for issue in issues:
        # Minimal signals sufficient for build_triage_prompt's text -- token estimate only
        # depends on message TEXT content, not on live classifier/retrieval calls.
        prompt_text = build_triage_prompt(
            issue_title=str(issue["title"]),
            issue_body=str(issue["body"]),
            classifier_top3=[{"label": "placeholder", "confidence": 0.5}] * 3,
            similar_issues=[{"number": 1, "score": 0.5, "text": "placeholder similar issue text"}] * 5,
            resolution_point_days=7.0,
            resolution_lower_days=1.0,
            resolution_upper_days=30.0,
            repo=issue["repo"],
        )

        messages_off = [{"role": "system", "content": SYSTEM_PROMPT_LEGACY}, *off_shots,
                         {"role": "user", "content": prompt_text}]
        messages_on = [{"role": "system", "content": SYSTEM_PROMPT_PROSE}, *on_shots,
                        {"role": "user", "content": prompt_text}]

        est_off = _estimate_prompt_tokens(messages_off)
        est_on = _estimate_prompt_tokens(messages_on)
        deltas.append(est_on - est_off)

        headroom_off = min(max_tokens_default, _GROQ_TPM_LIMIT - est_off - _PROMPT_SIZE_SAFETY_MARGIN)
        headroom_on = min(max_tokens_default, _GROQ_TPM_LIMIT - est_on - _PROMPT_SIZE_SAFETY_MARGIN)
        if headroom_off < _MIN_VIABLE_COMPLETION_TOKENS:
            ceiling_off_violations += 1
        if headroom_on < _MIN_VIABLE_COMPLETION_TOKENS:
            ceiling_on_violations += 1

    n = len(issues)
    deltas.sort()
    print(f"n={n} eval-set issues")
    print(f"Per-issue prompt-token delta (attribution ON - OFF), estimated via "
          f"_estimate_prompt_tokens (same estimator the live guard uses):")
    print(f"  min={deltas[0]}  p50={deltas[n // 2]}  max={deltas[-1]}  "
          f"mean={sum(deltas) / n:.1f}")
    print(f"\nGuard ceiling (_GROQ_TPM_LIMIT={_GROQ_TPM_LIMIT}, "
          f"margin={_PROMPT_SIZE_SAFETY_MARGIN}, floor={_MIN_VIABLE_COMPLETION_TOKENS}):")
    print(f"  issues that would need input-shrinking, attribution OFF: {ceiling_off_violations}/{n}")
    print(f"  issues that would need input-shrinking, attribution ON:  {ceiling_on_violations}/{n}")

    out = {
        "n": n,
        "delta_min": deltas[0], "delta_p50": deltas[n // 2], "delta_max": deltas[-1],
        "delta_mean": round(sum(deltas) / n, 2),
        "ceiling_violations_off": ceiling_off_violations,
        "ceiling_violations_on": ceiling_on_violations,
    }
    out_path = ROOT / "reports" / "attribution_token_cost.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
