from __future__ import annotations

"""Does the LLM's resolution prose agree with the conformal interval the API returns?

Zero LLM calls. Reads the synthesis entries of a recorded cassette: each one holds the prompt
the model saw (point estimate and "80% prediction interval") and the plan it wrote. The served
conformal interval is rebuilt exactly as api/app.py attached it before the prompt carried it:
[max(0, lo - Q), hi + Q] with Q from the committed CQR store.

Per repo it reports
  * overlap contradictions (the eval's existing check) against the raw and the conformal interval;
  * how many plans quote a number that equals one of the prompt's interval bounds or its point
    (a "quote"), and how many of those quotes disagree with the corresponding conformal bound
    beyond display rounding (0.05 d or 2 pct). That is the mismatch a reader of the page sees
    between the prose and the interval fields.

Usage: python scripts/prose_vs_conformal_interval.py --cassette eval/cassettes/eval_cassette.json \
    --cqr data/models/cqr_conformal_adjustments_v2.json --out reports/prose_vs_conformal_interval.json
"""

import argparse
import json
import re
from pathlib import Path

from triage_iq.api.loader import _load_conformal_adjustments
from triage_iq.models.resolution_consistency import UNIT_TO_DAYS, verify_resolution_consistency

_PROMPT_RE = re.compile(
    r"Repository: (?P<repo>\S+).*?Point estimate: (?P<pt>[\d.]+) days\s*\n"
    r"80% prediction interval(?: \(coverage-calibrated\))?: \[(?P<lo>[\d.]+)d, (?P<hi>[\d.]+)d\]",
    re.S,
)
_QUOTE_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s*({'|'.join(sorted(UNIT_TO_DAYS, key=len, reverse=True))})\b", re.I
)
_PROSE_FIELDS = ("expected_resolution_summary", "triage_summary")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.05, 0.02 * max(abs(a), abs(b)))


def analyse(cassette: dict, adjustments: dict, keys: set[str] | None = None) -> dict:
    per: dict[str, dict] = {}
    examples: dict[str, list] = {}
    for key, entry in cassette["entries"].items():
        if keys is not None and key not in keys:
            continue
        msgs = entry.get("request_messages") or []
        # few-shot examples are earlier user turns; the real request is the last one
        user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        m = _PROMPT_RE.search(user)
        if not m:
            continue
        try:
            plan = json.loads(entry["response"]["content"])
        except (KeyError, ValueError, TypeError):
            continue
        if "expected_resolution_summary" not in plan:
            continue
        repo = m["repo"]
        q = adjustments[repo]["q_adjustment_hours"] / 24.0
        lo, hi, pt = float(m["lo"]), float(m["hi"]), float(m["pt"])
        c_lo, c_hi = max(0.0, lo - q), hi + q
        s = per.setdefault(repo, dict(
            n=0, with_time_claim=0, contradicts_raw=0, contradicts_conformal=0,
            plans_quoting_bound=0, plans_with_bound_mismatch=0, bounds_identical_at_1dp=0,
        ))
        s["n"] += 1
        s["bounds_identical_at_1dp"] += (round(lo, 1), round(hi, 1)) == (round(c_lo, 1), round(c_hi, 1))
        text = " ".join(str(plan.get(f, "")) for f in _PROSE_FIELDS)
        rep_raw = verify_resolution_consistency(plan["expected_resolution_summary"], lo, hi)
        rep_con = verify_resolution_consistency(plan["expected_resolution_summary"], c_lo, c_hi)
        s["with_time_claim"] += rep_raw.has_time_claim
        s["contradicts_raw"] += rep_raw.contradicts
        s["contradicts_conformal"] += rep_con.contradicts
        quoted = [float(v) * UNIT_TO_DAYS[u.lower()] for v, u in _QUOTE_RE.findall(text)]
        # a quote "belongs" to a prompt number when it equals it at display precision
        pairs = [(lo, c_lo, "lower"), (hi, c_hi, "upper")]
        hit = [(name, raw, con, v) for v in quoted for raw, con, name in pairs if _close(v, raw)]
        if hit:
            s["plans_quoting_bound"] += 1
            bad = [h for h in hit if not _close(h[3], h[2])]
            if bad:
                s["plans_with_bound_mismatch"] += 1
                examples.setdefault(repo, []).append({
                    "quoted_days": round(bad[0][3], 3), "bound": bad[0][0],
                    "prompt_days": round(bad[0][1], 3), "conformal_days": round(bad[0][2], 3),
                    "summary": plan["expected_resolution_summary"][:220],
                })
    return {"per_repo": per, "examples": {k: v[:5] for k, v in examples.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cassette", required=True, type=Path)
    ap.add_argument("--cqr", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, help="restrict to the synthesis keys it records")
    a = ap.parse_args()
    adj = _load_conformal_adjustments(a.cqr.parent)
    keys = None
    if a.checkpoint:
        done = json.loads(a.checkpoint.read_text(encoding="utf-8"))["done"]
        keys = {r["synthesis_cache_key"] for r in done.values() if r.get("synthesis_cache_key")}
    res = analyse(json.loads(a.cassette.read_text(encoding="utf-8")), adj, keys)
    res["provenance"] = {
        "script": "scripts/prose_vs_conformal_interval.py",
        "cassette": a.cassette.name,
        "restricted_to_checkpoint": a.checkpoint.name if a.checkpoint else None,
        "Q_hours": {r: adj[r]["q_adjustment_hours"] for r in adj},
        "llm_calls": 0,
    }
    a.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
