"""Zero-token check: does the judge's component_match agree with exact string equality?

Reads the committed eval cassette (no network, no model). For each recorded judge call it takes the gold
component and the plan's predicted_component out of the judge's own prompt, and the component_match score out
of the judge's recorded response, and compares them. Writes reports/component_match_agreement_<date>.json.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASSETTE = ROOT / "eval" / "cassettes" / "eval_cassette.json"
OUT = ROOT / "reports" / "component_match_agreement_2026-10-16.json"


def _json_after(text: str, marker: str) -> dict:
    start = text.index(marker) + len(marker)
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(text[start:].lstrip())
    return obj


def main() -> None:
    entries = json.loads(CASSETTE.read_text(encoding="utf-8"))["entries"]
    rows = []
    for key, e in entries.items():
        user = e["request_messages"][-1]["content"]
        if "GOLD STANDARD:" not in user:
            continue
        gold = re.search(r"GOLD STANDARD:\s*\n\s*component:\s*(.+)", user).group(1).strip()
        title = re.search(r"Title:\s*(.+)", user).group(1).strip()
        plan = _json_after(user, "TRIAGE PLAN TO EVALUATE:")
        content = e["response"]["content"]
        resp = json.loads(content[content.index("{") : content.rindex("}") + 1])  # strips any stray prose
        pred = plan.get("predicted_component")
        rows.append(
            {
                "key": key[:12],
                "title": title[:90],
                "gold": gold,
                "predicted": pred,
                "component_source": plan.get("component_source"),
                "exact_equal": pred == gold,
                "component_match": resp["component_match"],
            }
        )
    n = len(rows)
    # Judge score for an exactly equal pair should be 2 (correct); for unequal 0 or 1 (wrong / plausible).
    agree = [r for r in rows if r["exact_equal"] == (r["component_match"] == 2)]
    dis = [r for r in rows if r not in agree]
    table = Counter((r["exact_equal"], r["component_match"]) for r in rows)
    result = {
        "n_judge_entries": n,
        "agreement_rule": "exact_equal iff component_match == 2",
        "agree": len(agree),
        "agreement_rate": round(len(agree) / n, 4) if n else None,
        "equal_pairs": sum(r["exact_equal"] for r in rows),
        "contingency": {f"exact_equal={k[0]}, component_match={k[1]}": v for k, v in sorted(table.items())},
        "disagreements": dis,
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements"}, indent=1))
    for r in dis:
        print(r["gold"], "|", r["predicted"], "| cm", r["component_match"], "|", r["title"])


if __name__ == "__main__":
    main()
