"""D1: merge the two hand-judged review shards (k8s + vscode) into one review file.

Reads:  reports/d1_pair_quality_review_k8s.json
        reports/d1_pair_quality_review_vscode.json
Writes: reports/d1_pair_quality_review.json
"""

from __future__ import annotations

import json
from pathlib import Path

K8S = Path("reports/d1_pair_quality_review_k8s.json")
VSCODE = Path("reports/d1_pair_quality_review_vscode.json")
OUT = Path("reports/d1_pair_quality_review.json")
SAMPLE = Path("reports/d1_pair_sample_for_review.json")


def main() -> None:
    k8s = json.loads(K8S.read_text(encoding="utf-8"))
    vscode = json.loads(VSCODE.read_text(encoding="utf-8"))
    merged = k8s + vscode

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    expected_keys = {(r["repo"], r["query_number"], r["original_number"]) for r in sample}
    got_keys = {(r["repo"], r["query_number"], r["original_number"]) for r in merged}
    missing = expected_keys - got_keys
    extra = got_keys - expected_keys
    unjudged = [r for r in merged if r.get("verdict") not in ("genuine", "incidental")]

    print(f"sample: {len(sample)}  k8s: {len(k8s)}  vscode: {len(vscode)}  merged: {len(merged)}")
    if missing:
        print(f"MISSING {len(missing)} keys from merged output: {sorted(missing)[:10]}")
    if extra:
        print(f"EXTRA {len(extra)} keys not in original sample: {sorted(extra)[:10]}")
    if unjudged:
        print(f"UNJUDGED {len(unjudged)} rows (verdict not genuine/incidental)")
        for r in unjudged[:5]:
            print("  ", r["repo"], r["query_number"], r["original_number"], r.get("verdict"))

    if not missing and not unjudged:
        OUT.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"OK -- wrote {len(merged)} judged rows to {OUT}")
    else:
        print("NOT writing output -- fix gaps above first")


if __name__ == "__main__":
    main()
