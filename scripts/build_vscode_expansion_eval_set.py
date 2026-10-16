"""Build the vscode expansion recording set: 63 real-label issues in eval_set.jsonl row format.

Input : reports/vscode_gold_expansion_2026-10-12/sample.jsonl (real repo component labels; produced by
        scripts/vscode_gold_expansion.py, candidates are disjoint from every train set, the gold set and the eval set)
Output: eval/expansions/vscode_2026-10-12/eval_set.jsonl + provenance.json  (NOT eval/eval_set.jsonl)

Row format and rules mirror eval/eval_set.jsonl: gold_priority is inferred from resolution speed exactly as
scripts/10_curate_triage_gold.py infer_priority does when a repo has no priority label (<1 d high, <7 d medium,
else low); similar_issues are the frozen top-5 from the served vscode index, retrieved on CPU float32 with the
issue itself excluded (eval/freeze_similar_issues.py does the same for the committed set).

Run from a checkout that holds data/models (the index is gitignored):  python scripts/build_vscode_expansion_eval_set.py
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU float32, as the committed freeze step: the prompt must be hardware independent

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE = ROOT / "reports" / "vscode_gold_expansion_2026-10-12" / "sample.jsonl"
OUT_DIR = ROOT / "eval" / "expansions" / "vscode_2026-10-12"
INDEX_DIR = ROOT / "data" / "models" / "similar_issue_index_microsoft_vscode_bge"
K = 5  # same top-k as eval/freeze_similar_issues.py


def infer_priority(resolution_days: float) -> str:
    # Same thresholds as scripts/10_curate_triage_gold.py::infer_priority (resolution_hours < 24 / < 7*24).
    if resolution_days < 1:
        return "high"
    if resolution_days < 7:
        return "medium"
    return "low"


def main() -> None:
    from triage_iq.models.similar_issues import SimilarIssueRetriever

    det = SimilarIssueRetriever.load(str(INDEX_DIR))
    rows = []
    for ln in SAMPLE.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        s = json.loads(ln)
        assert s["label_source"] == "repo", s["number"]  # real labels only (D70)
        num = int(s["number"])
        days = float(s["resolution_days"])
        rows.append(
            {
                "id": f"vscode-{num}",
                "repo": "microsoft/vscode",
                "number": num,
                "title": s["title"],
                "body": s["body"],
                "gold_component": s["repo_label_component"],
                "gold_priority": infer_priority(days),
                "actual_resolution_days": days,
                "created_at": str(s["created_at"]).replace(" ", "T"),
                "similar_issues": det.retrieve(s["title"] + ". " + s["body"], k=K, exclude_number=num),
            }
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "eval_set.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    prov = {
        "n": len(rows),
        "source": str(SAMPLE.relative_to(ROOT)).replace("\\", "/"),
        "index_faiss_sha256": hashlib.sha256((INDEX_DIR / "index.faiss").read_bytes()).hexdigest(),
        "k": K,
        "frozen_on": "cpu-float32",
        "priority_rule": "resolution_days <1 high, <7 medium, else low (10_curate_triage_gold.py infer_priority)",
        "label_source": "real repo component label for all rows; no LLM-derived labels",
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
