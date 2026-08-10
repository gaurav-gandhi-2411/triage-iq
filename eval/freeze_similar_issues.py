from __future__ import annotations
# MUST be first — forces CPU before any torch/sentence_transformers import
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""One-time script: freeze similar_issues for all eval issues on CPU float32.

Writes frozen top-5 similar_issues into each line of eval_set.jsonl and
records provenance in eval/frozen_retrieval_provenance.json.

Run once locally whenever the FAISS index is rebuilt or the eval set changes.

Usage:
    python eval/freeze_similar_issues.py [--dry-run]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402 (after sys.path patch)

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
PROVENANCE_PATH = ROOT / "eval" / "frozen_retrieval_provenance.json"
MANIFEST_PATH = ROOT / "data" / "models" / "MANIFEST.sha256"

REPO_SLUGS = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}
K = 5


def _index_hash(slug: str) -> str:
    manifest_lines = MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
    needle = f"data/models/similar_issue_index_{slug}_bge/index.faiss"
    for line in manifest_lines:
        if line.strip().endswith(needle):
            return line.split()[0]
    raise KeyError(f"index.faiss for {slug} not found in manifest")


def main(dry_run: bool = False) -> None:
    from triage_iq.models.similar_issues import SimilarIssueRetriever

    print("Device: CPU (CUDA_VISIBLE_DEVICES='')")
    print(f"Eval set: {EVAL_SET}")
    print(f"Dry run: {dry_run}\n")

    # Load indexes
    detectors: dict[str, SimilarIssueRetriever] = {}
    for repo, slug in REPO_SLUGS.items():
        idx_dir = str(ROOT / "data" / "models" / f"similar_issue_index_{slug}_bge")
        print(f"Loading {repo} index from {idx_dir} …")
        detectors[repo] = SimilarIssueRetriever.load(idx_dir)

    # Read eval set
    issues = []
    with open(EVAL_SET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                issues.append(json.loads(line))
    print(f"\nFreezing top-{K} similar_issues for {len(issues)} issues (CPU float32) …\n")

    updated: list[dict] = []
    for issue in issues:
        repo = issue["repo"]
        det = detectors[repo]
        text = issue["title"] + ". " + issue["body"]
        num = int(issue["number"])
        similar = det.retrieve(text, k=K, exclude_number=num)
        issue_out = dict(issue)
        issue_out["similar_issues"] = similar
        updated.append(issue_out)
        nums = [s["number"] for s in similar]
        print(f"  {issue['id']:20s}  top-{K}={nums}")

    if not dry_run:
        with open(EVAL_SET, "w", encoding="utf-8", newline="\n") as f:
            for iss in updated:
                f.write(json.dumps(iss, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(updated)} issues to {EVAL_SET}")

        # Write provenance
        provenance = {
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "k": K,
            "frozen_at": "2026-08-06",
            "frozen_on": "cpu-float32",
            "index_hashes": {
                slug: _index_hash(slug)
                for slug in REPO_SLUGS.values()
            },
        }
        PROVENANCE_PATH.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"Wrote provenance to {PROVENANCE_PATH}")
        print(json.dumps(provenance, indent=2))
    else:
        print("\n[dry-run — no files written]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
