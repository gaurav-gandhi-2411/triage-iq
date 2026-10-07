from __future__ import annotations

# MUST be set before any torch/sentence_transformers import -- forces CPU, like freeze_similar_issues.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""One-time script: freeze the retrieval QUERY EMBEDDING for each eval issue whose repo is in
RESOLUTION_EMBEDDINGS_REPOS (ADR-0062), on CPU float32.

Serving reuses the vector SimilarIssueRetriever.retrieve_with_embedding() computes to build the
resolution predictor's emb_* features. The eval harness replays retrieval from frozen top-k hits
(eval/frozen_retriever.py), so it needs the matching frozen vector or the eval pipeline would
zero-fill emb_* and drift from production. Writes eval/frozen_query_embeddings.npz
(repos / numbers / embeddings) and cross-checks that the live top-k equals the frozen top-k in
eval_set.jsonl for every issue (aborts otherwise: a drifted index means freeze_similar_issues.py
must be re-run first).

Run whenever the FAISS index, the embedder, the query instruction, or the eval set changes.

Usage:
    python eval/freeze_query_embeddings.py [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
OUT_PATH = ROOT / "eval" / "frozen_query_embeddings.npz"
REPO_SLUGS = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}
K = 5


def main(dry_run: bool = False) -> None:
    from triage_iq.models.similar_issues import SimilarIssueRetriever
    from triage_iq.models.triage import RESOLUTION_EMBEDDINGS_REPOS

    issues = [
        json.loads(line)
        for line in EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    detectors: dict[str, SimilarIssueRetriever] = {}
    repos: list[str] = []
    numbers: list[int] = []
    embs: list[np.ndarray] = []
    for issue in issues:
        repo = issue["repo"]
        if repo not in RESOLUTION_EMBEDDINGS_REPOS:
            continue
        if repo not in detectors:
            idx_dir = ROOT / "data" / "models" / f"similar_issue_index_{REPO_SLUGS[repo]}_bge"
            detectors[repo] = SimilarIssueRetriever.load(str(idx_dir))
        num = int(issue["number"])
        # Same text the eval pipeline builds in triage._collect_signals: "{title}. {body_clean}".
        text = f"{issue['title']}. {issue['body']}"
        hits, emb = detectors[repo].retrieve_with_embedding(text, k=K, exclude_number=num)
        frozen = [s["number"] for s in issue["similar_issues"]]
        if [h["number"] for h in hits] != frozen:
            raise SystemExit(
                f"{issue['id']}: live top-{K} {[h['number'] for h in hits]} != frozen {frozen}; "
                "re-run eval/freeze_similar_issues.py first"
            )
        repos.append(repo)
        numbers.append(num)
        embs.append(emb.astype(np.float32))
    arr = np.stack(embs)
    print(f"froze {len(embs)} query embeddings, shape {arr.shape}, dtype {arr.dtype}")
    if dry_run:
        print("[dry-run -- no file written]")
        return
    np.savez(OUT_PATH, repos=np.array(repos), numbers=np.array(numbers, dtype=np.int64), embeddings=arr)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    main(dry_run=parser.parse_args().dry_run)
