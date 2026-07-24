"""D1 task 3: build a full-corpus BGE index for the clean-eval baseline (NOT the served index).

The currently-served index (`data/models/dup_index_{repo}_bge`, loaded by
`src/triage_iq/api/loader.py`) is stale: k8s covers only #1-15002 (built before the Phase 2b
forward-scrape to #30000), vscode covers only 7,028 specific issues. D1's new hand-verified
eval sets are dominated by pairs mined from AFTER those ranges (k8s_forward_scrape, vscode's
2016-2025 dup_comment scrape) -- so measuring against the stale served index would cover only
16/150 k8s pairs and 4/200 vscode-duplicate pairs, nowhere near powered.

This builds a SEPARATE, D1-scoped index over the full currently-processed corpus
(data/processed/issues_{repo}.parquet -- the same off-the-shelf pretrained BAAI/bge-base-en-v1.5
embedder already in production, same SimilarIssueRetriever class and build_index() method as
scripts/08_build_similar_issue_index.py). This is embedding INFERENCE, not training -- no
gradient updates, no gold-pair fitting -- the identical "zero leakage, pretrained embedder"
reasoning ADR-0030 established for the served index. Written to a distinct directory; the
served production artifact is never read for writing and is untouched.

Reads:  data/processed/issues_{repo}.parquet
Writes: data/models/d1_full_corpus_index_{repo}_bge/  (NOT the served dup_index_* path)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
REPOS = ["kubernetes_kubernetes", "microsoft_vscode"]


def main() -> None:
    for repo in REPOS:
        out_dir = MODELS_DIR / f"d1_full_corpus_index_{repo}_bge"
        if (out_dir / "index.faiss").exists():
            log.info("[%s] index already exists at %s, skipping", repo, out_dir)
            continue
        df = pd.read_parquet(PROCESSED_DIR / f"issues_{repo}.parquet")
        log.info(
            "[%s] building full-corpus index: %d issues, #%d-#%d",
            repo,
            len(df),
            int(df["number"].min()),
            int(df["number"].max()),
        )
        detector = SimilarIssueRetriever(repo=repo, model_key="bge")
        t0 = time.perf_counter()
        detector.build_index(df)
        log.info("[%s] built in %.1fs", repo, time.perf_counter() - t0)
        detector.save(str(out_dir))
        log.info("[%s] saved to %s", repo, out_dir)


if __name__ == "__main__":
    main()
