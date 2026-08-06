"""LEVER 1: rebuild the D1-scoped full-corpus index using the fixed, tokenizer-based
_build_text() truncation instead of the 512-CHARACTER cut.

This does NOT touch the served production index (`data/models/dup_index_{repo}_bge`) --
writes to a distinct, clearly-suffixed directory, exactly the same "measure first, no prod
artifact touched" discipline as scripts/d1_build_full_corpus_index.py itself. A prod re-publish
is a separate, explicitly-escalated step gated on GG reviewing this lever's measured result.

Reads:  data/processed/issues_{repo}.parquet
Writes: data/models/d1_full_corpus_index_{repo}_bge_lever1/
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
        out_dir = MODELS_DIR / f"d1_full_corpus_index_{repo}_bge_lever1"
        df = pd.read_parquet(PROCESSED_DIR / f"issues_{repo}.parquet")
        log.info(
            "[%s] building LEVER1 (tokenizer-truncated) full-corpus index: %d issues, #%d-#%d",
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
