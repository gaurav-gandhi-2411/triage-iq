"""Phase 2b: rebuild the BASELINE BGE FAISS indexes over the GROWN corpus (v2).

The shipped baseline indexes (data/models/dup_index_{repo}_bge) cover the pre-Phase-2b
corpus (15,000 k8s / 7,028 vscode records). The W3-retry eval retrieves against the grown
corpus (~30K / ~13.6K), so both baseline and fine-tuned models must search the SAME v2
corpus — otherwise the delta confounds model quality with corpus size. New indexes go to
*_v2 dirs; the shipped baseline artifacts are not touched (the production loader may
reference them).

Output: data/models/dup_index_{repo}_bge_v2/{index.faiss, meta.pkl}
Reproduce: python scripts/phase2b_build_baseline_index_v2.py  (GPU ~10-20 min)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def main() -> None:
    for repo in ("kubernetes_kubernetes", "microsoft_vscode"):
        df = pd.read_parquet(f"data/processed/issues_{repo}.parquet")
        log.info("[%s] corpus rows: %d", repo, len(df))
        retriever = SimilarIssueRetriever(repo=repo)
        retriever.build_index(df)
        out = f"data/models/dup_index_{repo}_bge_v2"
        retriever.save(out)
        log.info("[%s] baseline v2 index -> %s", repo, out)


if __name__ == "__main__":
    main()
