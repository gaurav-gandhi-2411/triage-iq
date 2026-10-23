"""Build the CANDIDATE k8s similar-issue index over derived_v2 with the PRODUCTION code path.

Offline artifact only (design: docs/QP3B_DESIGN_2026-10.md, flag 2 prerequisites): nothing here is
served and nothing is written under data/models. The only job of this wrapper is to map the
derived_v2 issues schema onto what `SimilarIssueRetriever.build_index` expects (`title`,
`body_clean`, `number`); `body_clean` is `preprocess.clean_text(body)[0]`, the same cleaning the
served index's training parquet carries. Embedding, text recipe (`_build_text` with the model's
tokenizer and max_seq_length), FAISS IndexFlatIP and meta.pkl come from production code unchanged.

Usage: python scripts/qp3b_build_candidate_index.py [--limit N] [--threads 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DERIVED = Path(r"D:\triageiq-data\derived_v2\kubernetes__kubernetes")
OUT_DIR = Path(
    r"D:\triageiq-data\models\candidates\similar_issue_index_kubernetes_kubernetes_bge_v2"
)
REPO = "kubernetes_kubernetes"
MODEL_KEY = "bge"
SEED = 42

log = logging.getLogger("qp3b_index")


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_builder_schema(issues: pd.DataFrame) -> pd.DataFrame:
    """Map derived_v2 issues to the columns `build_index` reads (title, body_clean, number)."""
    from triage_iq.data.preprocess import clean_text

    return pd.DataFrame(
        {
            "number": issues["number"].astype("int64"),
            "title": issues["title"].fillna(""),
            "body_clean": issues["body"].fillna("").map(lambda b: clean_text(b)[0]),
        }
    )


def hf_revision(model_name: str) -> str | None:
    """Commit hash of the locally cached HF snapshot the encoder was loaded from."""
    snaps = Path.home() / ".cache/huggingface/hub" / f"models--{model_name.replace('/', '--')}"
    refs = snaps / "refs" / "main"
    return refs.read_text().strip() if refs.exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="debug: first N issues only")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import sys

    sys.path.insert(0, str(ROOT / "src"))
    import faiss
    import torch

    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    from triage_iq.models.similar_issues import SUPPORTED_MODELS, SimilarIssueRetriever

    src = DERIVED / "issues.parquet"
    issues = pd.read_parquet(src, columns=["number", "title", "body"])
    if args.limit:
        issues = issues.head(args.limit)
    df = to_builder_schema(issues)
    log.info("issues: %d", len(df))

    t0 = time.perf_counter()
    det = SimilarIssueRetriever(repo=REPO, model_key=MODEL_KEY)
    det.build_index(df)
    seconds = time.perf_counter() - t0
    det.save(str(args.out))

    model_name = SUPPORTED_MODELS[MODEL_KEY]
    manifest: dict[str, Any] = {
        "kind": "CANDIDATE (offline; not served; not in data/models or MANIFEST.sha256)",
        "repo": "kubernetes/kubernetes",
        "n_rows": int(det.index.ntotal),
        "model_id": model_name,
        "model_revision": hf_revision(model_name),
        "max_seq_length": det.model.max_seq_length,
        "text_recipe": "SimilarIssueRetriever._build_text(title, clean_text(body)[0], "
        "tokenizer=model.tokenizer, max_tokens=model.max_seq_length)",
        "index_type": "faiss.IndexFlatIP over L2-normalised embeddings",
        "build_seconds_cpu": round(seconds, 1),
        "threads": args.threads,
        "source": {"file": str(src), "sha256": sha256_file(src)},
        "files": {p.name: sha256_file(p) for p in sorted(args.out.glob("*")) if p.is_file()},
        "script": "scripts/qp3b_build_candidate_index.py",
        "seed": SEED,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("done in %.1fs; %d rows", seconds, manifest["n_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
