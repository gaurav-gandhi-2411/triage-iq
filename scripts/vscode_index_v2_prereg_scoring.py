"""Item 3c scoring (criteria 2 and 3 + reported items) for the vscode index v2 prereg.

Pre-registration: docs/QP_VSCODE_INDEX_V2_PREREG_2026-11.md. Both indexes are scored through the
SHIPPED retrieval path, SimilarIssueRetriever.retrieve(text, k=5, exclude_number=<query>,
created_before=<query created_at>, strict_dates=True), on the 1,705-query 2026 vscode test split of
dup_task_pairs.parquet. Definitions are those of scripts/served_artifacts_offline_scoring.py
(build_dup_queries: answerable = at least one canonical in the index; hit = a canonical in the top 5).

The served index has no dates and 2 of its issues cannot be dated (prereg: they are dropped from the
served index for this comparison and reported). `make_served_dated_copy` builds that copy in a scratch
dir: same vectors in the same order minus the undated rows, with created_at. Nothing under
data/models is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU only

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

DATA = Path(r"D:\triageiq-data")
REPO = "microsoft/vscode"
SEED, RESAMPLES = 42, 10_000


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rate(k: int, n: int) -> dict[str, Any]:
    """k of n with its Wilson 95% interval."""
    from qp0_fixlink_audit import wilson

    w = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n if n else None), "wilson95": (list(w) if w else None)}


def paired_bootstrap(a: Any, b: Any, idx: Any) -> dict[str, Any]:
    """Paired bootstrap of mean(a) - mean(b) over the given resample index matrix (seed 42 upstream)."""
    import numpy as np

    diff = a[idx].mean(1) - b[idx].mean(1)
    return {
        "point": float(a.mean() - b.mean()),
        "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "seed": SEED,
        "resamples": RESAMPLES,
        "n": int(len(a)),
    }


def make_served_dated_copy(src: Path, dst: Path, derived_dir: Path, raw_dir: Path) -> dict[str, Any]:
    """Dated copy of the served index with the undated rows dropped (vectors re-added in order)."""
    import faiss
    import joblib
    import numpy as np
    import pandas as pd
    from check_prompt_neighbour_leakage import load_issue_dates, raw_created_at

    if dst.exists():
        raise FileExistsError(f"{dst} exists; refusing to overwrite")
    meta = joblib.load(src / "meta.pkl")
    index = faiss.read_index(str(src / "index.faiss"))
    nums = np.asarray(meta["issue_numbers"])
    parquet = load_issue_dates(derived_dir, REPO)
    keep: list[int] = []
    dates: list[Any] = []
    dropped: list[int] = []
    sources = {"parquet": 0, "raw": 0}
    for i, n in enumerate(nums.tolist()):
        ts, src_name = parquet.get(int(n)), "parquet"
        if ts is None or pd.isna(ts):
            ts, src_name = raw_created_at(raw_dir, REPO, int(n)), "raw"
        parsed = pd.NaT if ts is None else pd.to_datetime(ts, utc=True, errors="coerce")
        if parsed is pd.NaT:
            dropped.append(int(n))
            continue
        keep.append(i)
        dates.append(parsed.tz_localize(None).to_datetime64())
        sources[src_name] += 1
    vecs = np.asarray(index.reconstruct_n(0, index.ntotal), dtype=np.float32)[keep]
    new_index = faiss.IndexFlatIP(vecs.shape[1])
    new_index.add(vecs)
    dst.mkdir(parents=True)
    faiss.write_index(new_index, str(dst / "index.faiss"))
    new_meta = {
        **meta,
        "issue_numbers": nums[keep],
        "texts": [meta["texts"][i] for i in keep],
        "created_at": np.asarray(dates, dtype="datetime64[ns]"),
    }
    joblib.dump(new_meta, str(dst / "meta.pkl"))
    return {
        "n_before": int(len(nums)),
        "n_after": int(len(keep)),
        "dropped_undated_issue_numbers": dropped,
        "date_sources": sources,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--served-dir", type=Path, default=ROOT / "data" / "models" / "similar_issue_index_microsoft_vscode_bge")
    ap.add_argument("--candidate-dir", type=Path, required=True, help="dated v2 candidate")
    ap.add_argument("--work-dir", type=Path, required=True, help="scratch; the served dated copy is built here")
    ap.add_argument("--derived-dir", type=Path, default=DATA / "derived_v2")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--prior-scores", type=Path, default=None, help="2026-10-31 unfiltered/offline scores.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    import qk_kaggle
    import torch
    from check_prompt_neighbour_leakage import load_issue_dates, raw_created_at
    from served_artifacts_offline_scoring import build_dup_queries

    from triage_iq.models.similar_issues import SimilarIssueRetriever, _build_text

    torch.set_num_threads(a.threads)
    served_copy = a.work_dir / "served_dated_dropped"
    dated_info = make_served_dated_copy(a.served_dir, served_copy, a.derived_dir, a.raw_dir)
    print("served dated copy:", json.dumps(dated_info), flush=True)

    pairs = pd.read_parquet(DATA / "splits" / "microsoft__vscode" / "dup_task_pairs.parquet")
    pairs = pairs[pairs["split_of_source"] == "test"]
    texts = qk_kaggle.build_texts(REPO, "issues", DATA)
    by_num = texts.set_index(texts["number"].astype(int))
    parquet = load_issue_dates(a.derived_dir, REPO)
    indexes = {"served": served_copy, "v2": a.candidate_dir}
    rets = {k: SimilarIssueRetriever.load(str(d)) for k, d in indexes.items()}
    queries = {}
    for name, ret in rets.items():
        queries[name], counts = build_dup_queries(
            pairs, {int(n) for n in ret.issue_numbers}, set(texts["number"].astype(int))
        )
        print(name, "n_vectors", len(ret.issue_numbers), counts, flush=True)
    n_q = len(queries["served"])
    assert [q.source for q in queries["served"]] == [q.source for q in queries["v2"]] and n_q == 1705

    def q_date(n: int) -> Any:
        ts = parquet.get(n)
        if ts is None or pd.isna(ts):
            ts = raw_created_at(a.raw_dir, REPO, n)
        if ts is None:
            raise ValueError(f"query #{n} has no created_at (fail closed)")
        return ts

    hits = {k: np.zeros(n_q, dtype=np.int8) for k in rets}
    t0 = time.time()
    done = 0
    for j in range(n_q):
        for name, ret in rets.items():
            q = queries[name][j]
            if not q.canonicals_indexed:
                continue
            row = by_num.loc[q.source]
            text = _build_text(pd.Series([row["title"]]), pd.Series([row["body_clean"]]), ret.model.tokenizer)[0]
            top = ret.retrieve(text, k=5, exclude_number=q.source, created_before=q_date(q.source), strict_dates=True)
            hits[name][j] = bool({h["number"] for h in top} & set(q.canonicals_indexed))
            done += 1
        if j % 200 == 0:
            print(f"query {j}/{n_q} retrievals={done} elapsed={time.time() - t0:.0f}s", flush=True)

    answerable = {k: np.array([bool(q.canonicals_indexed) for q in queries[k]], dtype=np.int8) for k in rets}
    idx = np.random.default_rng(SEED).integers(0, n_q, size=(RESAMPLES, n_q))
    both = np.flatnonzero(answerable["served"] & answerable["v2"])
    res: dict[str, Any] = {
        "prereg": "docs/QP_VSCODE_INDEX_V2_PREREG_2026-11.md",
        "n_queries": n_q,
        "filter": "retrieve(k=5, exclude_number=query, created_before=query created_at, strict_dates=True)",
        "index_files": {k: {p: sha256_file(d / p) for p in ("index.faiss", "meta.pkl")} for k, d in indexes.items()},
        "served_dated_copy": dated_info,
        "n_vectors": {k: int(len(r.issue_numbers)) for k, r in rets.items()},
        "criterion_2_answerable_share": {
            "served": rate(int(answerable["served"].sum()), n_q),
            "v2": rate(int(answerable["v2"].sum()), n_q),
            "paired_bootstrap_v2_minus_served": paired_bootstrap(answerable["v2"], answerable["served"], idx),
        },
        "criterion_3_r5_over_all_queries": {
            "served": rate(int(hits["served"].sum()), n_q),
            "v2": rate(int(hits["v2"].sum()), n_q),
            "paired_bootstrap_v2_minus_served": paired_bootstrap(hits["v2"], hits["served"], idx),
        },
        "reported_r5_over_answerable": {
            k: rate(int(hits[k].sum()), int(answerable[k].sum())) for k in rets
        },
        "reported_overlap_answerable_by_both": {
            "n": int(len(both)),
            "served": rate(int(hits["served"][both].sum()), len(both)),
            "v2": rate(int(hits["v2"][both].sum()), len(both)),
            "paired_bootstrap_v2_minus_served": paired_bootstrap(
                hits["v2"][both], hits["served"][both],
                np.random.default_rng(SEED).integers(0, len(both), size=(RESAMPLES, len(both))),
            ),
        },
    }
    if a.prior_scores:
        prior = json.loads(a.prior_scores.read_text(encoding="utf-8"))["created_before"]
        res["prior_2026_10_31_created_before"] = {
            "source": str(a.prior_scores),
            "served_r5_all_k": prior["served"]["r5_over_all_1705"]["k"],
            "v2_r5_all_k": prior["candidate"]["r5_over_all_1705"]["k"],
            "now_served_r5_all_k": res["criterion_3_r5_over_all_queries"]["served"]["k"],
            "now_v2_r5_all_k": res["criterion_3_r5_over_all_queries"]["v2"]["k"],
        }
    a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
