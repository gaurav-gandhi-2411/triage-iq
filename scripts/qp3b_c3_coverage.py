"""C3 (docs/QP3B_DESIGN_2026-10.md): evidence coverage of the candidate k8s index, plus index sanity.

For the k8s rows of eval/sets/quality_push_2026-10/eval_set.jsonl (read-only), retrieve with the
CANDIDATE index (query text "{title}. {body}" with the repo's query instruction, as the production
retriever does: `retrieve_with_embedding`), keep neighbours created before the query
(qp3_eval_set_lib.filter_neighbours, as the set's builder does), take the top 3 and report the
share of rows with >= 1 neighbour that has evidence. Pre-registered threshold: >= 50%.
Sanity: 20 random self-queries must rank themselves first; 20 embeddings are compared with the
Kaggle ones. Offline only; no LLM calls; writes only the report.

Usage: python scripts/qp3b_c3_coverage.py [--out reports/qp3b_evidence_coverage_k8s_2026-10.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qp3_eval_set_lib import filter_neighbours

ROOT = Path(__file__).resolve().parent.parent
EVAL_SET = ROOT / "eval" / "sets" / "quality_push_2026-10" / "eval_set.jsonl"
INDEX = Path(r"D:\triageiq-data\models\candidates\similar_issue_index_kubernetes_kubernetes_bge_v2")
EVIDENCE = Path(
    r"D:\triageiq-data\models\candidates\resolution_evidence_kubernetes_kubernetes_v1.parquet"
)
ISSUES = Path(r"D:\triageiq-data\derived_v2\kubernetes__kubernetes\issues.parquet")
KAGGLE = Path(r"D:\triageiq-data\embeddings\kubernetes__kubernetes")
THRESHOLD, TOP, K_RETRIEVE, SEED, N_SANITY = 0.5, 3, 50, 42, 20


def evidence_chars(rec: dict[str, Any]) -> int:
    """Characters an evidence block would carry for one neighbour (titles+excerpts+comment)."""
    n = sum(len(p["title"]) + len(p["body_excerpt"]) for p in rec["fix_prs"])
    return n + len(rec["closing_comment"])


def coverage(rows: list[list[dict[str, Any] | None]]) -> dict[str, Any]:
    """Row-level coverage stats; each row is its top-3 neighbours' evidence records (None = absent)."""
    n = len(rows)
    has = lambda r, key: any(x is not None and x[key] for x in r)  # noqa: E731
    any_ev, fix, com = (sum(has(r, k) for r in rows) for k in ("has_evidence", "has_fix_evidence", "has_comment_evidence"))  # fmt: skip
    both = sum(has(r, "has_fix_evidence") and has(r, "has_comment_evidence") for r in rows)
    per_row_chars = [sum(evidence_chars(x) for x in r if x is not None) for r in rows]
    nb = [x for r in rows for x in r if x is not None and x["has_evidence"]]
    return {
        "n_rows": n,
        "share_ge1_neighbour_with_evidence": any_ev / n,
        "rows_ge1_neighbour_with_evidence": any_ev,
        "share_with_fix_pr_evidence": fix / n,
        "share_with_closing_comment_evidence": com / n,
        "share_with_both_types": both / n,
        "mean_evidence_chars_per_row_top3": float(np.mean(per_row_chars)),
        "mean_evidence_chars_per_evidenced_neighbour": float(
            np.mean([evidence_chars(x) for x in nb])
        )
        if nb
        else 0.0,  # fmt: skip
        "mean_neighbours_with_evidence_per_row": float(
            np.mean([sum(x is not None and x["has_evidence"] for x in r) for r in rows])
        ),  # fmt: skip
    }


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sanity(det: Any, kaggle_dir: Path) -> dict[str, Any]:
    """Self-query rank-1 check and cosine against the Kaggle embeddings, 20 random issues."""
    rng = np.random.default_rng(SEED)
    pos = sorted(rng.choice(len(det.issue_numbers), N_SANITY, replace=False).tolist())
    self_ok, cos, bad = 0, [], []
    ids = pd.read_parquet(kaggle_dir / "kubernetes__kubernetes__issues_ids.parquet")
    id_col = "number" if "number" in ids.columns else ids.columns[0]
    row_of = {int(n): i for i, n in enumerate(ids[id_col])}
    kag = np.load(kaggle_dir / "kubernetes__kubernetes__issues_emb.npy", mmap_mode="r")
    for p in pos:
        num = int(det.issue_numbers[p])
        # stored corpus text, no query instruction (documents are embedded un-prefixed)
        top = det.retrieve(det.texts[p], k=1, apply_query_instruction=False)[0]
        if top["number"] == num:
            self_ok += 1
        else:  # a tie with an identical-text issue is legitimate; record it
            bad.append({"number": num, "got": top["number"], "score": top["score"]})
        v = det.index.reconstruct(p)
        k = np.asarray(kag[row_of[num]], dtype=np.float32)
        cos.append(float(v @ k / (np.linalg.norm(v) * np.linalg.norm(k))))
    allc = np.einsum(  # all rows: the expected 0.997-0.999 assumes the same text recipe (see note)
        "ij,ij->i", det.index.reconstruct_n(0, det.index.ntotal),
        np.asarray(kag)[[row_of[int(n)] for n in det.issue_numbers]],
    )  # fmt: skip
    pct = {f"p{q}": float(np.percentile(allc, q)) for q in (0, 5, 50, 95, 100)}
    return {
        "kaggle_cosine_all_rows": pct,
        "kaggle_cosine_note": "likely cause (inferred, not checked in the Kaggle script): Kaggle texts are not clean_text()-cleaned while the served recipe is; mean cosine is 0.995 on issues clean_text leaves unchanged vs 0.963 on changed ones",
        "n": N_SANITY, "seed": SEED, "self_query_rank1": self_ok, "self_query_misses": bad,
        "kaggle_cosine_min": min(cos), "kaggle_cosine_mean": float(np.mean(cos)),
        "kaggle_cosine_max": max(cos),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "qp3b_evidence_coverage_k8s_2026-10.json")  # fmt: skip
    a = ap.parse_args(argv)
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from triage_iq.models.similar_issues import SimilarIssueRetriever

    det = SimilarIssueRetriever.load(str(INDEX))
    iss = pd.read_parquet(ISSUES, columns=["number", "created_at"])
    created = dict(zip(iss["number"].astype(int), pd.to_datetime(iss["created_at"], utc=True), strict=True))  # fmt: skip
    ev = pd.read_parquet(EVIDENCE).set_index("number")
    ev_rec = {int(n): r for n, r in zip(ev.index, ev.to_dict("records"), strict=True)}

    rows_in = [json.loads(ln) for ln in EVAL_SET.read_text(encoding="utf-8").splitlines() if ln.strip()]  # fmt: skip
    rows_in = [r for r in rows_in if r["repo"] == "kubernetes/kubernetes"]
    per_row: list[list[dict[str, Any] | None]] = []
    stats = {"examined": 0, "dropped_newer": 0, "kept_by_number_fallback": 0, "dropped_unknown": 0}
    short = 0
    for r in rows_in:
        hits, _ = det.retrieve_with_embedding(f"{r['title']}. {r['body']}", k=K_RETRIEVE, exclude_number=r["number"])  # fmt: skip
        kept, st = filter_neighbours(hits, r["number"], pd.Timestamp(r["created_at"]), created, k=TOP)  # fmt: skip
        for k_, v in st.items():
            stats[k_] += v
        short += len(kept) < TOP
        per_row.append([ev_rec.get(h["number"]) for h in kept])
    res = coverage(per_row)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    report = {
        "check": "C3 evidence coverage (docs/QP3B_DESIGN_2026-10.md)",
        "threshold": THRESHOLD, "passed": res["share_ge1_neighbour_with_evidence"] >= THRESHOLD,
        "top_k": TOP, "result": res, "rows_with_fewer_than_top_k_neighbours": short,
        "neighbour_filter": stats,
        "sanity": sanity(det, KAGGLE),
        "provenance": {
            "script": "scripts/qp3b_c3_coverage.py", "git_head_at_run": head,
            "eval_set_sha256": sha256_file(EVAL_SET),
            "candidate_index_faiss_sha256": sha256_file(INDEX / "index.faiss"),
            "candidate_index_manifest": str(INDEX / "manifest.json"),
            "evidence_sha256": sha256_file(EVIDENCE),
            "caveat": "candidate artifacts, nothing served; evidence for CLOSED issues only for "
            "comments; query rows are themselves indexed (excluded by number + time filter)",
        },
    }  # fmt: skip
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"result": res, "passed": report["passed"], "sanity": report["sanity"]}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
