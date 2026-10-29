"""Re-freeze the Q3b 150-issue set for the single final-configuration re-record (D92.6).

Offline, no LLM call, no cassette. Reads the committed eval/sets/quality_push_2026-10/eval_set.jsonl
(the sampled issues, strata and weights are NOT redrawn) and, in place:
 1. re-freezes `similar_issues` (and the `*_in_index` meta) of the repos named by --index-dir with
    the given index, through qp3_build_eval_set.freeze: the same retrieve_with_embedding call, the
    same K_RETRIEVE=50 and the same created-before-the-query filter as the original build. Rows of
    other repos are left untouched (byte-identical);
 2. adds `body_raw` (the uncleaned issue body from derived_v2 issues.parquet, matched by number) to
    every row, and checks clean_text(body_raw) equals the stored cleaned body;
 3. keeps frozen_query_embeddings.npz byte-identical when the re-frozen query vectors agree to
    1e-5 (they depend on the query text and encoder only, not on the index; CPU float32 reruns
    differ by ~3e-7), and stops if they do not;
 4. patches reports/qp3_eval_set_150_2026-10.json (freeze, files, relevance_scoring, refreeze).

Usage: python scripts/qp3b_refreeze_set.py --index-dir kubernetes/kubernetes=<index dir> [--scratch DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp3_build_eval_set as q  # noqa: E402
from qp3_eval_set_lib import dump_line  # noqa: E402

from triage_iq.data.preprocess import clean_text  # noqa: E402


def verify_manifest(index_dir: Path) -> dict[str, str]:
    """sha256 of index.faiss / meta.pkl, checked against the directory's manifest.json `files`."""
    manifest = json.loads((index_dir / "manifest.json").read_text("utf-8"))
    got: dict[str, str] = {}
    for name, want in manifest["files"].items():
        got[name] = hashlib.sha256((index_dir / name).read_bytes()).hexdigest()
        if got[name] != want:
            raise ValueError(f"{index_dir / name}: sha256 {got[name]} != manifest {want}")
    return got


EMB_ATOL = (
    1e-5  # CPU float32 encodes differ by ~3e-7 between runs (thread-order sums); cosine > 1 - 1e-6
)


def embeddings_match(old: Any, new: Any) -> tuple[bool, float]:
    """(same rows in the same order and every component within EMB_ATOL, max abs difference)."""
    if not (
        np.array_equal(old["repos"], new["repos"])
        and np.array_equal(old["numbers"], new["numbers"])
    ):
        return False, float("inf")
    diff = float(np.abs(old["embeddings"] - new["embeddings"]).max())
    return diff <= EMB_ATOL, diff


def raw_bodies(repo: str, numbers: list[int]) -> dict[int, str]:
    """number -> uncleaned body from derived_v2 issues.parquet (the column the builder cleans)."""
    iss = q._read(q.REPOS[repo]["slug"], "issues", columns=["number", "body"],
                  filters=[("number", "in", numbers)])  # fmt: skip
    return {int(n): ("" if pd.isna(b) else str(b)) for n, b in zip(iss["number"], iss["body"], strict=True)}  # fmt: skip


def add_body_raw(rows: list[dict[str, Any]], raw: dict[str, dict[int, str]]) -> dict[str, Any]:
    """Insert `body_raw` right after `body` in every row (in place); fail loudly on a missing issue
    or on a raw body that does not clean to the stored body. Returns per-repo change counts."""
    stats: dict[str, dict[str, int]] = {}
    for r in rows:
        st = stats.setdefault(r["repo"], {"rows": 0, "text_changed": 0, "identical": 0})
        body_raw = raw[r["repo"]].get(r["number"])
        if body_raw is None:
            raise KeyError(f"{r['id']}: not in derived_v2 issues")
        if clean_text(body_raw)[0] != r["body"]:
            raise ValueError(f"{r['id']}: clean_text(body_raw) != stored body")
        items = [(k, v) for k, v in r.items() if k != "body_raw"]
        r.clear()
        for k, v in items:
            r[k] = v
            if k == "body":
                r["body_raw"] = body_raw
        st["rows"] += 1
        st["text_changed" if body_raw != r["body"] else "identical"] += 1
    return stats


def relevance_counts(rows: list[dict[str, Any]], repos: list[str]) -> dict[str, dict[str, int]]:
    """Per repo: rows with a dup/fix-peer target, with one in the frozen index, indexable earlier."""
    out = {}
    for repo in repos:
        mine = [r["meta"] for r in rows if r["repo"] == repo]
        out[repo] = {
            "with_dup_or_peer": sum(bool(m["dup_targets"] or m["fix_peers"]) for m in mine),
            "target_in_served_index": sum(
                bool(m["dup_targets_in_index"] or m["fix_peers_in_index"]) for m in mine
            ),
            "indexable_in_derived_v2_before_query": sum(bool(m["indexable_targets"]) for m in mine),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index-dir", action="append", default=[], metavar="REPO=DIR", required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    a = ap.parse_args(argv)
    index_dirs = {k: Path(v) for k, v in (s.split("=", 1) for s in a.index_dir)}
    unknown = set(index_dirs) - set(q.REPOS)
    if unknown:
        raise SystemExit(f"unknown repo(s): {sorted(unknown)}")
    shas = {repo: verify_manifest(d) for repo, d in index_dirs.items()}
    a.scratch.mkdir(parents=True, exist_ok=True)

    path, npz = q.OUT_DIR / "eval_set.jsonl", q.OUT_DIR / "frozen_query_embeddings.npz"
    rows = [json.loads(x) for x in path.read_text("utf-8").split("\n") if x]
    old_nb = {r["id"]: [h["number"] for h in r["similar_issues"]] for r in rows}
    old_other = {r["id"]: dump_line(r) for r in rows if r["repo"] not in index_dirs}

    info = q.freeze(rows, index_dirs=index_dirs, only_repos=set(index_dirs),
                    embeddings_out=a.scratch / "emb.npz")  # fmt: skip
    assert all(dump_line(r) == old_other[r["id"]] for r in rows if r["id"] in old_other)
    old_e, new_e = np.load(npz), np.load(a.scratch / "emb.npz")
    emb_equal, emb_diff = embeddings_match(old_e, new_e)
    if not emb_equal:
        raise SystemExit(f"query embeddings changed (max abs diff {emb_diff}): investigate")

    raw = {repo: raw_bodies(repo, [r["number"] for r in rows if r["repo"] == repo]) for repo in q.REPOS}  # fmt: skip
    body_stats = add_body_raw(rows, raw)
    path.write_text("".join(dump_line(r) + "\n" for r in rows), "utf-8", newline="\n")

    changed = {
        repo: sum(old_nb[r["id"]] != [h["number"] for h in r["similar_issues"]]
                  for r in rows if r["repo"] == repo)
        for repo in index_dirs
    }  # fmt: skip
    rep = json.loads(q.REPORT.read_text("utf-8"))
    rep["freeze"]["neighbour_filter"].update(info["neighbour_filter"])
    rep["freeze"]["index_sha256"].update(info["index_sha256"])
    rep["files"] = {p.name: q.sha256_file(p) for p in sorted(q.OUT_DIR.iterdir())}
    per_repo = relevance_counts(rows, list(q.REPOS))
    rep["relevance_scoring"]["per_repo"] = per_repo
    unscorable = sorted(r for r, c in per_repo.items() if c["target_in_served_index"] == 0)
    rep["relevance_scoring"]["status"] = (
        "re-frozen 2026-10-29 for the final configuration; UNSCORABLE for "
        f"{unscorable or 'no repo'}: 0 sampled duplicate/fix targets are in their frozen index"
    )
    rep["refreeze"] = {
        "script": "scripts/qp3b_refreeze_set.py", "index_dirs": {k: str(v) for k, v in index_dirs.items()},
        "index_files_sha256": shas, "rows_top_k_changed": changed,
        "query_embeddings_max_abs_diff_so_npz_untouched": emb_diff, "body_raw": body_stats,
        "other_repo_rows_byte_identical_apart_from_body_raw": True,
    }  # fmt: skip
    q.REPORT.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str), "utf-8")
    print(json.dumps({"changed": changed, "filter": info["neighbour_filter"], "body_raw": body_stats,
                      "relevance": per_repo}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
