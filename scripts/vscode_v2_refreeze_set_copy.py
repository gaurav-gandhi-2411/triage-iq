"""Item 3b helper: re-freeze the vscode rows of a COPY of a recording set against the dated v2 index.

Prereg docs/QP_VSCODE_INDEX_V2_PREREG_2026-11.md, criterion 1. Copies `--src-set` to `--dst-set`
(the source is read only, never written; `--dst-set` must not exist), then for every
microsoft/vscode row replaces `similar_issues` with the top-5 from the dated candidate index using
the same call as eval/freeze_similar_issues.py (text = "title. body", exclude self, strict
created-before filter). Non-vscode rows are copied byte-for-byte (asserted). The query-embedding
npz is copied unchanged: only repos in triage.RESOLUTION_EMBEDDINGS_REPOS (k8s) carry one.
Prints/writes how many vscode rows changed neighbours.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU float32, as the freeze scripts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VSCODE = "microsoft/vscode"
K = 5


def refreeze(src_set: Path, dst_set: Path, index_dir: Path) -> dict:
    """Write the re-frozen copy and return the change summary."""
    from triage_iq.models.similar_issues import SimilarIssueRetriever

    if dst_set.exists():
        raise FileExistsError(f"{dst_set} exists; refusing to overwrite")
    lines = (src_set / "eval_set.jsonl").read_bytes().decode("utf-8").split("\n")
    ret = SimilarIssueRetriever.load(str(index_dir))
    assert ret.created_at is not None, "candidate index has no created_at"
    out_lines: list[str] = []
    changed, per_row, n_vscode, n_other_identical = 0, [], 0, 0
    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
        issue = json.loads(line)
        if issue["repo"] != VSCODE:
            out_lines.append(line)  # byte-identical
            n_other_identical += 1
            continue
        n_vscode += 1
        new = ret.retrieve(
            issue["title"] + ". " + issue["body"],
            k=K,
            exclude_number=int(issue["number"]),
            created_before=issue["created_at"],
            strict_dates=True,
        )
        old_nums = [s["number"] for s in issue["similar_issues"]]
        new_nums = [s["number"] for s in new]
        changed += old_nums != new_nums
        per_row.append({"id": issue["id"], "old": old_nums, "new": new_nums})
        out_lines.append(json.dumps({**issue, "similar_issues": new}, ensure_ascii=False))
    dst_set.mkdir(parents=True)
    for p in src_set.iterdir():
        if p.name == "eval_set.jsonl":
            continue
        if p.is_file():
            shutil.copyfile(p, dst_set / p.name)  # RECORDING_CONFIG.json, frozen_query_embeddings.npz
    (dst_set / "cassettes").mkdir()  # empty: the copy is never recorded into
    (dst_set / "eval_set.jsonl").write_bytes("\n".join(out_lines).encode("utf-8"))
    return {
        "n_vscode_rows": n_vscode,
        "n_other_rows_byte_identical": n_other_identical,
        "n_vscode_rows_changed_neighbours": changed,
        "n_vscode_rows_unchanged": n_vscode - changed,
        "rows": per_row,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src-set", type=Path, required=True)
    ap.add_argument("--dst-set", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()
    res = refreeze(a.src_set, a.dst_set, a.index_dir)
    if a.json_out:
        a.json_out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
