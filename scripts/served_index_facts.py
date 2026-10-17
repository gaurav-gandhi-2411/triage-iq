"""Facts about the served similar-issue FAISS indexes (reads data/models, writes a report).

README #151 quotes the vscode index size and vector count; this makes them re-checkable
(scripts/readme_numbers_check.py reads reports/served_index_facts.json). Run from a checkout that holds
data/models (gitignored): python scripts/served_index_facts.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import faiss  # noqa: E402

OUT = ROOT / "reports" / "served_index_facts.json"
REPOS = {"microsoft/vscode": "microsoft_vscode", "kubernetes/kubernetes": "kubernetes_kubernetes"}


def main() -> None:
    result: dict = {}
    for repo, slug in REPOS.items():
        path = ROOT / "data" / "models" / f"similar_issue_index_{slug}_bge" / "index.faiss"
        index = faiss.read_index(str(path))
        result[repo] = {
            "ntotal": int(index.ntotal),
            "dim": int(index.d),
            "file_bytes": path.stat().st_size,
            "file_mb_decimal": round(path.stat().st_size / 1e6, 1),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
