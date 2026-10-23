"""C1 (docs/QP3B_DESIGN_2026-10.md): flags-off strict replay of the 127 recorded issues.

Zero tokens, zero network: sockets are blocked and the cassette is strict (a miss raises), so a
completed run proves every one of the 127 synthesis calls hit an existing cassette key, i.e. the
messages (system + few-shot + user prompt) are byte-identical to the recorded ones. The script
also dumps every plan and metadata (timing fields removed) so two runs (main vs a branch) can be
compared with --compare for byte-identical JSON. Served models come from --models-dir and are
hash-checked against that tree's MANIFEST.sha256 first.

Usage:
  python scripts/qp3b_c1_replay.py --models-dir DIR --out OUT.json
  python scripts/qp3b_c1_replay.py --compare A.json B.json
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

COMMITTED = (ROOT / "eval" / "cassettes" / "eval_cassette.json", ROOT / "eval" / "eval_set.jsonl")
EXPANSION = (
    ROOT / "reports" / "vscode_expansion_2026-10-12" / "eval_cassette.json",
    ROOT / "eval" / "expansions" / "vscode_2026-10-12" / "eval_set.jsonl",
)
TIMING_KEYS = ("_latency_ms",)  # every metadata key ending in this is wall-clock noise


def block_sockets() -> None:
    """Make any network attempt raise, so a silent live call cannot happen."""

    def _deny(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("network blocked: C1 replay must be offline")

    socket.socket.connect = _deny  # type: ignore[method-assign]
    socket.create_connection = _deny  # type: ignore[assignment]


def verify_manifest(models_dir: Path) -> int:
    """SHA-256 every MANIFEST line whose file exists under the served tree; returns the count.

    Raises SystemExit on a mismatch or when nothing was checked (an empty check is a failure).
    """
    tree = models_dir.parent.parent
    checked = 0
    for line in (models_dir / "MANIFEST.sha256").read_text("utf-8").splitlines():
        digest, _, rel = line.strip().partition(" ")
        rel = rel.lstrip("*").strip()
        # data/processed is read from this checkout (run_eval.PROCESSED_DIR), models from the tree
        path = next((t / rel for t in (tree, ROOT) if (t / rel).is_file()), None)
        if not digest or path is None:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{rel}: sha256 differs from MANIFEST")
        checked += 1
    if not checked:
        raise SystemExit("MANIFEST check verified zero files: failed check")
    return checked


def strip_timing(meta: dict[str, Any]) -> dict[str, Any]:
    """Metadata without wall-clock fields."""
    return {k: v for k, v in meta.items() if not k.endswith(TIMING_KEYS)}


def replay(cassette: Path, eval_set: Path) -> list[dict[str, Any]]:
    """Strict replay of one recorded set; one record per issue."""
    from checkpoint_validation import build_replay_context, issue_to_row

    issues, _c, assistants, _ = build_replay_context(cassette, eval_set, need_judge=False)
    out = []
    for issue in issues:
        plan, meta = assistants[issue["repo"]].triage_with_metadata(issue_to_row(issue))
        out.append(
            {
                "id": issue["id"],
                "plan": plan.model_dump(mode="json"),
                "metadata": strip_timing(meta),
            }
        )
    return out


def compare(a: Path, b: Path) -> int:
    """Exit 0 iff the two dumps are byte-identical JSON; prints the first differing ids."""
    ja, jb = (json.loads(p.read_text("utf-8")) for p in (a, b))
    ka = {r["id"]: json.dumps(r, sort_keys=True) for r in ja["rows"]}
    kb = {r["id"]: json.dumps(r, sort_keys=True) for r in jb["rows"]}
    diff = sorted(k for k in ka.keys() | kb.keys() if ka.get(k) != kb.get(k))
    print(json.dumps({"n_a": len(ka), "n_b": len(kb), "n_differing": len(diff), "ids": diff[:10]}))
    return int(bool(diff) or not ka)


def main(argv: list[str]) -> int:
    if argv[:1] == ["--compare"]:
        return compare(Path(argv[1]), Path(argv[2]))
    args = dict(zip(argv[::2], argv[1::2], strict=False))
    models_dir, out = Path(args["--models-dir"]), Path(args["--out"])
    block_sockets()
    n_files = verify_manifest(models_dir)
    import run_eval

    run_eval.MODELS_DIR = models_dir
    rows = replay(*COMMITTED) + replay(*EXPANSION)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "an issue appears in both recorded sets"
    out.write_text(
        json.dumps({"manifest_files_verified": n_files, "rows": rows}, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"issues_replayed": len(rows), "manifest_files_verified": n_files}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
