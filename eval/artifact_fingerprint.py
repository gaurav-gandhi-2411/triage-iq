from __future__ import annotations

"""SHA-256 fingerprinting of every model artifact that feeds a triage prompt.

ADR-0059: `eval/record_cassettes.py` resolves its model directory relative to whichever
checkout's copy of the script actually executes (`Path(__file__).parent.parent`). A recording
pass launched from the wrong checkout silently produces a cassette conditioned on the WRONG
classifier -- no exception, no visible symptom, because `load_classifier()` and
`_collect_signals()` don't know or care which checkout they were loaded from. The 2026-09-05
"full 64-issue re-record" incident: the recorded cassette's embedded `classifier_top3` values
matched the main repo checkout's pre-retrain classifier exactly, not this worktree's retrained
one, discovered only because a later replay hit `CassetteMissError`.

This module makes classifier/retrieval/resolution identity a first-class, checkable fact
instead of an assumption: every artifact that can change what a prompt looks like gets hashed,
the combined hash is folded into the recording checkpoint's key (so a swapped artifact can
never be silently mistaken for "already recorded", the same discipline
`record_cassettes.py._compute_prompt_hash()` already applies to prompt/schema changes), and
the per-repo hashes are stamped onto each cassette entry as it's written (so the cassette
itself -- the artifact that actually ships and gets replayed in CI -- can be checked against
the current checkout without needing the checkpoint file, which gets cleared/regenerated
across recording campaigns).
"""

import hashlib
import json
from pathlib import Path

REPO_SLUGS = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}

# Repo-specific artifacts (one path per repo, keyed by relative path so the manifest reads the
# same regardless of which absolute checkout produced it).
_PER_REPO_TEMPLATES = [
    "data/models/component_classifier_{slug}.pkl",
    "data/models/resolution_predictor_{slug}.pkl",
    "data/processed/{slug}_temporal_train.parquet",
    "data/models/similar_issue_index_{slug}_bge/index.faiss",
    "data/models/similar_issue_index_{slug}_bge/meta.pkl",
]

# Shared across repos.
_SHARED_PATHS = [
    "data/models/cqr_conformal_adjustments.json",
]

EXPECTED_HASHES_PATH_REL = "eval/cassettes/EXPECTED_ARTIFACT_HASHES.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_paths_for_repo(repo: str) -> list[str]:
    """Relative (repo-root-anchored, POSIX-separated) paths of every artifact that feeds
    `repo`'s triage prompt, plus the shared conformal store."""
    slug = REPO_SLUGS[repo]
    return [t.format(slug=slug) for t in _PER_REPO_TEMPLATES] + list(_SHARED_PATHS)


def compute_artifact_hashes(root: Path, repo: str | None = None) -> dict[str, str]:
    """SHA-256 of every prompt-feeding artifact, keyed by relative path.

    `repo=None` hashes every repo's artifacts (used for the startup print / expected-hash
    file); `repo=<name>` hashes only that repo's own artifacts (used to tag a single cassette
    entry with just the artifacts that actually influenced it).
    """
    repos = [repo] if repo is not None else list(REPO_SLUGS)
    rel_paths: set[str] = set()
    for r in repos:
        rel_paths.update(artifact_paths_for_repo(r))

    hashes: dict[str, str] = {}
    for rel in sorted(rel_paths):
        p = root / rel
        if not p.exists():
            hashes[rel] = "MISSING"
            continue
        hashes[rel] = _sha256_file(p)
    return hashes


def combined_hash(hashes: dict[str, str]) -> str:
    """Short, order-independent fingerprint of a full hash dict, for checkpoint keying."""
    payload = json.dumps(hashes, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def expected_hashes_path(root: Path) -> Path:
    return root / EXPECTED_HASHES_PATH_REL


def load_expected_hashes(root: Path) -> dict[str, str] | None:
    path = expected_hashes_path(root)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["artifact_hashes"]


def save_expected_hashes(root: Path, hashes: dict[str, str], note: str) -> None:
    path = expected_hashes_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Committed, human-reviewed expected SHA-256 for every model artifact that feeds "
            "a triage prompt. eval/record_cassettes.py refuses to run against a checkout "
            "whose resolved artifacts don't match this file -- see ADR-0059. Regenerate "
            "deliberately (scripts or a one-off) whenever you intend to record against a "
            "genuinely different classifier/predictor/index/conformal store; a mismatch is "
            "meant to stop the run, not be silently reconciled."
        ),
        "note": note,
        "artifact_hashes": hashes,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def diff_against_expected(current: dict[str, str], expected: dict[str, str]) -> list[str]:
    """Human-readable mismatch lines; empty list means a clean match."""
    lines: list[str] = []
    all_keys = sorted(set(current) | set(expected))
    for key in all_keys:
        c = current.get(key, "MISSING FROM CURRENT")
        e = expected.get(key, "MISSING FROM EXPECTED FILE")
        if c != e:
            lines.append(f"  {key}: expected={e[:16]}… actual={c[:16]}…")
    return lines
