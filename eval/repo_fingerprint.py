"""Per-repo artifact fingerprint for the recording checkpoint key.

Until now a checkpoint entry's `artifact_hash` was `combined_hash()` over BOTH repos' artifacts,
so changing a vscode-only artifact (its index, classifier, ...) invalidated every k8s entry too.
A repo's prompts depend only on that repo's own artifacts plus the shared CQR store
(`artifact_fingerprint.artifact_paths_for_repo`), so each entry is now keyed by the combined hash
of ITS repo's artifacts. The startup expected-hash gate (ADR-0059) is untouched and still checks
every artifact of both repos.

`artifact_hash` arguments across the recorder, the launcher and the tests accept either a plain
`str` (one hash for every repo, the previous behaviour; what the older tests pass) or a
`RepoHashes` ({repo: hash}).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import artifact_fingerprint as af

ArtifactHash = str | Mapping[str, str]


class RepoHashes(dict):
    """{repo name: combined hash of that repo's artifacts}; str() lists them for logs/status."""

    def __str__(self) -> str:
        return "; ".join(f"{repo}={h}" for repo, h in sorted(self.items()))


def per_repo_hashes(full_hashes: Mapping[str, str], config: Any = None) -> RepoHashes:
    """Per-repo combined hashes carved out of ONE `compute_artifact_hashes(root, config=...)` pass
    (identical to `compute_artifact_hashes(root, repo=repo, config=config)`; no file is hashed twice)."""
    out = RepoHashes()
    for repo in af.REPO_SLUGS:
        rels = af.artifact_paths_for_repo(repo, config)
        out[repo] = af.combined_hash({rel: full_hashes.get(rel, "MISSING") for rel in rels})
    return out


def hash_for(artifact_hash: str | Mapping[str, str], repo: str | None) -> str | None:
    """The hash an entry of `repo` is keyed by; None when the repo is unknown or not hashed."""
    if isinstance(artifact_hash, str):
        return artifact_hash
    return artifact_hash.get(repo) if repo is not None else None


@lru_cache(maxsize=8)
def _repo_by_issue_cached(path: str, mtime_ns: int) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                issue = json.loads(line)
                out[issue["id"]] = issue["repo"]
    return out


def repo_by_issue(eval_set_path: Path) -> dict[str, str]:
    """{issue_id: repo} from an eval_set.jsonl (re-read when the file changes)."""
    return _repo_by_issue_cached(str(eval_set_path), Path(eval_set_path).stat().st_mtime_ns)


def entry_is_current(
    rec: Mapping[str, Any], model: str, prompt_hash: str, artifact_hash: str | Mapping[str, str],
    repos: Mapping[str, str],
) -> bool:
    """True iff `rec` was recorded under the current model, prompt and ITS repo's artifacts."""
    expected = hash_for(artifact_hash, repos.get(rec.get("issue_id", "")))
    return (
        expected is not None and rec.get("model") == model and rec.get("prompt_hash") == prompt_hash
        and rec.get("artifact_hash") == expected
    )
