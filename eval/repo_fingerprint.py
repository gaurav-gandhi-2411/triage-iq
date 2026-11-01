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
from collections.abc import Callable, Mapping
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


def plan_migration(
    done: Mapping[str, dict], *, model: str, prompt_hash: str, current: RepoHashes, legacy_combined: str,
    repos: Mapping[str, str], current_provenance: Mapping[str, Mapping[str, str]],
    provenance_of: Callable[[str], Mapping[str, str] | None],
) -> tuple[dict[str, dict], dict[str, str]]:
    """Old-key (combined-hash) entries that may be re-keyed per repo, and why the others may not.

    Returns ({old_key: entry}, {issue_id: reason}). An entry is a candidate only if proven:
      * plan-bearing: the cassette entry its synthesis_cache_key points at is stamped with exactly
        the current per-repo artifact hashes of its repo. (The caller must still run checkpoint
        validation on the candidates and migrate only the VALID ones.)
      * plan-less (a permanent failure: no cassette entry, so nothing is stamped): its old key
        equals today's old-style combined hash, i.e. every artifact is unchanged since it was
        recorded, which covers its repo's.
    Entries of another model/prompt, or already keyed per repo, are not considered.
    """
    candidates: dict[str, dict] = {}
    rejected: dict[str, str] = {}
    already = set(current.values())
    for old_key, rec in done.items():
        if rec.get("model") != model or rec.get("prompt_hash") != prompt_hash:
            continue
        if rec.get("artifact_hash") in already:
            continue
        issue_id = rec["issue_id"]
        repo = repos.get(issue_id)
        if repo is None:
            rejected[issue_id] = "issue not in the eval set"
        elif _new_key(rec, current[repo]) in done:
            rejected[issue_id] = "a per-repo entry already exists"
        elif rec.get("plan") is not None:
            stamped = provenance_of(rec["synthesis_cache_key"]) if rec.get("synthesis_cache_key") else None
            if stamped is None:
                rejected[issue_id] = "cassette entry carries no stamped artifact provenance"
            elif dict(stamped) != dict(current_provenance[repo]):
                rejected[issue_id] = f"stamped artifacts differ from the current {repo} artifacts"
            else:
                candidates[old_key] = rec
        elif rec.get("artifact_hash") == legacy_combined:
            candidates[old_key] = rec
        else:
            rejected[issue_id] = "no cassette entry to prove its artifacts and its old key is not the current combined hash"
    return candidates, rejected


def _new_key(rec: Mapping[str, Any], new_hash: str) -> str:
    return f"{rec['issue_id']}::{rec['model']}::{rec['prompt_hash']}::{new_hash}"


def apply_migration(
    done: dict[str, dict], migrated_old_keys: list[str], current: RepoHashes, repos: Mapping[str, str],
) -> int:
    """Re-key the given entries in place under their repo's hash; returns how many."""
    for old_key in migrated_old_keys:
        rec = done.pop(old_key)
        new_hash = current[repos[rec["issue_id"]]]
        done[_new_key(rec, new_hash)] = {**rec, "artifact_hash": new_hash}
    return len(migrated_old_keys)
