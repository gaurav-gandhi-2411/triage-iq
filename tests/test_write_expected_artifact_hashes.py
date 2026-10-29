"""scripts/write_expected_artifact_hashes.py: manifest cross-check and config-aware regeneration.

Hermetic: a throwaway checkout under tmp_path (tiny files, its own MANIFEST), no real model.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "eval", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import artifact_fingerprint as af  # noqa: E402
import write_expected_artifact_hashes as w  # noqa: E402

GOOD = {"vscode_classifier": "family_v2", "k8s_index": "v2", "resolution_evidence": True}


def _fake_checkout(tmp_path: Path, config: dict | None) -> tuple[Path, Path]:
    """Root with every default and (if config) config artifact, a matching MANIFEST, and a set dir."""
    set_dir = tmp_path / "eval" / "sets" / "s"
    (set_dir / "cassettes").mkdir(parents=True)
    (set_dir / "eval_set.jsonl").write_text("", encoding="utf-8")
    if config:
        (set_dir / "RECORDING_CONFIG.json").write_text(json.dumps(config), encoding="utf-8")
    from recording_config import parse_config

    cfg = parse_config(config) if config else None
    rels: set[str] = set()
    for repo in af.REPO_SLUGS:
        rels.update(af.artifact_paths_for_repo(repo, cfg))
    lines = []
    for rel in sorted(rels):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(rel.encode())
        lines.append(f"{hashlib.sha256(rel.encode()).hexdigest()}  {rel}")
    (tmp_path / w.MANIFEST_REL).write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return tmp_path, set_dir


def test_config_adds_exactly_the_four_extra_entries(tmp_path: Path) -> None:
    root, set_dir = _fake_checkout(tmp_path, GOOD)
    hashes, _rows, problems, _cfg = w.build(root, set_dir)
    assert problems == []
    assert len(hashes) == 15  # 11 default + family_v2 bundle + k8s v2 index (2) + evidence parquet
    default_root, default_set = _fake_checkout(tmp_path / "d", None)
    assert len(w.build(default_root, default_set)[0]) == 11


def test_mismatch_missing_and_unlisted_artifacts_block_the_write(tmp_path: Path) -> None:
    root, set_dir = _fake_checkout(tmp_path, GOOD)
    victim = next(iter(sorted(af.artifact_paths_for_repo("microsoft/vscode"))))
    (root / victim).write_bytes(b"changed")
    manifest = root / w.MANIFEST_REL
    kept = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if "evidence" not in ln]
    manifest.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert w.main(["--set-dir", str(set_dir), "--root", str(root)]) == 1
    assert not af.expected_hashes_path(root).exists()
    _hashes, _rows, problems, _ = w.build(root, set_dir)
    assert any(p.startswith("MISMATCH") for p in problems)
    assert any(p.startswith("NOT-IN-MANIFEST") for p in problems)


def test_writes_the_file_the_recorder_checks_and_keeps_the_running_note(tmp_path: Path) -> None:
    root, set_dir = _fake_checkout(tmp_path, GOOD)
    af.save_expected_hashes(root, {"x": "y"}, note="old note")
    assert w.main(["--set-dir", str(set_dir), "--root", str(root), "--note", "new note"]) == 0
    saved = json.loads(af.expected_hashes_path(root).read_text(encoding="utf-8"))
    assert saved["note"] == "old note | new note"
    # what the recorder compares: current fingerprint under the set config == expected, no diff
    current = af.compute_artifact_hashes(
        root, config=w.load_recording_config(root, set_dir / "cassettes")
    )
    assert af.diff_against_expected(current, af.load_expected_hashes(root) or {}) == []


def test_check_mode_writes_nothing(tmp_path: Path) -> None:
    root, set_dir = _fake_checkout(tmp_path, GOOD)
    assert w.main(["--set-dir", str(set_dir), "--root", str(root), "--check"]) == 0
    assert not af.expected_hashes_path(root).exists()
