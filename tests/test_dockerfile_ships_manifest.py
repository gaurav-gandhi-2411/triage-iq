"""The prod image must carry MANIFEST.sha256 and every artifact it lists.

loader._check_manifest_drift logs "MANIFEST.sha256 not found ... skipping drift check" and returns
when the file is absent, so an image without it passes every deploy gate while the runtime check
silently does nothing (observed on revision triageiq-api-00042-ves, 2026-10-07).
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker" / "Dockerfile.prod"
MANIFEST = ROOT / "data" / "models" / "MANIFEST.sha256"


def _copy_sources() -> list[str]:
    srcs = []
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*COPY\s+(?!--from)(\S+)\s+\S+", line)
        if m:
            srcs.append(m.group(1).rstrip("/"))
    return srcs


def test_dockerfile_copies_manifest() -> None:
    assert "data/models/MANIFEST.sha256" in _copy_sources()


def test_every_manifest_entry_is_copied_into_the_image() -> None:
    srcs = _copy_sources()
    entries = [ln.split("  ", 1)[1].strip() for ln in MANIFEST.read_text().splitlines() if ln.strip()]
    assert entries
    missing = [
        e for e in entries
        if not any(fnmatch.fnmatch(e, s) or e.startswith(s + "/") for s in srcs)
    ]
    assert not missing, f"manifest lists artifacts the prod image never copies: {missing}"
