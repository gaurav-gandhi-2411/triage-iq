"""Every pin in requirements.lock must satisfy the matching constraint in requirements.txt.

Fast, docker-free subset of scripts/check_lock_in_sync.py. It catches the exact failure that
shipped silently for weeks (2026-08..10): Dependabot raised sentence-transformers to >=5.7.0
and groq to >=1.6.0 in requirements.txt while the lock kept pinning 2.7.0 / 1.2.0. The full
regen check (docker, ~10 min) runs in CI only when those files change; this runs every time.
"""

from __future__ import annotations

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent


def _requirements(path: Path) -> list[Requirement]:
    reqs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith(("-", "--")):
            reqs.append(Requirement(line))
    return reqs


def _lock_pins() -> dict[str, Version]:
    pins: dict[str, Version] = {}
    for raw in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^\s;#]+)", raw.strip())
        if m:
            pins[str(canonicalize_name(m.group(1)))] = Version(m.group(2))
    return pins


def test_lock_pins_satisfy_requirements_txt() -> None:
    pins = _lock_pins()
    violations = []
    for req in _requirements(ROOT / "requirements.txt"):
        name = str(canonicalize_name(req.name))
        if name not in pins:
            # Every constrained package must appear in the lock; a missing pin is drift too.
            violations.append(f"{req.name}: constrained in requirements.txt but not pinned in lock")
        elif not req.specifier.contains(pins[name], prereleases=True):
            violations.append(
                f"{req.name}: lock pins {pins[name]} but requirements.txt wants {req.specifier}"
            )
    assert not violations, (
        "requirements.lock has drifted from requirements.txt (regenerate it with the command in "
        "its header, in a Linux container; do not hand-edit):\n  " + "\n  ".join(violations)
    )


def test_drift_detection_catches_a_raised_floor() -> None:
    """The check must fail on the original incident shape, or it protects nothing."""
    req = Requirement("groq>=1.6.0")
    assert not req.specifier.contains(Version("1.2.0"), prereleases=True)
    assert Requirement("groq>=1.2.0,<1.6").specifier.contains(Version("1.2.0"), prereleases=True)
