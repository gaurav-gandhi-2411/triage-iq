"""Repo-wide pytest hooks.

Installs the tracked-cassette write guard (tools/test_guards/tracked_cassette_guard.py) for every test run
that starts below the repo root, `tests/` and `eval/` alike: no test, by any code path, may write to a
git-tracked cassette directory, and a session that leaves one changed fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "tools" / "test_guards"))

import tracked_cassette_guard as _guard  # noqa: E402

_PROTECTED = _guard.install(_ROOT)
_BEFORE: dict[str, str] = {}


def pytest_sessionstart(session: pytest.Session) -> None:
    global _BEFORE
    _BEFORE = _guard.digest(_PROTECTED)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    changed = _guard.changed(_BEFORE, _guard.digest(_PROTECTED))
    if changed:
        sys.stderr.write(
            "\nTRACKED CASSETTE FILES CHANGED DURING THE TEST SESSION (something wrote outside the "
            "in-process guard, e.g. a subprocess):\n  " + "\n  ".join(changed) + "\n"
        )
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
