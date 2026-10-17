"""Fail if requirements.lock is not what a no-upgrade pip-compile of requirements.txt produces.

Why this exists: Dependabot raised two floors in requirements.txt (sentence-transformers
>=5.7.0, groq >=1.6.0) without regenerating requirements.lock, so the lock that production
installs silently diverged from requirements.txt for weeks, and any later scoped regen would
have dragged both upgrades in as a side effect. This check regenerates the lock the way its
header documents (Linux container, same pip-compile flags) and fails on any non-comment diff.

pip-compile keeps existing pins that still satisfy requirements.txt, so a clean tree
regenerates identically; a floor raised past the pin forces a different version and shows up
as a diff here.

Usage: python scripts/check_lock_in_sync.py [--image python:3.11-slim]
Requires docker. Exit 0 = in sync, 1 = drift (diff printed), 2 = tooling failure.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess  # noqa: S404 -- fixed argv, no shell, no untrusted input
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Must match the command in requirements.lock's own header.
COMPILE = (
    "pip install -q pip-tools && pip-compile requirements.txt "
    "--extra-index-url https://download.pytorch.org/whl/cpu "
    "--output-file=requirements.lock --no-header --no-annotate -q"
)


def normalize(text: str) -> list[str]:
    """Drop comments and blank lines: only pins and index options are compared."""
    return [
        ln.rstrip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="python:3.11-slim")
    args = parser.parse_args()
    if shutil.which("docker") is None:
        print("docker not found on PATH", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ("requirements.txt", "requirements.lock"):
            shutil.copy(REPO / name, work / name)
        # S603/S607: argv list with a fixed docker binary; the only variable input is --image.
        proc = subprocess.run(  # noqa: S603
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{work}:/w",
                "-w",
                "/w",
                args.image,
                "bash",
                "-c",
                COMPILE,
            ],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            print(
                f"pip-compile failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}",
                file=sys.stderr,
            )
            return 2
        old = normalize((REPO / "requirements.lock").read_text(encoding="utf-8"))
        new = normalize((work / "requirements.lock").read_text(encoding="utf-8"))
    if old == new:
        print(f"requirements.lock is in sync with requirements.txt ({len(old)} non-comment lines).")
        return 0
    print(
        "requirements.lock is OUT OF SYNC with requirements.txt. Regenerate it with the command in"
    )
    print("its header (Linux container) and review the diff; do not hand-edit.\n")
    sys.stdout.writelines(
        difflib.unified_diff(
            [f"{ln}\n" for ln in old],
            [f"{ln}\n" for ln in new],
            "committed lock",
            "regenerated lock",
        )
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
