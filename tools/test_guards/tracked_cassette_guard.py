"""No test may write to a git-tracked cassette location, by any code path.

Why (2026-10-12): a test's fake recorder built a path with `Path(env["PYTHONPATH"].split(":")[0]).parent`.
On Windows `PYTHONPATH` is `C:\\...;C:\\...`, so the split gave `C`, the path became the RELATIVE
`./eval/cassettes/eval_cassette.json`, and with the repo root as cwd the test overwrote the committed
cassette (6,019 lines became `{"entries": 2}`). Nothing stopped it: `CassettePlayer.allow_record` only guards
writes that go THROUGH `CassettePlayer.set`; this write was a plain `Path.write_text`. It was caught by
`git status`, by habit.

Two layers, both installed from the repo-root `conftest.py`, neither with an opt-out inside tests:

1. A Python audit hook (`sys.addaudithook`, cannot be removed once added) that raises
   `TrackedCassetteWriteBlocked` (a `PermissionError`) BEFORE any file open for writing, any rename, replace,
   remove or rmdir, any copy or mkdir that resolves under a protected directory. Audit events fire for every
   route into the C-level file calls (`open`, `Path.write_text`, `os.open`, `shutil.copy`, `tarfile`,
   `os.replace`, ...), and the path is resolved against the cwd at call time, so a relative path that lands
   in a protected directory is caught however it was built.
2. A session tripwire: sha256 of every file under the protected directories at session start, compared at
   session end. It covers what an in-process hook cannot see (a subprocess, a native extension), and turns
   the pytest exit status into a failure.

Protected = every directory that contains a git-tracked file and has a `cassettes` path component (found via
`git ls-files`; if git is unavailable, every directory named `cassettes` under the repo root).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path


class TrackedCassetteWriteBlocked(PermissionError):
    """Raised when code tries to modify a git-tracked cassette directory during a test run."""


def protected_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    dirs: set[Path] = set()
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True, timeout=60, check=True
        ).stdout
        for rel in filter(None, out.split("\0")):
            parts = rel.split("/")
            if "cassettes" in parts[:-1]:
                i = parts.index("cassettes")
                dirs.add((root.joinpath(*parts[: i + 1])).resolve())
    except (OSError, subprocess.SubprocessError):
        skip = {".git", ".venv", "node_modules", "venv"}
        for dirpath, dirnames, _files in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            if os.path.basename(dirpath) == "cassettes":
                dirs.add(Path(dirpath).resolve())
    return sorted(dirs)


def _norm(p: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(p))))


def _inside(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == pre or path.startswith(pre + os.sep) for pre in prefixes)


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_installed = False


def install(root: Path) -> list[Path]:
    """Install the audit hook once per process. Returns the protected directories."""
    global _installed
    dirs = protected_dirs(root)
    if _installed:
        return dirs
    prefixes = tuple(_norm(d) for d in dirs)
    if not prefixes:
        return dirs

    def hook(event: str, args: tuple) -> None:  # noqa: ANN401
        try:
            if event == "open":
                path, mode, flags = args
                if not isinstance(path, (str, bytes, os.PathLike)):
                    return  # an fd
                wants_write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                    isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
                )
                if wants_write and _inside(_norm(path), prefixes):
                    raise TrackedCassetteWriteBlocked(f"write to tracked cassette path blocked: {_norm(path)}")
            elif event in {"os.remove", "os.rmdir", "os.truncate", "os.mkdir", "shutil.rmtree"}:
                if isinstance(args[0], (str, bytes, os.PathLike)) and _inside(_norm(args[0]), prefixes):
                    raise TrackedCassetteWriteBlocked(f"{event} on tracked cassette path blocked: {_norm(args[0])}")
            elif event in {"os.rename", "shutil.copyfile", "shutil.copy", "shutil.copy2", "shutil.move"}:
                for p in args[:2]:
                    if isinstance(p, (str, bytes, os.PathLike)) and _inside(_norm(p), prefixes):
                        raise TrackedCassetteWriteBlocked(f"{event} touching tracked cassette path blocked: {_norm(p)}")
        except TrackedCassetteWriteBlocked:
            raise
        except Exception:  # noqa: BLE001, S110 -- a probe failure must never break unrelated file access
            return

    sys.addaudithook(hook)
    _installed = True
    return dirs


def digest(dirs: list[Path]) -> dict[str, str]:
    """sha256 of every file under the protected directories (keyed by absolute path)."""
    out: dict[str, str] = {}
    for d in dirs:
        for dirpath, _dn, files in os.walk(d):
            for f in files:
                p = Path(dirpath) / f
                try:
                    out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
                except OSError:
                    out[str(p)] = "unreadable"
    return out


def changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
