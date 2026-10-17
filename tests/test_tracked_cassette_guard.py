"""The tracked-cassette guard fires on the bug of 2026-10-12 and on every other route to the same files.

The exact bug: a test's fake recorder derived the cassette path from `PYTHONPATH` with
`split(":")[0]`. On Windows `PYTHONPATH` is `C:\\a\\src;C:\\a\\eval`, so the first piece is `C`, the path
becomes the RELATIVE `eval/cassettes/eval_cassette.json`, and with the repo root as the working directory
that is the committed cassette. The guard has to stop that write before it happens, whoever builds the path.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "test_guards"))
sys.path.insert(0, str(ROOT / "scripts" / "cloud_recording"))

import tracked_cassette_guard as guard  # noqa: E402

TRACKED = ROOT / "eval" / "cassettes" / "eval_cassette.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def committed_file_is_untouched():  # noqa: ANN201
    before = _sha(TRACKED)
    yield
    assert _sha(TRACKED) == before, "a guard test modified the committed cassette"


def test_the_guard_is_installed_and_protects_the_tracked_cassette_directory() -> None:
    dirs = guard.protected_dirs(ROOT)
    assert (ROOT / "eval" / "cassettes").resolve() in dirs


def test_the_exact_bug_of_2026_10_12_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    # What the fake recorder saw on Windows, and the expression that produced the wrong path.
    pythonpath = r"C:\work\app\src;C:\work\app\eval"
    root_guess = Path(pythonpath.split(":")[0]).parent  # "C" -> Path("C").parent == Path(".")
    assert root_guess == Path(".")
    target = root_guess / "eval" / "cassettes" / "eval_cassette.json"
    assert not target.is_absolute()
    monkeypatch.chdir(ROOT)  # how pytest was invoked from the repo root
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        target.write_text('{"entries": 2}', encoding="utf-8")


@pytest.mark.parametrize("mode", ["w", "a", "x", "r+", "wb", "ab"])
def test_every_write_mode_is_blocked_and_reads_are_not(mode: str) -> None:
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        open(TRACKED, mode)  # noqa: SIM115, PTH123
    assert TRACKED.read_text(encoding="utf-8")  # reading is fine
    assert TRACKED.stat().st_size > 1000


def test_low_level_and_library_routes_are_blocked_too(tmp_path: Path) -> None:
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        os.open(TRACKED, os.O_WRONLY)
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        TRACKED.write_bytes(b"x")
    src = tmp_path / "x.json"
    src.write_text("{}", encoding="utf-8")
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        shutil.copyfile(src, TRACKED)
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        shutil.copy(src, TRACKED.parent)  # copy INTO the directory creates a file there
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        os.replace(src, TRACKED)
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        os.rename(src, TRACKED.parent / "new.json")
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        TRACKED.unlink()
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        (TRACKED.parent / "stray.json").write_text("x", encoding="utf-8")  # a NEW file is also pollution


def test_the_cloud_job_cannot_unpack_a_bundle_into_the_real_cassette_directory() -> None:
    import job_entrypoint as je

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        manifest = b'{"lineage": "x", "files": {"eval_cassette.json": "y"}}'
        for name, data in (("eval_cassette.json", b"{}"), ("manifest.json", manifest)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        je.unpack_bundle(buf.getvalue(), ROOT / "eval" / "cassettes", "x")


def test_a_job_configured_with_the_repo_root_cannot_run_the_recorder_path(tmp_path: Path) -> None:
    import job_entrypoint as je

    def runner(cmd: list[str], env: dict, timeout_s: int) -> tuple[int, str]:
        (je.ROOT / "eval" / "cassettes" / "eval_cassette.json").write_text("{}", encoding="utf-8")
        return 0, ""

    class Empty:
        def blob(self, name: str):  # noqa: ANN201
            raise AssertionError

        def list_blobs(self, prefix: str = ""):  # noqa: ANN201
            return []

    class Bucket(Empty):
        objects: dict = {}

        def blob(self, name: str):  # noqa: ANN201
            outer = self

            class B:
                def upload_from_string(self, data, content_type="", if_generation_match=None):  # noqa: ANN001
                    outer.objects[name] = data

                def download_as_bytes(self):  # noqa: ANN201
                    return outer.objects[name]

            return B()

    cfg = je.JobConfig(bucket=Bucket(), root=je.ROOT, ledger_dir=tmp_path / "l", timeout_s=60, owner="t", runner=runner)
    with pytest.raises(guard.TrackedCassetteWriteBlocked):
        je.run_job(cfg)


def test_locations_that_are_not_tracked_cassette_directories_stay_writable(tmp_path: Path) -> None:
    (tmp_path / "cassettes").mkdir()
    p = tmp_path / "cassettes" / "scratch.json"  # a tmp dir that merely has the same name
    p.write_text("{}", encoding="utf-8")
    p.unlink()
    (ROOT / "reports").is_dir()


def test_tripwire_notices_a_change_the_in_process_hook_cannot_see(tmp_path: Path) -> None:
    d = tmp_path / "cassettes"
    d.mkdir()
    (d / "a.json").write_text("1", encoding="utf-8")
    before = guard.digest([d])
    assert guard.changed(before, guard.digest([d])) == []
    (d / "a.json").write_text("2", encoding="utf-8")  # as a subprocess would
    (d / "b.json").write_text("3", encoding="utf-8")
    assert [Path(p).name for p in guard.changed(before, guard.digest([d]))] == ["a.json", "b.json"]
