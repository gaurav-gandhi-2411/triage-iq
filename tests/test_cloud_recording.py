"""Cloud Run recording job: GCS snapshot store, ledger merge, and the job's orchestration (no network, no GCP).

The bucket is an in-memory fake with the two behaviours the code relies on: `list_blobs(prefix=)` and
create-only uploads (`if_generation_match=0`) that fail with a 412 when the object already exists.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "groq_budget"))
sys.path.insert(0, str(ROOT / "scripts" / "cloud_recording"))

import groq_budget as gb  # noqa: E402
import job_entrypoint as je  # noqa: E402
from gcs_state import (  # noqa: E402
    SnapshotStore,
    decode_state,
    encode_state,
    merge_ledgers,
    push_state,
)

M120 = "openai/gpt-oss-120b"


class PreconditionFailed(Exception):
    code = 412


class FakeBlob:
    def __init__(self, bucket: FakeBucket, name: str) -> None:
        self.bucket, self.name = bucket, name

    def upload_from_string(self, data: bytes, content_type: str = "", if_generation_match: int | None = None) -> None:
        if if_generation_match == 0 and self.name in self.bucket.objects:
            raise PreconditionFailed("object exists")
        self.bucket.objects[self.name] = bytes(data)

    def download_as_bytes(self) -> bytes:
        return self.bucket.objects[self.name]


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self, name)

    def list_blobs(self, prefix: str = ""):  # noqa: ANN201
        return [FakeBlob(self, n) for n in sorted(self.objects) if n.startswith(prefix)]


def _entry(ts: datetime, tokens: int, project: str = "benchmark", source: str = "t") -> dict:
    return {"ts": ts.isoformat(), "model": M120, "project": project, "tokens": tokens, "source": source}


NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------------- snapshot store
def test_snapshots_are_append_only_and_latest_is_the_highest_sequence() -> None:
    s = SnapshotStore(FakeBucket(), "recorder/ledger")
    assert s.latest() is None
    assert s.push(-1, b"one") and s.push(0, b"two") and s.push(1, b"three")
    assert s.latest() == (2, b"three")


def test_a_lost_race_is_reported_not_overwritten() -> None:
    b = FakeBucket()
    a, c = SnapshotStore(b, "p"), SnapshotStore(b, "p")
    assert a.push(-1, b"first")
    assert not c.push(-1, b"second")  # same sequence number: create-only refuses
    assert a.latest() == (0, b"first")  # nothing was overwritten


def test_other_exceptions_are_not_swallowed() -> None:
    class Boom:
        def blob(self, name: str):  # noqa: ANN201
            class B:
                def upload_from_string(self, *a, **k):  # noqa: ANN002, ANN003
                    raise RuntimeError("network down")

            return B()

        def list_blobs(self, prefix: str = ""):  # noqa: ANN201
            return []

    with pytest.raises(RuntimeError):
        SnapshotStore(Boom(), "p").push(-1, b"x")


def test_merge_ledgers_unions_and_dedupes_oldest_first() -> None:
    e1, e2, e3 = _entry(NOW, 100), _entry(NOW + timedelta(minutes=1), 200), _entry(NOW - timedelta(hours=1), 50)
    out = merge_ledgers({"entries": [e1, e2]}, {"entries": [e2, e3]})
    assert [e["tokens"] for e in out["entries"]] == [50, 100, 200]


def test_push_state_merges_remote_spend_and_retries_after_a_lost_race() -> None:
    b = FakeBucket()
    store = SnapshotStore(b, "ledger")
    remote_entry, mine = _entry(NOW, 1_000, "benchmark"), _entry(NOW, 6_000, "triageiq-recorder", "rec")
    store.push(-1, encode_state({"entries": [remote_entry]}, None))

    class RacyStore(SnapshotStore):
        raced = False

        def push(self, base_seq: int, data: bytes) -> bool:
            if not self.raced:  # someone else publishes first (their snapshot includes what was there)
                self.raced = True
                prior, _ = decode_state(self.latest()[1])  # type: ignore[index]
                super().push(base_seq, encode_state(merge_ledgers(prior, {"entries": [_entry(NOW, 7, "x")]}), None))
            return super().push(base_seq, data)

    assert push_state(RacyStore(b, "ledger"), {"entries": [mine]}, None, sleep=lambda s: None)
    ledger, rec = decode_state(SnapshotStore(b, "ledger").latest()[1])  # type: ignore[index]
    assert sorted(e["tokens"] for e in ledger["entries"]) == [7, 1_000, 6_000] and rec is None


# ---------------------------------------------------------------------------------------- the job
def _cfg(tmp_path: Path, bucket: FakeBucket, runner, owner: str = "exec-1", now: float | None = None) -> je.JobConfig:  # noqa: ANN001
    root = tmp_path / "app"
    (root / "eval" / "cassettes").mkdir(parents=True, exist_ok=True)
    cass = root / "eval" / "cassettes" / "eval_cassette.json"
    if not cass.exists():
        cass.write_text('{"entries": 1}', encoding="utf-8")
    return je.JobConfig(
        bucket=bucket,
        root=root,
        ledger_dir=tmp_path / f"ledger-{owner}",
        timeout_s=600,
        owner=owner,
        now=(lambda: now) if now is not None else (lambda: NOW.timestamp()),
        runner=runner,
    )


def _recorder(tokens: int = 6_000, rc: int = 0, change_cassette: bool = True):  # noqa: ANN202
    calls: list[dict] = []

    def run(cmd: list[str], env: dict, timeout_s: int) -> tuple[int, str]:
        calls.append({"cmd": cmd, "env": env, "timeout": timeout_s})
        gb.Ledger(Path(env["GROQ_BUDGET_DIR"])).record(gb.RECORDER, M120, tokens, "record_cassettes:test-1")
        if change_cassette:
            p = Path(env["PYTHONPATH"].split(os.pathsep)[0]).parent / "eval" / "cassettes" / "eval_cassette.json"
            p.write_text(json.dumps({"entries": 2}), encoding="utf-8")
        return rc, "tail"

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_a_clean_run_records_publishes_ledger_and_cassette_and_releases_the_flag(tmp_path: Path) -> None:
    b = FakeBucket()
    run = _recorder()
    assert je.run_job(_cfg(tmp_path, b, run)) == 0
    assert len(run.calls) == 1
    assert run.calls[0]["cmd"][1:] == ["eval/record_cassettes.py", "--mode", "synthesis"]
    assert run.calls[0]["env"]["GROQ_BUDGET_RECORDING_PARENT"] == "1"
    ledger, rec = decode_state(SnapshotStore(b, "recorder/ledger").latest()[1])  # type: ignore[index]
    assert rec is None  # flag released
    assert [e["tokens"] for e in ledger["entries"]] == [6_000]
    assert SnapshotStore(b, "recorder/cassette").latest() is not None  # new cassette bundle published


def test_spend_accumulates_across_runs_through_the_bucket(tmp_path: Path) -> None:
    b = FakeBucket()
    assert je.run_job(_cfg(tmp_path, b, _recorder(6_000), "exec-1")) == 0
    assert je.run_job(_cfg(tmp_path, b, _recorder(5_500), "exec-2")) == 0
    ledger, _ = decode_state(SnapshotStore(b, "recorder/ledger").latest()[1])  # type: ignore[index]
    assert sorted(e["tokens"] for e in ledger["entries"]) == [5_500, 6_000]


def test_a_live_flag_held_by_another_execution_means_busy_and_nothing_runs(tmp_path: Path) -> None:
    b = FakeBucket()
    store = SnapshotStore(b, "recorder/ledger")
    other = {"project": gb.RECORDER, "models": [M120], "pid": 0, "owner": "someone-else",
             "since_ts": NOW.timestamp(), "expires_ts": NOW.timestamp() + 3_000}
    store.push(-1, encode_state({"entries": []}, other))
    run = _recorder()
    assert je.run_job(_cfg(tmp_path, b, run, "exec-2")) == 0
    assert run.calls == []
    _, rec = decode_state(store.latest()[1])  # type: ignore[index]
    assert rec["owner"] == "someone-else"  # the busy execution did not touch their flag


def test_an_expired_flag_does_not_block(tmp_path: Path) -> None:
    b = FakeBucket()
    store = SnapshotStore(b, "recorder/ledger")
    dead = {"project": gb.RECORDER, "models": [M120], "pid": 0, "owner": "crashed",
            "since_ts": NOW.timestamp() - 9_000, "expires_ts": NOW.timestamp() - 100}
    store.push(-1, encode_state({"entries": []}, dead))
    run = _recorder()
    assert je.run_job(_cfg(tmp_path, b, run, "exec-3")) == 0
    assert len(run.calls) == 1


def test_out_of_budget_records_nothing_but_still_releases_the_flag(tmp_path: Path) -> None:
    b = FakeBucket()
    full = {"entries": [_entry(NOW - timedelta(hours=1), 160_000, "benchmark")]}
    SnapshotStore(b, "recorder/ledger").push(-1, encode_state(full, None))
    run = _recorder()
    assert je.run_job(_cfg(tmp_path, b, run)) == 0
    assert run.calls == []
    _, rec = decode_state(SnapshotStore(b, "recorder/ledger").latest()[1])  # type: ignore[index]
    assert rec is None


def test_benchmark_spend_in_the_shared_ledger_limits_the_recorder(tmp_path: Path) -> None:
    b = FakeBucket()
    # 156K already spent by other projects in the last 24 h; the 160K allocatable leaves room for no call
    SnapshotStore(b, "recorder/ledger").push(-1, encode_state({"entries": [_entry(NOW - timedelta(hours=2), 156_000)]}, None))
    run = _recorder()
    je.run_job(_cfg(tmp_path, b, run))
    assert run.calls == []


def test_a_crashing_recorder_still_publishes_the_spend_and_fails_the_job(tmp_path: Path) -> None:
    b = FakeBucket()
    assert je.run_job(_cfg(tmp_path, b, _recorder(4_000, rc=1, change_cassette=False))) == 1
    ledger, rec = decode_state(SnapshotStore(b, "recorder/ledger").latest()[1])  # type: ignore[index]
    assert [e["tokens"] for e in ledger["entries"]] == [4_000] and rec is None


def test_budget_stop_and_timeout_exit_codes_are_clean(tmp_path: Path) -> None:
    for rc in (3, 124):
        b = FakeBucket()
        assert je.run_job(_cfg(tmp_path / str(rc), b, _recorder(rc=rc))) == 0


def test_a_bundle_from_a_different_cassette_lineage_is_ignored(tmp_path: Path) -> None:
    b = FakeBucket()
    assert je.run_job(_cfg(tmp_path / "a", b, _recorder(), "exec-1")) == 0  # publishes bundle for lineage A
    # a new image whose baked cassette differs (someone committed a newer cassette): different lineage
    cfg = _cfg(tmp_path / "b", b, _recorder(change_cassette=False), "exec-2")
    (cfg.cassette_dir / "eval_cassette.json").write_text('{"entries": 99}', encoding="utf-8")
    je.run_job(cfg)
    assert json.loads((cfg.cassette_dir / "eval_cassette.json").read_text(encoding="utf-8")) == {"entries": 99}


def test_main_refuses_to_start_without_bucket_or_key(monkeypatch: pytest.MonkeyPatch, capsys) -> None:  # noqa: ANN001
    monkeypatch.delenv("RECORD_GCS_BUCKET", raising=False)
    assert je.main() == 1
    monkeypatch.setenv("RECORD_GCS_BUCKET", "b")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert je.main() == 1
    assert "not set" in capsys.readouterr().err


# ------------------------------------------------------------------------------------ pull_bundle
def test_pull_bundle_restores_only_a_bundle_from_the_same_lineage(tmp_path: Path) -> None:
    import pull_bundle as pb

    b = FakeBucket()
    assert je.run_job(_cfg(tmp_path / "job", b, _recorder(), "exec-1")) == 0  # publishes a bundle
    mine = tmp_path / "laptop" / "eval" / "cassettes"
    mine.mkdir(parents=True)
    (mine / "eval_cassette.json").write_text('{"entries": 1}', encoding="utf-8")  # same baked lineage
    assert "restored" in pb.pull(b, "recorder", mine)
    assert json.loads((mine / "eval_cassette.json").read_text(encoding="utf-8")) == {"entries": 2}
    (mine / "eval_cassette.json").write_text('{"entries": 77}', encoding="utf-8")  # a newer committed cassette
    assert "ignored" in pb.pull(b, "recorder", mine)
    assert json.loads((mine / "eval_cassette.json").read_text(encoding="utf-8")) == {"entries": 77}
    assert "no cassette bundle" in pb.pull(FakeBucket(), "recorder", mine)
