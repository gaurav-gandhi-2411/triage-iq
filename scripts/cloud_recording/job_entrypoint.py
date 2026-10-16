"""Cloud Run job entrypoint: run the Groq synthesis recorder in GCP, with all state in GCS.

One execution:
  1. pull the shared Groq budget ledger (latest snapshot under gs://<bucket>/<prefix>/ledger/) into
     $GROQ_BUDGET_DIR, so the recorder's budget gate sees what every other project spent;
  2. claim the recording-active flag by publishing the next snapshot (create-only, so two executions racing
     for it cannot both win); if another execution holds an unexpired flag, exit 0 as "busy";
  3. restore the recorder's cassette + checkpoint from the latest bundle, but only if the bundle extends the
     same cassette lineage that was baked into this image (otherwise a newer committed cassette would be
     clobbered by an older bundle);
  4. check the shared budget (160K allocatable of the 200K/day pool, 40K production reserve never touched);
     out of budget -> record nothing;
  5. run `eval/record_cassettes.py --mode synthesis` (bounded by RECORD_TIMEOUT_S);
  6. whatever happened, publish the merged ledger (the spend is real even if the recorder crashed), publish a
     new cassette bundle if the cassette changed, and release the flag.

Nothing is ever overwritten or deleted in GCS (snapshots are new objects) and nothing is pushed to GitHub:
a person pulls the latest bundle and commits it through a normal PR.

Env: RECORD_GCS_BUCKET (required), RECORD_GCS_PREFIX (default "recorder"), GROQ_API_KEY (Secret Manager),
RECORD_TIMEOUT_S (default 3000), GROQ_BUDGET_DIR (default /tmp/groq_budget), CLOUD_RUN_EXECUTION (owner id).
Exit codes: 0 done / out of budget / busy, 1 recorder or infrastructure failure.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "groq_budget"))

import groq_budget as gb  # noqa: E402
from gcs_state import (  # noqa: E402
    SnapshotStore,
    decode_state,
    encode_state,
    open_bucket,
    push_state,
)

PROJECT = gb.RECORDER
MODEL = "openai/gpt-oss-120b"
CASSETTE_FILES = ("eval_cassette.json", "recording_checkpoint.json")
NEXT_CALL_TOKENS = 5_000
FLAG_MARGIN_S = 600


@dataclass
class JobConfig:
    bucket: Any
    prefix: str = "recorder"
    root: Path = ROOT
    ledger_dir: Path = Path("/tmp/groq_budget")
    timeout_s: int = 3000
    owner: str = "local"
    now: Callable[[], float] = time.time
    runner: Callable[[list[str], dict, int], tuple[int, str]] = field(default=None)  # type: ignore[assignment]

    @property
    def cassette_dir(self) -> Path:
        return self.root / "eval" / "cassettes"


def default_runner(cmd: list[str], env: dict, timeout_s: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)  # noqa: S603
        return p.returncode, (p.stdout or "")[-4000:] + (p.stderr or "")[-2000:]
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        return 124, f"timeout after {timeout_s}s\n{out[-3000:]}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


# --------------------------------------------------------------------------------- cassette bundle
def pack_bundle(cassette_dir: Path, lineage: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        manifest = {"lineage": lineage, "files": {}}
        for name in CASSETTE_FILES:
            p = cassette_dir / name
            if p.exists():
                manifest["files"][name] = _sha(p)
                tar.add(p, arcname=name)
        raw = json.dumps(manifest, sort_keys=True).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


def unpack_bundle(data: bytes, cassette_dir: Path, lineage: str) -> str:
    """Restore a bundle if it extends this image's cassette lineage. Returns what happened."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read().decode())  # type: ignore[union-attr]
        if manifest.get("lineage") != lineage:
            return "ignored: bundle is from a different cassette lineage (the image has a newer committed cassette)"
        for name in manifest["files"]:
            member = tar.extractfile(name)
            assert member is not None
            (cassette_dir / name).write_bytes(member.read())
    return "restored"


# ------------------------------------------------------------------------------------------ the run
def _flag(cfg: JobConfig) -> dict:
    t = cfg.now()
    return {
        "project": PROJECT,
        "models": [MODEL],
        "pid": 0,  # remote execution: liveness is the TTL, not a local pid
        "owner": cfg.owner,
        "since_ts": t,
        "expires_ts": t + cfg.timeout_s + FLAG_MARGIN_S,
    }


def claim(cfg: JobConfig, store: SnapshotStore) -> tuple[bool, dict]:
    """Publish a snapshot carrying our recording flag. (claimed, current ledger doc)."""
    for _ in range(6):
        snap = store.latest()
        base, raw = snap if snap else (-1, None)
        ledger, rec = decode_state(raw)
        if rec and rec.get("owner") != cfg.owner and float(rec.get("expires_ts", 0)) > cfg.now():
            return False, ledger
        if store.push(base, encode_state(ledger, _flag(cfg))):
            return True, ledger
    return False, ledger


def run_job(cfg: JobConfig) -> int:
    runner = cfg.runner or default_runner
    ledger_store = SnapshotStore(cfg.bucket, f"{cfg.prefix}/ledger")
    cass_store = SnapshotStore(cfg.bucket, f"{cfg.prefix}/cassette")

    ok, remote_ledger = claim(cfg, ledger_store)
    if not ok:
        print(json.dumps({"event": "busy", "detail": "another execution holds the recording flag"}))
        return 0

    code, summary = 1, {}
    before, restored, before_sha = 0, "not reached", ""
    lineage = _sha(cfg.cassette_dir / "eval_cassette.json")
    try:
        cfg.ledger_dir.mkdir(parents=True, exist_ok=True)
        (cfg.ledger_dir / "ledger.json").write_text(json.dumps(remote_ledger), encoding="utf-8")
        led = gb.Ledger(cfg.ledger_dir)
        before = led.spent(MODEL)

        snap = cass_store.latest()
        restored = "no bundle yet (starting from the cassette baked into the image)"
        if snap:
            restored = unpack_bundle(snap[1], cfg.cassette_dir, lineage)
        before_sha = _sha(cfg.cassette_dir / "eval_cassette.json")

        # The job's own clock (cfg.now), not the ledger's wall clock: the budget decision must be made at the
        # same instant the flag and the entries were stamped with (and tests that freeze cfg.now stay valid).
        d = led.check(PROJECT, MODEL, NEXT_CALL_TOKENS, now=datetime.fromtimestamp(cfg.now(), tz=timezone.utc))
        if not d.allowed:
            summary = {"event": "skipped_budget", "reason": d.reason, "retry_after_s": d.retry_after_s}
            code = 0
        else:
            env = dict(os.environ)
            env.update(
                GROQ_BUDGET_DIR=str(cfg.ledger_dir),
                GROQ_BUDGET_RECORDING_PARENT="1",  # this job already holds the (remote) flag
                TRIAGE_PROMPT_INCLUDE_ATTRIBUTION="1",
                PYTHONPATH=f"{cfg.root / 'src'}{os.pathsep}{cfg.root / 'eval'}",
            )
            rc, tail = runner([sys.executable, "eval/record_cassettes.py", "--mode", "synthesis"], env, cfg.timeout_s)
            # record_cassettes: 0 = done or clean TPD stop, 3 = budget-reserve stop; anything else is a failure.
            code = 0 if rc in (0, 3, 124) else 1
            summary = {"event": "recorded", "recorder_exit": rc, "tail": tail[-600:]}
    finally:
        # The spend is real even when the recorder crashed, so always publish the ledger and release the flag.
        ledger_file = cfg.ledger_dir / "ledger.json"
        local = json.loads(ledger_file.read_text(encoding="utf-8")) if ledger_file.exists() else remote_ledger
        published = push_state(ledger_store, local, None)
        after_sha = _sha(cfg.cassette_dir / "eval_cassette.json")
        cass_pushed = False
        if before_sha and after_sha and after_sha != before_sha:
            latest = cass_store.latest()
            cass_pushed = cass_store.push(latest[0] if latest else -1, pack_bundle(cfg.cassette_dir, lineage))
        summary.update(
            ledger_published=published,
            cassette_bundle_published=cass_pushed,
            bundle_restore=restored,
            tokens_spent_this_run=gb.Ledger(cfg.ledger_dir).spent(MODEL) - before,
        )
        print(json.dumps(summary))
    if not published:
        return 1
    return code


def main() -> int:
    bucket_name = os.environ.get("RECORD_GCS_BUCKET")
    if not bucket_name:
        print("RECORD_GCS_BUCKET is not set", file=sys.stderr)
        return 1
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set (Secret Manager env var missing)", file=sys.stderr)
        return 1
    cfg = JobConfig(
        bucket=open_bucket(bucket_name),
        prefix=os.environ.get("RECORD_GCS_PREFIX", "recorder"),
        ledger_dir=Path(os.environ.get("GROQ_BUDGET_DIR", "/tmp/groq_budget")),
        timeout_s=int(os.environ.get("RECORD_TIMEOUT_S", "3000")),
        owner=os.environ.get("CLOUD_RUN_EXECUTION", f"manual-{int(time.time())}"),
    )
    return run_job(cfg)


if __name__ == "__main__":
    sys.exit(main())
