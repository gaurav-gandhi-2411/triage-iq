"""Append-only snapshots in a GCS bucket: the shared Groq ledger and the recorder state live there, so a
Cloud Run job (or any machine) can pick up exactly where the last run stopped.

Why append-only: the only IAM this needs is create + read on one prefix. A snapshot is a NEW object
`<prefix>/<seq:08d>.bin` written with `if_generation_match=0` ("create only if it does not exist"), so two
writers racing for the same sequence number cannot both win and nothing is ever overwritten or deleted.
The latest snapshot is the highest sequence number under the prefix.

`google-cloud-storage` is imported only by `open_bucket`; everything else takes any bucket-like object
(`.blob(name)`, `.list_blobs(prefix=)`), which is how the tests use an in-memory fake.
"""

from __future__ import annotations

import json
import time
from typing import Any

SEQ_WIDTH = 8


class SnapshotStore:
    def __init__(self, bucket: Any, prefix: str) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _name(self, seq: int) -> str:
        return f"{self.prefix}/{seq:0{SEQ_WIDTH}d}.bin"

    def latest(self) -> tuple[int, bytes] | None:
        """(sequence, bytes) of the newest snapshot, or None when the prefix is empty."""
        best = -1
        for blob in self.bucket.list_blobs(prefix=self.prefix + "/"):
            stem = blob.name.rsplit("/", 1)[-1]
            if stem.endswith(".bin") and stem[:-4].isdigit():
                best = max(best, int(stem[:-4]))
        if best < 0:
            return None
        return best, self.bucket.blob(self._name(best)).download_as_bytes()

    def push(self, base_seq: int, data: bytes) -> bool:
        """Write snapshot base_seq+1. False when someone else already wrote it (re-read, merge, retry)."""
        blob = self.bucket.blob(self._name(base_seq + 1))
        try:
            blob.upload_from_string(data, content_type="application/octet-stream", if_generation_match=0)
        except Exception as exc:  # noqa: BLE001 -- PreconditionFailed (412) from google.api_core, or the fake's
            if getattr(exc, "code", None) == 412 or type(exc).__name__ == "PreconditionFailed":
                return False
            raise
        return True


def open_bucket(name: str) -> Any:
    from google.cloud import storage  # lazy: only the real job needs it

    return storage.Client().bucket(name)


# ----------------------------------------------------------------------------------- ledger payload
def _key(e: dict) -> tuple:
    return (e.get("ts"), e.get("model"), e.get("project"), e.get("tokens"), e.get("source"))


def merge_ledgers(a: dict, b: dict) -> dict:
    """Union of two ledger documents ({"entries": [...]}), de-duplicated, oldest first."""
    seen: dict[tuple, dict] = {}
    for e in list(a.get("entries", [])) + list(b.get("entries", [])):
        seen.setdefault(_key(e), e)
    return {"version": 2, "entries": sorted(seen.values(), key=lambda e: e["ts"])}


def encode_state(ledger: dict, recording: dict | None) -> bytes:
    return json.dumps({"ledger": ledger, "recording": recording}, sort_keys=True).encode()


def decode_state(raw: bytes | None) -> tuple[dict, dict | None]:
    if raw is None:
        return {"version": 2, "entries": []}, None
    doc = json.loads(raw.decode())
    return doc.get("ledger", {"version": 2, "entries": []}), doc.get("recording")


def push_state(
    store: SnapshotStore,
    ledger: dict,
    recording: dict | None,
    *,
    keep_remote_recording: bool = False,
    retries: int = 6,
    sleep: Any = time.sleep,
) -> bool:
    """Merge `ledger` into the latest remote ledger and publish it, retrying on a lost race.

    `recording` replaces the remote flag unless `keep_remote_recording` is set (a pure ledger sync that must
    not touch someone else's flag)."""
    for attempt in range(retries):
        snap = store.latest()
        base, remote_raw = (snap if snap else (-1, None))
        remote_ledger, remote_rec = decode_state(remote_raw)
        merged = merge_ledgers(remote_ledger, ledger)
        rec = remote_rec if keep_remote_recording else recording
        if store.push(base, encode_state(merged, rec)):
            return True
        sleep(min(2.0, 0.2 * (attempt + 1)))
    return False
