"""Heartbeat + staleness guard for long-running local jobs (GPU training, long local evals).

Motivation: three silent training stalls this session were only caught by checking
nvidia-smi/process state directly rather than trusting "still running" self-reports -- the
same failure class as a false-green CI check (a control that's technically alive but has
stopped doing the thing its name implies). A stalled process still holds its PID and GPU
allocation, so process-liveness alone can't distinguish "working" from "stuck"; only forward
progress can. This module makes progress self-reporting: a job appends one line per interval,
and staleness is judged from *that*, not from whether the process is still resident.

Writer usage (inside a training/eval loop):
    from scripts.heartbeat import Heartbeat
    hb = Heartbeat("reports/heartbeat_deberta_multi_k8s.jsonl")
    ...
    hb.beat(step=step, note=f"loss={loss:.4f}")

Staleness check (run standalone, or from another process/Monitor loop):
    python scripts/heartbeat.py check reports/heartbeat_deberta_multi_k8s.jsonl --max-age-minutes 10
    # exit 0 = fresh; exit 1 = stale (job appears stalled); exit 2 = no heartbeat recorded yet

`check --watch` polls until the file goes stale or is deleted, printing one line per state
change -- suitable as a Monitor tool command so a stall surfaces as a notification instead of
requiring a manual nvidia-smi/process check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path


class Heartbeat:
    """Appends one JSON line per `beat()` call: {"ts": <unix epoch seconds>, "step": ...,
    "note": ...}. Append-only and crash-safe (each beat is flushed immediately) -- a killed or
    hung process simply stops producing new lines, which is exactly the signal `check()` reads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def beat(self, step: int | None = None, note: str = "") -> None:
        record = {"ts": time.time(), "step": step, "note": note}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()


@dataclass
class StalenessResult:
    exists: bool
    fresh: bool
    age_seconds: float | None
    last_record: dict | None


def check_staleness(path: str | Path, max_age_minutes: float) -> StalenessResult:
    """Reads the last line of the heartbeat log and compares its timestamp to now. A missing
    or empty file is reported as `exists=False, fresh=False` -- never treated as "fresh" by
    default, since "no heartbeat yet" and "definitely fine" are not the same claim (rule 98a:
    fail closed, not open, when a check can't verify)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return StalenessResult(exists=False, fresh=False, age_seconds=None, last_record=None)

    last_line = ""
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        return StalenessResult(exists=False, fresh=False, age_seconds=None, last_record=None)

    record = json.loads(last_line)
    age_seconds = time.time() - record["ts"]
    fresh = age_seconds <= max_age_minutes * 60
    return StalenessResult(exists=True, fresh=fresh, age_seconds=age_seconds, last_record=record)


def _format_status(result: StalenessResult, path: str) -> str:
    if not result.exists:
        return f"NO HEARTBEAT: '{path}' has no recorded beats yet"
    age_min = result.age_seconds / 60
    state = "OK" if result.fresh else "STALLED"
    step = result.last_record.get("step")
    note = result.last_record.get("note") or ""
    return f"{state}: last beat {age_min:.1f}min ago (step={step}{', ' + note if note else ''})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check staleness of a heartbeat log once, or with --watch.")
    check.add_argument("path", type=str)
    check.add_argument("--max-age-minutes", type=float, required=True)
    check.add_argument("--watch", action="store_true", help="Poll until stale or the file disappears.")
    check.add_argument("--interval-seconds", type=float, default=30.0)

    args = ap.parse_args()

    if args.command == "check":
        prev_status = None
        while True:
            result = check_staleness(args.path, args.max_age_minutes)
            status = _format_status(result, args.path)
            if status != prev_status:
                print(status, flush=True)
                prev_status = status
            if not args.watch:
                if not result.exists:
                    return 2
                return 0 if result.fresh else 1
            if result.exists and not result.fresh:
                return 1
            time.sleep(args.interval_seconds)

    return 1


if __name__ == "__main__":
    sys.exit(main())
