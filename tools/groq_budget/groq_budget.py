"""Machine-wide Groq token budget ledger. Standard library only, so any project can use it.

Groq's free-tier daily token pool (TPD) is a ROLLING 24 h window, per model, shared by every key
and project in the organisation. Nothing on the Groq side tells a caller what is left, so this
module keeps a ledger of what this machine's tooling spent and lets each caller ask permission
before a request ("may project P spend about E tokens on model M now?").

What it knows: every call some project recorded, plus any spend Groq itself reported in a 429
("Limit L, Used U, Requested R"), folded in as `external-observed`. What it cannot see:
production serving (Cloud Run) and CI smoke tests. That is what the production reserve is for: it
is never allocated to any project.

Defaults (override per model in <root>/config.json, see README.md):
  openai/gpt-oss-120b   cap 200,000 / rolling 24 h
                        production reserve 40,000 (never allocated)
                        project `benchmark`: at most 100,000
                        project `triageiq-recorder`: whatever is left of the 160,000 allocatable
  other models          cap 200,000, reserve 40,000, no per-project caps

A project that records (TriageIQ re-recording cassettes) sets the recording-active flag; while it
is set, other projects' `check` for those models answers "wait".

Ledger root: $GROQ_BUDGET_DIR, else ~/.groq_budget. Concurrent processes are serialised with a
lock file; writes are atomic (temp file + replace).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

WINDOW = timedelta(hours=24)
KEEP = 2 * WINDOW  # entries older than this are dropped on write
RECORDER = "triageiq-recorder"
EXTERNAL = "external-observed"
EXTERNAL_BUCKETS = 24  # an observed gap is held as this many equal slices across the trailing window
LEGACY_MODEL = "openai/gpt-oss-120b"  # the pre-2026-10-12 TriageIQ ledger had no model field
STOP_MARKER = "=== BUDGET RESERVE STOP ==="  # kept: run_recording_unattended greps for it

BUILTIN_CONFIG: dict[str, dict] = {
    "default": {"daily_cap": 200_000, "production_reserve": 40_000, "projects": {}},
    "openai/gpt-oss-120b": {
        "daily_cap": 200_000,
        "production_reserve": 40_000,
        "projects": {"benchmark": 100_000},
    },
}


def root_dir() -> Path:
    override = os.environ.get("GROQ_BUDGET_DIR")
    return Path(override) if override else Path.home() / ".groq_budget"


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------------------- locking
@contextlib.contextmanager
def _locked(root: Path, timeout_s: float = 15.0, stale_s: float = 30.0) -> Iterator[None]:
    """Cross-process mutex via an exclusively created lock file (portable to Windows)."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "ledger.lock"
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_s:  # holder died; take it over
                    lock.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(f"groq_budget: could not lock {lock} within {timeout_s}s") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def _atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------------------------- the ledger
@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str  # "ok" | "recording_active" | "pool_exhausted" | "project_cap"
    detail: str
    retry_after_s: int = 0


class Ledger:
    def __init__(self, root: Path | None = None, path: Path | None = None) -> None:
        """`path` pins the ledger FILE (its directory then holds the lock and flag files); used by
        callers that keep an explicit ledger location, e.g. $TRIAGEIQ_GROQ_LEDGER and tests."""
        self.root = path.parent if path is not None else (root or root_dir())
        self.path = path if path is not None else self.root / "ledger.json"
        self.recording_path = self.root / "recording.json"

    # ---- config
    def config_for(self, model: str) -> dict:
        cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in BUILTIN_CONFIG["default"].items()}
        user: dict = {}
        cfg_file = self.root / "config.json"
        if cfg_file.exists():
            user = json.loads(cfg_file.read_text(encoding="utf-8"))
        for layer in (BUILTIN_CONFIG.get(model, {}), user.get("default", {}), user.get(model, {})):
            for k, v in layer.items():
                if k == "projects":
                    cfg["projects"] = {**cfg["projects"], **v}
                else:
                    cfg[k] = v
        cfg["allocatable"] = cfg["daily_cap"] - cfg["production_reserve"]
        return cfg

    # ---- storage
    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))["entries"]
        out = []
        for e in raw:  # pre-shared-ledger TriageIQ entries have no model/project
            e.setdefault("model", LEGACY_MODEL)
            e.setdefault("project", RECORDER)
            out.append(e)
        return out

    def _write(self, entries: list[dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.path, {"version": 2, "entries": entries})

    def record(
        self,
        project: str,
        model: str,
        tokens: int,
        source: str = "",
        cached_tokens: int | None = None,
        now: datetime | None = None,
    ) -> None:
        """Append one call's TOTAL tokens (prompt + completion) for `project` on `model`."""
        t = now or _now()
        entry: dict = {
            "ts": t.isoformat(),
            "model": model,
            "project": project,
            "tokens": int(tokens),
            "source": source,
        }
        if cached_tokens is not None:
            entry["cached_tokens"] = int(cached_tokens)
        with _locked(self.root):
            keep = [e for e in self._read() if datetime.fromisoformat(e["ts"]) > t - KEEP]
            keep.append(entry)
            self._write(keep)

    def spent(self, model: str, project: str | None = None, now: datetime | None = None) -> int:
        """Tokens in the trailing 24 h on `model` (all projects, or just `project`)."""
        t = now or _now()
        return sum(
            e["tokens"]
            for e in self._read()
            if e["model"] == model
            and (project is None or e["project"] == project)
            and datetime.fromisoformat(e["ts"]) > t - WINDOW
        )

    def by_project(self, model: str, now: datetime | None = None) -> dict[str, int]:
        t = now or _now()
        out: dict[str, int] = {}
        for e in self._read():
            if e["model"] == model and datetime.fromisoformat(e["ts"]) > t - WINDOW:
                out[e["project"]] = out.get(e["project"], 0) + e["tokens"]
        return out

    def _free_at(self, model: str, need: int, project: str | None, now: datetime) -> int:
        """Seconds until enough old spend ages out for `need` tokens (0 if unknown)."""
        freed = 0
        for e in sorted(self._read(), key=lambda x: x["ts"]):
            ts = datetime.fromisoformat(e["ts"])
            if e["model"] == model and (project is None or e["project"] == project) and ts > now - WINDOW:
                freed += e["tokens"]
                if freed >= need:
                    return max(1, int((ts + WINDOW - now).total_seconds()))
        return 0

    # ---- the decision
    def check(self, project: str, model: str, est: int, now: datetime | None = None) -> Decision:
        t = now or _now()
        rec = self.active_recording(model, now=t)
        if rec and rec["project"] != project:
            wait = max(1, min(60, int(rec["expires_ts"] - t.timestamp())))
            return Decision(
                False,
                "recording_active",
                f"project {rec['project']} is recording on {model} (pid {rec['pid']}); wait for it to finish",
                wait,
            )
        cfg = self.config_for(model)
        total = self.spent(model, now=t)
        if total + est > cfg["allocatable"]:
            need = total + est - cfg["allocatable"]
            wait = self._free_at(model, need, None, t)
            when = (t + timedelta(seconds=wait)).strftime("%Y-%m-%d %H:%M UTC") if wait else "unknown"
            return Decision(
                False,
                "pool_exhausted",
                f"{STOP_MARKER}\nLedger: {total} tokens spent on {model} in the last 24 h across all projects; "
                f"allocatable is {cfg['allocatable']} ({cfg['daily_cap']} cap minus a {cfg['production_reserve']} "
                f"production reserve that is never allocated). Room for one more call from about {when}.",
                wait,
            )
        cap = cfg["projects"].get(project)
        if cap is not None:
            mine = self.spent(model, project, now=t)
            if mine + est > cap:
                return Decision(
                    False,
                    "project_cap",
                    f"project {project} has spent {mine} of its {cap} tokens on {model} in the last 24 h",
                    self._free_at(model, mine + est - cap, project, t),
                )
        return Decision(True, "ok", f"{total} of {cfg['allocatable']} allocatable tokens used on {model}")

    # ---- Groq's own number
    def observe_429(self, text: str, model: str, now: datetime | None = None) -> int:
        """Fold a Groq TPD 429 into the ledger; returns the external tokens now held (0 if none).

        The 429 body's "Used" is the only ground truth for the whole organisation over the trailing
        24 h. Spend the ledger did not record (production, other keys) is `used - ledger spend`, and it
        happened at unknown times inside that window. It is therefore held as 24 equal hourly buckets
        spread across the window, so it ages out progressively like real usage does, instead of as one
        block that blocks for 24 h after the 429 (the 2026-10-12 defect: a 179K block observed at 14:49
        UTC kept the ledger shut for the full day although real usage was draining hour by hour).

        A new observation REPLACES the previous one for this model: its "Used" is the newer ground
        truth, so it can correct the estimate downwards as well as upwards. The sum of the buckets is
        exactly the external figure at the moment of observation."""
        parsed = parse_tpd_error(text)
        if parsed is None:
            return 0
        _limit, used, _requested = parsed
        t = now or _now()
        with _locked(self.root):
            entries = self._read()
            mine = [e for e in entries if e["model"] == model and e["project"] != EXTERNAL]
            known = sum(
                e["tokens"] for e in mine if datetime.fromisoformat(e["ts"]) > t - WINDOW
            )
            external = max(0, used - known)
            keep = [
                e
                for e in entries
                if not (e["model"] == model and e["project"] == EXTERNAL)
                and datetime.fromisoformat(e["ts"]) > t - KEEP
            ]
            if external > 0:
                share, extra = divmod(external, EXTERNAL_BUCKETS)
                step = WINDOW / EXTERNAL_BUCKETS
                for k in range(EXTERNAL_BUCKETS):
                    keep.append(
                        {
                            "ts": (t - WINDOW + step * (k + 0.5)).isoformat(),
                            "model": model,
                            "project": EXTERNAL,
                            "tokens": share + (extra if k == EXTERNAL_BUCKETS - 1 else 0),
                            "source": (
                                f"observed-429 at {t.isoformat()}: org Used {used}, ledger {known}; "
                                f"external {external} spread over the window, bucket {k + 1}/{EXTERNAL_BUCKETS}"
                            ),
                        }
                    )
            self._write(sorted(keep, key=lambda e: e["ts"]))
        return external

    # ---- recording-active flag
    def start_recording(
        self, project: str, models: list[str], ttl_s: int = 6 * 3600, pid: int | None = None
    ) -> None:
        now = _now().timestamp()
        with _locked(self.root):
            _atomic_write(
                self.recording_path,
                {
                    "project": project,
                    "models": models,
                    "pid": pid if pid is not None else os.getpid(),
                    "since_ts": now,
                    "expires_ts": now + ttl_s,
                },
            )

    def stop_recording(self, project: str) -> bool:
        with _locked(self.root):
            if not self.recording_path.exists():
                return False
            if json.loads(self.recording_path.read_text(encoding="utf-8"))["project"] != project:
                return False
            self.recording_path.unlink()
            return True

    def active_recording(self, model: str, now: datetime | None = None) -> dict | None:
        """The live recording flag covering `model`, or None (expired TTL or dead pid = not active)."""
        if not self.recording_path.exists():
            return None
        try:
            rec = json.loads(self.recording_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        t = (now or _now()).timestamp()
        if t >= rec["expires_ts"] or not _pid_alive(int(rec["pid"])) or model not in rec["models"]:
            return None
        return rec

    def wait_until_allowed(
        self, project: str, model: str, est: int, timeout_s: float = 3600, poll_s: float = 15
    ) -> Decision:
        deadline = time.monotonic() + timeout_s
        while True:
            d = self.check(project, model, est)
            if d.allowed or time.monotonic() >= deadline:
                return d
            time.sleep(min(poll_s, max(1, d.retry_after_s or poll_s)))

    # ---- one-time import of the pre-shared TriageIQ ledger
    def migrate_legacy(self, legacy_path: Path) -> int:
        """Copy entries of ~/.triageiq/groq_tpd_ledger.json that are not already here (idempotent).
        The legacy file is left untouched. Returns the number of entries imported."""
        if not legacy_path.exists():
            return 0
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))["entries"]
        with _locked(self.root):
            have = {(e["ts"], e["tokens"], e["source"]) for e in self._read()}
            add = [
                {
                    "ts": e["ts"],
                    "model": LEGACY_MODEL,
                    "project": RECORDER,
                    "tokens": e["tokens"],
                    "source": e.get("source", ""),
                    **({"cached_tokens": e["cached_tokens"]} if "cached_tokens" in e else {}),
                }
                for e in legacy
                if (e["ts"], e["tokens"], e.get("source", "")) not in have
            ]
            if add:
                self._write(sorted(self._read() + add, key=lambda e: e["ts"]))
        return len(add)


_USED_RE = re.compile(r"Limit (\d+), Used (\d+), Requested (\d+)")


def parse_tpd_error(text: str) -> tuple[int, int, int] | None:
    """(limit, used, requested) from a Groq TPD 429 message, or None if it is not one."""
    if "tokens per day" not in text.lower():
        return None
    m = _USED_RE.search(text)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="groq_budget", description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=None, help="ledger directory (default $GROQ_BUDGET_DIR or ~/.groq_budget)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser, est: bool = False) -> None:
        p.add_argument("--project", required=True)
        p.add_argument("--model", default=LEGACY_MODEL)
        if est:
            p.add_argument("--est", type=int, default=5_000, help="tokens the next call may spend")

    add_common(sub.add_parser("check", help="exit 0 if the call may go ahead, 4 if it must wait"), est=True)
    p = sub.add_parser("wait", help="block until check passes (or --timeout); exit 0 / 4")
    add_common(p, est=True)
    p.add_argument("--timeout", type=float, default=3600)
    p = sub.add_parser("record", help="record a finished call's total tokens")
    add_common(p)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--cached", type=int, default=None)
    p.add_argument("--source", default="")
    p = sub.add_parser("observe-429", help="fold a Groq TPD 429 body (--text or stdin) into the ledger")
    p.add_argument("--model", default=LEGACY_MODEL)
    p.add_argument("--text", default=None)
    p = sub.add_parser("status", help="print trailing-24h usage per project")
    p.add_argument("--model", default=LEGACY_MODEL)
    p = sub.add_parser("recording", help="start | stop | show the recording-active flag")
    p.add_argument("action", choices=["start", "stop", "show"])
    p.add_argument("--project", default=RECORDER)
    p.add_argument("--model", action="append", default=None)
    p.add_argument("--ttl", type=int, default=6 * 3600)
    p = sub.add_parser("migrate-legacy", help="import ~/.triageiq/groq_tpd_ledger.json")
    p.add_argument("--from", dest="src", type=Path, default=Path.home() / ".triageiq" / "groq_tpd_ledger.json")

    a = ap.parse_args(argv)
    led = Ledger(a.root)
    if a.cmd in ("check", "wait"):
        d = led.check(a.project, a.model, a.est) if a.cmd == "check" else led.wait_until_allowed(
            a.project, a.model, a.est, timeout_s=a.timeout
        )
        print(f"{'ALLOW' if d.allowed else 'WAIT'} [{d.reason}] {d.detail}" + (f" (retry in ~{d.retry_after_s}s)" if d.retry_after_s else ""))
        return 0 if d.allowed else 4
    if a.cmd == "record":
        led.record(a.project, a.model, a.tokens, a.source, a.cached)
        return 0
    if a.cmd == "observe-429":
        added = led.observe_429(a.text if a.text is not None else sys.stdin.read(), a.model)
        print(f"added {added} external tokens")
        return 0
    if a.cmd == "status":
        cfg = led.config_for(a.model)
        usage = led.by_project(a.model)
        print(f"{a.model}: {sum(usage.values())} used / {cfg['allocatable']} allocatable (cap {cfg['daily_cap']}, reserve {cfg['production_reserve']})")
        for proj, n in sorted(usage.items(), key=lambda kv: -kv[1]):
            cap = cfg["projects"].get(proj)
            print(f"  {proj}: {n}" + (f" / {cap}" if cap else ""))
        rec = led.active_recording(a.model)
        print(f"  recording-active: {rec['project'] + ' pid ' + str(rec['pid']) if rec else 'no'}")
        return 0
    if a.cmd == "recording":
        if a.action == "start":
            led.start_recording(a.project, a.model or [LEGACY_MODEL], a.ttl)
        elif a.action == "stop":
            print("stopped" if led.stop_recording(a.project) else "no flag held by this project")
        else:
            print(json.dumps(led.active_recording(a.model[0] if a.model else LEGACY_MODEL)))
        return 0
    print(f"imported {led.migrate_legacy(a.src)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
