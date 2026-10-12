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

Ledger root: $GROQ_BUDGET_DIR, else ~/.groq_budget. Concurrent processes are serialised with a
lock file; writes are atomic (temp file + replace).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

WINDOW = timedelta(hours=24)
KEEP = 2 * WINDOW  # entries older than this are dropped on write
RECORDER = "triageiq-recorder"
EXTERNAL = "external-observed"
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


# --------------------------------------------------------------------------------------- the ledger
@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str  # "ok" | "pool_exhausted" | "project_cap"
    detail: str
    retry_after_s: int = 0


class Ledger:
    def __init__(self, root: Path | None = None, path: Path | None = None) -> None:
        """`path` pins the ledger FILE (its directory then holds the lock and flag files); used by
        callers that keep an explicit ledger location, e.g. $TRIAGEIQ_GROQ_LEDGER and tests."""
        self.root = path.parent if path is not None else (root or root_dir())
        self.path = path if path is not None else self.root / "ledger.json"

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
        """Fold a Groq TPD 429 into the ledger; returns the tokens added (0 if not a TPD 429)."""
        parsed = parse_tpd_error(text)
        if parsed is None:
            return 0
        _limit, used, _requested = parsed
        t = now or _now()
        external = used - self.spent(model, now=t)
        if external <= 0:
            return 0
        self.record(
            EXTERNAL,
            model,
            external,
            f"observed-429: org Used {used} exceeds ledger (external spend)",
            now=t,
        )
        return external

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
