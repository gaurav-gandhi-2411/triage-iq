from __future__ import annotations

"""Unattended driver for eval/record_cassettes.py — waits out Groq's rate-limit window and
resumes automatically, without a human/CC session babysitting it.

This is a wrapper, not a replacement: record_cassettes.py's own checkpoint (keyed by
(issue_id, model, prompt_hash) as of the 2026-08-30 fix) is still the source of truth for
what's actually recorded. This script only decides WHEN to re-invoke it and WHETHER an
exit is "wait and retry" vs "stop, a human needs to look at this."

Stop conditions (per the working agreement -- these abort the loop, they do not retry):
  - A fallback/degraded synthesis ("SYNTHESIS DEGRADED" in record_cassettes.py's output).
  - A truncated completion ("TRUNCATED COMPLETION").
  - A checkpoint/model mismatch (record_cassettes.py's own STOP on an untagged entry).
  - Any subprocess outcome that doesn't match a known retryable shape -- fail closed
    (rule 98a): an unrecognized failure is a stop, never an infinite silent retry.

Retryable (these just wait and re-invoke the same command):
  - A rate limit, TPD or sustained TPM ("TPD HIT" in the output) -- Groq's error text
    usually names an exact wait ("Please try again in Xm Ys"); parsed and used with a
    1-minute buffer, falling back to a fixed default if the text can't be parsed.
  - A connection error ("CONNECTION LOST") -- shorter fixed backoff, not the TPD wait.

Terminal "done" state is computed from the checkpoint file directly, NOT from the
subprocess's exit code or "RECORDING COMPLETE" text -- record_cassettes.py's own summary
logic exits 1 ("NOT RECORDING COMPLETE (zero live synthesis calls this run)") on every
resume once the only issues left are ones it permanently skips (checkpointed with
plan=None, e.g. a genuine early-termination failure) -- that would otherwise be an
infinite retry trap. This script instead reads recording_checkpoint.json each iteration
and stops once every one of the 64 issues is either resolved (has a judge_score) or
permanently dead (plan=None) under the CURRENTLY configured model+prompt -- there is
nothing left any retry could accomplish at that point.

Early termination on a issue not previously seen is expected data, not a stop condition
-- it gets logged and the loop continues to the next issue automatically (that's just
another "dead" issue counted in the terminal-state check above).
"""

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORKTREE_ROOT = Path(__file__).parent.parent
MAIN_REPO_ENV = Path(r"C:\Users\gaura\ml-projects\triage-iq\.env")
PYTHON = WORKTREE_ROOT / ".venv" / "Scripts" / "python.exe"
RECORD_SCRIPT = WORKTREE_ROOT / "eval" / "record_cassettes.py"
CHECKPOINT_PATH = WORKTREE_ROOT / "eval" / "cassettes" / "recording_checkpoint.json"
EVAL_SET_PATH = WORKTREE_ROOT / "eval" / "eval_set.jsonl"
STATUS_PATH = WORKTREE_ROOT / "eval" / "cassettes" / "RECORDING_STATUS.txt"
LOG_DIR = WORKTREE_ROOT / "eval" / "cassettes" / "unattended_logs"
PID_PATH = WORKTREE_ROOT / "eval" / "cassettes" / "unattended_recorder.pid"

TOTAL_ISSUES = sum(1 for _ in EVAL_SET_PATH.open(encoding="utf-8") if _.strip())

DEFAULT_TPD_WAIT_S = 30 * 60  # fallback if Groq's error text can't be parsed for a wait
CONNECTION_WAIT_S = 5 * 60
RETRY_BUFFER_S = 60

_TPD_WAIT_RE = re.compile(r"try again in\s+(?:(\d+)m)?\s*(?:([\d.]+)s)?", re.IGNORECASE)

HARD_STOP_MARKERS = [
    "SYNTHESIS DEGRADED (not a genuine completion)",
    "TRUNCATED COMPLETION",
    "predates the (issue_id, model, prompt_hash) keying fix",
    "GROQ_API_KEY not set",
    # 2026-09-03 (ADR-0055 Part P1a/2c): a FIRST degraded_schema_invalid on an issue is
    # NOT a hard stop -- it's logged and the run continues (see record_cassettes.py),
    # so it deliberately does not appear here. Only a REPRODUCED failure (2nd
    # occurrence on the same issue) is a hard stop.
    "SCHEMA VALIDATION FAILURE REPRODUCED",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_groq_key() -> str:
    if not MAIN_REPO_ENV.exists():
        raise RuntimeError(f"Main repo .env not found at {MAIN_REPO_ENV}")
    for line in MAIN_REPO_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GROQ_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"GROQ_API_KEY not found in {MAIN_REPO_ENV}")


def _current_model_and_hash() -> tuple[str, str]:
    """Import record_cassettes.py's own hashing logic rather than re-deriving it, so this
    wrapper can never drift out of sync with what a real invocation would compute."""
    sys.path.insert(0, str(WORKTREE_ROOT / "src"))
    sys.path.insert(0, str(RECORD_SCRIPT.parent))
    import record_cassettes as rc  # noqa: E402

    return rc.TRIAGE_MODEL, rc._compute_prompt_hash()


def _checkpoint_progress(model: str, prompt_hash: str) -> tuple[int, int, list[str]]:
    """(resolved, dead, dead_issue_ids) among entries matching (model, prompt_hash)."""
    if not CHECKPOINT_PATH.exists():
        return 0, 0, []
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    resolved = 0
    dead: list[str] = []
    for rec in data.get("done", {}).values():
        if rec.get("model") != model or rec.get("prompt_hash") != prompt_hash:
            continue
        if rec.get("judge_score") is not None:
            resolved += 1
        # tpd_hit AND schema_invalid_retry entries are excluded from "dead" (2026-09-03,
        # ADR-0055 Part P1a/2c) -- both are retryable on a later resume (record_cassettes.py
        # excludes them from done_ids the same way), not permanently skipped. Counting
        # either as dead here would let the terminal-stop check below (resolved+dead ==
        # TOTAL_ISSUES) fire before a retryable issue was ever actually retried.
        elif rec.get("plan") is None and not rec.get("tpd_hit") and not rec.get("schema_invalid_retry"):
            dead.append(rec.get("issue_id", "?"))
    return resolved, len(dead), dead


def _write_status(lines: list[str]) -> None:
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_tpd_wait(text: str) -> int:
    m = _TPD_WAIT_RE.search(text)
    if not m:
        return DEFAULT_TPD_WAIT_S
    minutes = int(m.group(1)) if m.group(1) else 0
    seconds = float(m.group(2)) if m.group(2) else 0.0
    total = minutes * 60 + seconds
    return int(total) if total > 0 else DEFAULT_TPD_WAIT_S


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(__import__("os").getpid()), encoding="utf-8")

    groq_key = _load_groq_key()
    model, prompt_hash = _current_model_and_hash()

    import os

    env = os.environ.copy()
    env["GROQ_API_KEY"] = groq_key

    iteration = 0
    while True:
        iteration += 1
        resolved, dead_count, dead_ids = _checkpoint_progress(model, prompt_hash)

        if resolved + dead_count >= TOTAL_ISSUES:
            _write_status([
                f"STOPPED (terminal): {_now()}",
                f"Model: {model}  Prompt hash: {prompt_hash}",
                f"Resolved: {resolved}/{TOTAL_ISSUES}  Permanently dead (early-termination): {dead_count}",
                f"Dead issue ids: {dead_ids}",
                "All 64 issues are accounted for -- nothing left for a retry to accomplish.",
                "Decide on the dead issue(s) per docs/SESSION_RESUME_2026-08-30.md before",
                "declaring the recording complete.",
                f"Iterations run: {iteration - 1}",
            ])
            print("DONE: all issues accounted for, see RECORDING_STATUS.txt")
            return

        _write_status([
            f"RUNNING: {_now()}",
            f"Model: {model}  Prompt hash: {prompt_hash}",
            f"Resolved: {resolved}/{TOTAL_ISSUES}  Permanently dead so far: {dead_count} {dead_ids}",
            f"Iteration {iteration}: invoking record_cassettes.py now...",
        ])

        log_path = LOG_DIR / f"iter_{iteration:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        proc = subprocess.run(
            [str(PYTHON), str(RECORD_SCRIPT)],
            cwd=str(WORKTREE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(output, encoding="utf-8")

        hard_stop = next((m for m in HARD_STOP_MARKERS if m in output), None)
        if hard_stop:
            resolved, dead_count, dead_ids = _checkpoint_progress(model, prompt_hash)
            _write_status([
                f"BLOCKED (hard stop): {_now()}",
                f"Reason: {hard_stop!r} found in subprocess output.",
                f"Model: {model}  Prompt hash: {prompt_hash}",
                f"Resolved: {resolved}/{TOTAL_ISSUES}  Dead: {dead_count} {dead_ids}",
                f"Full log: {log_path}",
                f"Iterations run: {iteration}",
            ])
            print(f"BLOCKED (hard stop): {hard_stop!r} -- see RECORDING_STATUS.txt")
            return

        if "=== TPD HIT" in output:
            wait_s = _parse_tpd_wait(output) + RETRY_BUFFER_S
            resolved, dead_count, dead_ids = _checkpoint_progress(model, prompt_hash)
            resume_at = datetime.now(timezone.utc).timestamp() + wait_s
            _write_status([
                f"WAITING (rate limit): {_now()}",
                f"Model: {model}  Prompt hash: {prompt_hash}",
                f"Resolved: {resolved}/{TOTAL_ISSUES}  Dead: {dead_count} {dead_ids}",
                f"Sleeping {wait_s}s (~{wait_s // 60}m), resuming at "
                f"{datetime.fromtimestamp(resume_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}.",
                f"Full log: {log_path}",
                f"Iterations run: {iteration}",
            ])
            time.sleep(wait_s)
            continue

        if "=== CONNECTION LOST" in output:
            resolved, dead_count, dead_ids = _checkpoint_progress(model, prompt_hash)
            _write_status([
                f"WAITING (connection error): {_now()}",
                f"Model: {model}  Prompt hash: {prompt_hash}",
                f"Resolved: {resolved}/{TOTAL_ISSUES}  Dead: {dead_count} {dead_ids}",
                f"Sleeping {CONNECTION_WAIT_S}s (~{CONNECTION_WAIT_S // 60}m) then retrying.",
                f"Full log: {log_path}",
                f"Iterations run: {iteration}",
            ])
            time.sleep(CONNECTION_WAIT_S)
            continue

        # Anything else (including exit 0 mid-run, or an exit 1 this wrapper doesn't
        # recognize) is a fail-closed stop -- never loop silently on an unknown outcome.
        resolved, dead_count, dead_ids = _checkpoint_progress(model, prompt_hash)
        _write_status([
            f"BLOCKED (unrecognized outcome): {_now()}",
            f"Subprocess exit code: {proc.returncode}",
            f"Model: {model}  Prompt hash: {prompt_hash}",
            f"Resolved: {resolved}/{TOTAL_ISSUES}  Dead: {dead_count} {dead_ids}",
            f"Full log: {log_path}",
            "This wrapper does not retry an outcome it doesn't recognize -- inspect the",
            "log and decide manually.",
            f"Iterations run: {iteration}",
        ])
        print(f"BLOCKED (unrecognized outcome, exit={proc.returncode}) -- see RECORDING_STATUS.txt")
        return


if __name__ == "__main__":
    main()
