from __future__ import annotations

"""Unattended driver for eval/record_cassettes.py — waits out Groq's rate-limit window and
resumes automatically, without a human/CC session babysitting it.

2026-09-06 (ADR-0060): split into --mode synthesis / --mode judge, mirroring
record_cassettes.py's own split. Synthesis and judging have different resource profiles and
different "done" definitions:
  - synthesis: Groq calls only. Runs at BelowNormal priority with BLAS threads capped
    (OMP/MKL/OPENBLAS_NUM_THREADS=2) so it yields CPU to whatever else is running on this
    machine. "Done" means every issue has a plan (judge_score irrelevant here).
  - judge: local Ollama only, zero Groq calls. Ollama's qwen3:8b holds ~5.5 GB VRAM on an
    8 GB card while loaded, so this mode checks free VRAM (nvidia-smi) before EVERY
    invocation and waits rather than contending with another GPU-heavy process. "Done"
    means every issue has a judge_score (record_cassettes.py's original definition,
    unchanged).

This is a wrapper, not a replacement: record_cassettes.py's own checkpoint (keyed by
(issue_id, model, prompt_hash, artifact_hash) as of ADR-0058/0059) is still the source of
truth for what's actually recorded. This script only decides WHEN to re-invoke it and
WHETHER an exit is "wait and retry" vs "stop, a human needs to look at this."

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
  - (judge mode only) Insufficient free VRAM -- waits and re-checks rather than starting.
"""

import argparse
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

sys.path.insert(0, str(WORKTREE_ROOT / "src"))
sys.path.insert(0, str(RECORD_SCRIPT.parent))
import record_cassettes as rc  # noqa: E402
import artifact_fingerprint  # noqa: E402
# _parse_tpd_wait lives in record_cassettes.py, not duplicated here -- it sees the raw Groq
# error text first and needs the identical regex to report an accurate resume estimate in
# RECORDING_STATUS.txt at the moment it hits the wall (ADR-0060); importing it keeps this
# wrapper's own (later, coarser) re-parse of captured stdout from ever drifting out of sync.

TOTAL_ISSUES = sum(1 for _ in EVAL_SET_PATH.open(encoding="utf-8") if _.strip())

CONNECTION_WAIT_S = 5 * 60
RETRY_BUFFER_S = 60
VRAM_CHECK_INTERVAL_S = 2 * 60
MIN_FREE_VRAM_MB = 6 * 1024

HARD_STOP_MARKERS = [
    "SYNTHESIS DEGRADED (not a genuine completion)",
    "TRUNCATED COMPLETION",
    "predates the composite-key fix",
    "resolved ROOT to",  # ADR-0059: wrong-checkout tripwire in record_cassettes.py
    "does not exist. This file is the explicit, committed statement",  # missing expected-hash file
    "resolved artifacts do not match",  # ADR-0059: artifact mismatch against expected hashes
    "GROQ_API_KEY not set",
    "JUDGE PASS STOPPED (unexpected rate-limit signal)",
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


def _current_model_and_hash() -> tuple[str, str, str]:
    """Reuse record_cassettes.py's own hashing logic (module-level import above) rather
    than re-deriving it, so this wrapper can never drift out of sync with what a real
    invocation would compute."""
    artifact_hashes = artifact_fingerprint.compute_artifact_hashes(WORKTREE_ROOT)
    return rc.TRIAGE_MODEL, rc._compute_prompt_hash(), artifact_fingerprint.combined_hash(artifact_hashes)


def _checkpoint_entries(model: str, prompt_hash: str, artifact_hash: str) -> list[dict]:
    if not CHECKPOINT_PATH.exists():
        return []
    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    return [
        rec for rec in data.get("done", {}).values()
        if rec.get("model") == model and rec.get("prompt_hash") == prompt_hash
        and rec.get("artifact_hash") == artifact_hash
    ]


def _progress(model: str, prompt_hash: str, artifact_hash: str) -> dict:
    """Both synthesis and judge progress, always both -- Phase 3e: the status file must
    show N/64 synthesis-done AND N/64 judged regardless of which mode is currently running,
    so a human glancing at RECORDING_STATUS.txt never has to guess the other number."""
    entries = _checkpoint_entries(model, prompt_hash, artifact_hash)
    synthesized = sum(1 for rec in entries if rec.get("plan") is not None)
    # 2026-09-23 (judge-provenance fix, B4): a judge_score recorded under a since-changed
    # judge model/rubric must not count as "judged" here either -- this drives the judge
    # loop's own terminal check (main()'s `p["judged"] + len(p["dead"]) >= TOTAL_ISSUES`),
    # and without this it would consider judging complete based on stale scores. Reuses
    # record_cassettes.py's own predicate (rc, imported above) rather than re-deriving it.
    judged = sum(1 for rec in entries if rec.get("judge_score") is not None and rc._judge_config_current(rec))
    dead = [
        rec.get("issue_id", "?") for rec in entries
        if rec.get("plan") is None and not rec.get("tpd_hit") and not rec.get("schema_invalid_retry")
    ]
    return {"synthesized": synthesized, "judged": judged, "dead": dead}


def _write_status(mode: str, model: str, prompt_hash: str, artifact_hash: str, extra: list[str]) -> None:
    p = _progress(model, prompt_hash, artifact_hash)
    lines = [
        f"Mode: {mode}  Updated: {_now()}",
        f"Model: {model}  Prompt hash: {prompt_hash}  Artifact hash: {artifact_hash}",
        f"Synthesis-done: {p['synthesized']}/{TOTAL_ISSUES}   Judged: {p['judged']}/{TOTAL_ISSUES}"
        f"   Permanently dead: {len(p['dead'])} {p['dead']}",
        *extra,
    ]
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _free_vram_mb() -> int | None:
    """None means "couldn't determine" (nvidia-smi missing/failed) -- callers treat that as
    "assume contended, wait" (fail closed on resource availability: waiting is cheap and
    reversible, proceeding into a genuinely contended GPU is not)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def _wait_for_vram(mode: str, model: str, prompt_hash: str, artifact_hash: str) -> None:
    """Phase 3c: never start the judge pass while another process holds most of this
    machine's 8 GB card. Waits and re-checks rather than starting -- 'slower is fine'."""
    if mode != "judge":
        return
    while True:
        free_mb = _free_vram_mb()
        if free_mb is not None and free_mb >= MIN_FREE_VRAM_MB:
            return
        _write_status(mode, model, prompt_hash, artifact_hash, [
            f"WAITING (VRAM): free={free_mb if free_mb is not None else 'unknown (nvidia-smi check failed)'} MiB, "
            f"need >={MIN_FREE_VRAM_MB} MiB before starting the judge pass.",
            f"Re-checking in {VRAM_CHECK_INTERVAL_S}s.",
        ])
        time.sleep(VRAM_CHECK_INTERVAL_S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("synthesis", "judge"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(__import__("os").getpid()), encoding="utf-8")

    import os

    env = os.environ.copy()
    if mode == "synthesis":
        groq_key = _load_groq_key()
        env["GROQ_API_KEY"] = groq_key
        env["TRIAGE_PROMPT_INCLUDE_ATTRIBUTION"] = "1"
        # Phase 3b: cap BLAS thread pools so the classifier/TF-IDF matrix ops this process
        # does don't spin up threads competing with whatever else is on this CPU. Must be
        # set before the child process imports numpy/sklearn -- setting them here (env dict
        # for a FRESH subprocess) achieves that; setting them mid-process in
        # record_cassettes.py itself would be too late (BLAS reads them once at library init).
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[var] = "2"
        # Note: record_cassettes.py's synthesis path never imports torch, FAISS, or
        # sentence-transformers (confirmed directly, ADR-0060 Phase 1b measurement) --
        # frozen retrieval reads pre-computed similar_issues from eval_set.jsonl, no BGE
        # embedding or GPU work happens in this process. torch.set_num_threads and
        # "force BGE to CPU" have no code path to apply to here; not implemented because
        # there is nothing for them to constrain, not because they were skipped.
    priority_flag = subprocess.BELOW_NORMAL_PRIORITY_CLASS

    # _current_model_and_hash() computes the prompt hash by reading THIS process's own
    # os.environ (not the `env` dict built above, which only affects the child subprocess)
    # -- set it here too so this wrapper's notion of "current config" matches exactly what
    # the child (which inherits `env`) will compute for itself.
    if mode == "synthesis":
        os.environ["TRIAGE_PROMPT_INCLUDE_ATTRIBUTION"] = "1"
    model, prompt_hash, artifact_hash = _current_model_and_hash()

    iteration = 0
    while True:
        iteration += 1
        p = _progress(model, prompt_hash, artifact_hash)

        if mode == "synthesis":
            terminal = p["synthesized"] + len(p["dead"]) >= TOTAL_ISSUES
        else:
            terminal = p["judged"] + len(p["dead"]) >= TOTAL_ISSUES

        if terminal:
            _write_status(mode, model, prompt_hash, artifact_hash, [
                "STOPPED (terminal): nothing left for a retry to accomplish in this mode.",
                f"Iterations run: {iteration - 1}",
            ])
            print(f"DONE ({mode}): all issues accounted for, see RECORDING_STATUS.txt")
            return

        _wait_for_vram(mode, model, prompt_hash, artifact_hash)

        _write_status(mode, model, prompt_hash, artifact_hash, [
            f"RUNNING: iteration {iteration}, invoking record_cassettes.py --mode {mode} now...",
        ])

        log_path = LOG_DIR / f"{mode}_iter_{iteration:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        proc = subprocess.run(
            [str(PYTHON), str(RECORD_SCRIPT), "--mode", mode],
            cwd=str(WORKTREE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=priority_flag,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(output, encoding="utf-8")

        hard_stop = next((m for m in HARD_STOP_MARKERS if m in output), None)
        if hard_stop:
            _write_status(mode, model, prompt_hash, artifact_hash, [
                f"BLOCKED (hard stop): {hard_stop!r} found in subprocess output.",
                f"Full log: {log_path}",
                f"Iterations run: {iteration}",
            ])
            print(f"BLOCKED (hard stop): {hard_stop!r} -- see RECORDING_STATUS.txt")
            return

        if "=== TPD HIT" in output:
            wait_s = rc._parse_tpd_wait(output) + RETRY_BUFFER_S
            resume_at = datetime.now(timezone.utc).timestamp() + wait_s
            _write_status(mode, model, prompt_hash, artifact_hash, [
                f"WAITING (rate limit): sleeping {wait_s}s (~{wait_s // 60}m), resuming at "
                f"{datetime.fromtimestamp(resume_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}.",
                f"Full log: {log_path}",
                f"Iterations run: {iteration}",
            ])
            time.sleep(wait_s)
            continue

        if "=== CONNECTION LOST" in output:
            _write_status(mode, model, prompt_hash, artifact_hash, [
                f"WAITING (connection error): sleeping {CONNECTION_WAIT_S}s (~{CONNECTION_WAIT_S // 60}m) then retrying.",
                f"Full log: {log_path}",
                f"Iterations run: {iteration}",
            ])
            time.sleep(CONNECTION_WAIT_S)
            continue

        done_marker = "=== SYNTHESIS COMPLETE ===" if mode == "synthesis" else "=== JUDGE COMPLETE ==="
        if proc.returncode == 0 and done_marker in output:
            _write_status(mode, model, prompt_hash, artifact_hash, [
                f"DONE ({mode} complete): {done_marker}",
                f"Full log: {log_path}",
                f"Iterations run: {iteration}",
            ])
            print(f"DONE: {mode} complete this iteration, see RECORDING_STATUS.txt")
            return

        if mode == "judge" and proc.returncode == 0 and "=== NOTHING TO JUDGE ===" in output and "Not yet synthesized: 0" in output:
            _write_status(mode, model, prompt_hash, artifact_hash, [
                "DONE (judge complete): nothing left to judge, all issues fully judged.",
                f"Full log: {log_path}",
                f"Iterations run: {iteration}",
            ])
            print("DONE: judge complete, see RECORDING_STATUS.txt")
            return

        # Anything else (including exit 0 mid-run, or an exit 1 this wrapper doesn't
        # recognize) is a fail-closed stop -- never loop silently on an unknown outcome.
        _write_status(mode, model, prompt_hash, artifact_hash, [
            f"BLOCKED (unrecognized outcome): subprocess exit code {proc.returncode}.",
            f"Full log: {log_path}",
            "This wrapper does not retry an outcome it doesn't recognize -- inspect the log and decide manually.",
            f"Iterations run: {iteration}",
        ])
        print(f"BLOCKED (unrecognized outcome, exit={proc.returncode}) -- see RECORDING_STATUS.txt")
        return


if __name__ == "__main__":
    main()
