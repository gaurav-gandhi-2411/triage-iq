from __future__ import annotations

"""One-time recording pass: runs the full TriageIQ pipeline + judge over
eval/eval_set.jsonl and saves ALL LLM interactions to
eval/cassettes/eval_cassette.json.

Run ONCE locally with live GROQ_API_KEY set.  CI never runs this script.

Usage:
    python eval/record_cassettes.py

Per-issue TPD deferral (2026-08-23): a Groq TPD (tokens-per-day) hit while
processing ONE issue no longer aborts the whole pass. It marks that issue
"deferred" in recording_checkpoint.json and moves on to the NEXT not-yet-done
issue instead — so an issue that structurally needs two LLM calls (its first
response fails JSON parsing, so triage.py's own retry logic fires a second
call) doesn't repeatedly consume an entire pass's recovered TPD headroom on a
doomed attempt while every other, easier (single-call) issue sits unattempted
behind it in file order. Never-deferred issues are always tried before
previously-deferred ones; deferred issues are only retried once no
never-deferred issue remains. The pass still short-circuits (stops attempting
further issues) the first time TPD is actually hit, since a global per-account
budget being exhausted for one issue means it is exhausted for all of them —
there is no point burning a ~108s backoff sequence per remaining issue to
rediscover that.

Self-resuming across a TPD boundary (2026-08-25): Groq's 429 body for a TPD
hit names an exact recovery time ("Please try again in 6m40.464s") because
the 200K-token/day limit is a rolling 24h window, not a fixed midnight
reset — headroom recovers continuously as old usage ages out, in minutes, not
a day. A prior version of this script exited 1 on the first TPD hit and
relied on a human (or a scheduler) to notice and re-invoke it; that
expectation silently failed — the checkpoint sat untouched for 30+ hours with
nothing running. main() now parses that "try again in ..." duration
(_parse_tpd_retry_seconds) and sleeps exactly that long (+ a small buffer)
before automatically resuming the same process, instead of exiting. It only
exits 1 if the wait can't be determined (Groq's error message doesn't match
the expected shape) or after several consecutive passes make zero progress —
both fail loud rather than sleeping on a guess or spinning silently, per this
repo's own fail-closed rule for guards that can't verify what they're
waiting on.

Exit codes:
    0 — all issues recorded successfully
    1 — recording incomplete for a non-retryable reason (a non-TPD failure,
        an unparseable TPD error, or no progress across several consecutive
        TPD-wait cycles) — genuine incompleteness, not "run me again later"
    1 — unexpected error
"""

import json
import logging
import numpy as np
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import os

from cassette import CassettePlayer
from frozen_retriever import build_frozen_retrievers
from triage_iq.models.component_classifier import load_classifier
from triage_iq.evaluation.triage_eval import DIMENSION_MAX, JudgeScore, TriageJudge
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
CASSETTE_PATH = ROOT / "eval" / "cassettes" / "eval_cassette.json"
CHECKPOINT_PATH = ROOT / "eval" / "cassettes" / "recording_checkpoint.json"

REPO_MAP = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}

SYNTHESIS_DELAY = 1.5  # seconds between synthesis calls (8B model: high TPM, 1.5s is safe)
# Local judge (ADR-0019): no rate limit, no delay needed between judge calls.
JUDGE_DELAY = 0.0
JUDGE_MODEL = "qwen3:8b"
JUDGE_PROVIDER = "ollama"

# Self-resume tuning (2026-08-25). TPD_WAIT_BUFFER pads Groq's own "try again in"
# duration so a boundary reached a second early (clock drift, our sleep's own
# call overhead) doesn't immediately re-hit the same 429. MAX_CONSECUTIVE_STALLS
# bounds the self-resume loop: if that many passes in a row make zero progress
# (nothing recorded, deferred set unchanged) despite each waiting out Groq's own
# stated recovery time, something other than "budget hasn't recovered yet" is
# wrong -- fail loud instead of sleeping forever on a premise that isn't holding.
#
# Raised from 5 to 20 the same day (live, before it fired): a single ~6,000-token
# item near the end of the run hit 4 consecutive passes with zero progress, each
# one correctly parsing and honoring Groq's own "try again in ~40m" estimate and
# still failing on retry -- Groq's recovery estimate evidently isn't precise
# enough, on a near-saturated account, to reliably land inside its own stated
# window. That's real, observed, legitimate TPD recovery being slower than
# expected, not a bug -- 5 was too low a bar for "something else is wrong" and
# would have aborted a job that was going to finish fine. 20 gives ~13+ hours of
# genuine waiting (at the ~40min/pass observed here) before concluding something
# other than recovery-speed is actually broken.
TPD_WAIT_BUFFER = 15.0
MAX_CONSECUTIVE_STALLS = 20


def _is_tpd_error(exc: Exception) -> bool:
    """True only for genuine per-day token exhaustion (cannot retry same day)."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("tokens per day", "daily limit", "tpd"))


def _is_rate_limit_error(exc: Exception) -> bool:
    """True for any Groq 429 — per-minute or per-day."""
    return "rate_limit_exceeded" in str(exc).lower() or _is_tpd_error(exc)


def _is_connection_error(exc: Exception) -> bool:
    """True for network/connectivity failures that should PAUSE recording, not mark issues failed."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("connection error", "connection refused", "getaddrinfo",
                                    "connecterror", "apiconnectionerror", "timed out", "timeout"))


_RETRY_AFTER_RE = re.compile(
    r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", re.IGNORECASE
)


def _parse_tpd_retry_seconds(exc: Exception) -> float | None:
    """Parse Groq's own "Please try again in 6m40.464s" out of a TPD 429 body.

    Returns None (never a guessed default) if the message doesn't match the
    expected shape -- an unparseable wait is a reason to fail loud, not to
    sleep for an invented duration (see module docstring, 2026-08-25).
    """
    m = _RETRY_AFTER_RE.search(str(exc))
    if not m or not any(m.groups()):
        return None
    h, mnt, s = m.groups()
    return (int(h or 0) * 3600) + (int(mnt or 0) * 60) + float(s or 0)


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        data.setdefault("deferred", {})
        logger.info("Checkpoint: %d issues already recorded, %d deferred (need >1 call)",
                    len(data.get("done", {})), len(data["deferred"]))
        return data
    return {"done": {}, "deferred": {}}


def save_checkpoint(data: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": eval/cassettes/*.json is declared `text eol=lf` in .gitattributes (same
    # reasoning as CassettePlayer._save() in cassette.py -- see ADR-0025).
    CHECKPOINT_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )


def load_eval_set() -> list[dict]:
    issues = []
    with open(EVAL_SET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                issues.append(json.loads(line))
    return issues


def main() -> None:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        logger.error("GROQ_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    issues = load_eval_set()
    logger.info("Eval set: %d issues (%s)",
                len(issues),
                ", ".join(f"{r}: {sum(1 for i in issues if i['repo']==r)}"
                          for r in sorted(set(i['repo'] for i in issues))))

    # Load cassette in record mode (strict=False)
    cassette = CassettePlayer(CASSETTE_PATH, strict=False)
    logger.info("Cassette: %d entries already recorded at %s", cassette.stats()["entries"], CASSETTE_PATH)

    # Load checkpoint
    checkpoint = load_checkpoint()
    _done_entries = checkpoint.get("done", {})
    # Exclude tpd_hit entries so they are retried — their synthesis is cached, only the judge reruns.
    done_ids = {k for k, v in _done_entries.items() if not v.get("tpd_hit")}
    n_tpd_retry = len(_done_entries) - len(done_ids)
    logger.info("Checkpoint: %d issues already processed (%d tpd_hit will retry)",
                len(done_ids), n_tpd_retry)
    n_skipped_at_start = len(done_ids)  # frozen here -- the loop below re-derives done_ids per pass

    # Build frozen retrievers — deterministic prompts regardless of hardware
    try:
        frozen_retrievers = build_frozen_retrievers(EVAL_SET)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # Load models per repo
    models: dict[str, dict] = {}
    for repo, slug in REPO_MAP.items():
        models_dir = ROOT / "data" / "models"
        processed_dir = ROOT / "data" / "processed"
        try:
            # load_classifier() dispatches on the pkl's model_kind marker (ADR-0036).
            classifier = load_classifier(models_dir, slug)
            predictor = ResolutionTimePredictor.load(
                str(models_dir / f"resolution_predictor_{slug}.pkl")
            )
            train_df = pd.read_parquet(processed_dir / f"{slug}_temporal_train.parquet")
            assistant = TriageAssistant(
                repo=repo,
                classifier=classifier,
                detector=frozen_retrievers[repo],  # frozen, not live FAISS
                predictor=predictor,
                train_df=train_df,
                groq_api_key=groq_key,
                cache=cassette,
            )
            models[repo] = {
                "classifier": classifier,
                "predictor": predictor,
                "train_df": train_df,
                "assistant": assistant,
            }
            logger.info("Models loaded for %s", repo)
        except Exception as exc:
            logger.error("Failed to load models for %s: %s", repo, exc)
            sys.exit(1)

    judge = TriageJudge(
        model=JUDGE_MODEL,
        provider=JUDGE_PROVIDER,
        temperature=0.0,
        ollama_seed=42,
        cache=cassette,
    )

    # Throwaway warm-up call (ADR-0019): the first inference after a fresh Ollama model
    # load produces different output than subsequent calls on the same loaded instance —
    # each mode is independently reproducible, but they differ from each other. Absorbing
    # that one-time cold-start here means every REAL judge call below (issue 1 through the
    # last) is uniformly warm-mode, so a from-scratch re-run of this whole script
    # reproduces the identical cassette. Not cached, not counted, discarded immediately.
    logger.info("Ollama judge warm-up call (absorbing cold-start variance)...")
    judge._ollama_completion([{"role": "user", "content": "Reply with just: OK"}])
    logger.info("Warm-up done — judge is now in steady warm-mode for the rest of this run.")

    total_synthesis_recorded = 0
    total_judge_recorded = 0
    consecutive_stalls = 0
    pass_num = 0

    # Self-resuming pass loop (2026-08-25): each iteration is exactly what the old
    # single-shot main() body did, but a TPD hit now sleeps out Groq's own stated
    # recovery time and loops back instead of exiting -- see module docstring.
    # done_ids/results/deferred_map/queue are re-derived from `checkpoint` at the top
    # of every pass because `checkpoint["done"]`/`checkpoint["deferred"]` are mutated
    # in place (via save_checkpoint()) throughout the pass below.
    while True:
        pass_num += 1
        n_synthesis_recorded = 0
        n_judge_recorded = 0

        done_ids = {k for k, v in checkpoint["done"].items() if not v.get("tpd_hit")}
        # Pre-seed results with everything already recorded so the end-of-run summary/
        # completeness check (below) still sees them without re-iterating the whole eval set.
        results: dict[str, dict] = {iid: v for iid, v in checkpoint["done"].items()
                                     if iid in done_ids}

        deferred_map: dict[str, dict] = checkpoint.get("deferred", {})

        # Priority queue: never-deferred issues first (in original file order), so a
        # single-call-only issue gets first crack at whatever TPD headroom has recovered
        # this pass. Only fall through to previously-deferred (known >1-call) issues once
        # nothing else remains — see module docstring.
        not_done = [(idx, iss) for idx, iss in enumerate(issues) if iss["id"] not in done_ids]
        fresh_queue = [(idx, iss) for idx, iss in not_done if iss["id"] not in deferred_map]
        hard_queue = [(idx, iss) for idx, iss in not_done if iss["id"] in deferred_map]
        queue = fresh_queue if fresh_queue else hard_queue
        if not fresh_queue and hard_queue:
            logger.info(
                "All %d remaining issues are previously-deferred (needed >1 LLM call last "
                "time) — attempting them now since no never-deferred issue is left: %s",
                len(hard_queue), [iss["id"] for _, iss in hard_queue],
            )

        recorded_ids_this_pass: list[str] = []
        deferred_ids_this_pass: list[str] = []
        budget_exhausted_this_pass = False
        last_tpd_exc: Exception | None = None
        entries_before_pass = cassette.stats()["entries"]

        logger.info("=== Pass %d: %d issues queued (%d fresh, %d previously-deferred) ===",
                    pass_num, len(queue), len(fresh_queue), len(hard_queue))

        for qpos, (i, issue) in enumerate(queue):
            issue_id = issue["id"]
            repo = issue["repo"]

            if qpos > 0:
                time.sleep(SYNTHESIS_DELAY)

            logger.info("[%d/%d] %s — triaging …", i + 1, len(issues), issue_id)

            row = pd.Series({
                "title": issue["title"],
                "body_clean": issue["body"],
                "number": issue["number"],
                "created_at": pd.Timestamp(issue["created_at"]) if issue.get("created_at") else pd.Timestamp("now", tz="UTC"),
            })

            # --- Synthesis ---
            assistant = models[repo]["assistant"]
            plan = None
            triage_error = None
            try:
                plan, meta = assistant.triage_with_metadata(row)
                n_synthesis_recorded += 1
                logger.info("  synthesis → %s (cache_hit=%s)", plan.predicted_component, meta.get("llm_cache_hit"))
            except Exception as exc:
                if _is_tpd_error(exc):
                    last_tpd_exc = exc
                    logger.warning(
                        "  %s deferred: Groq TPD hit mid-issue (after %d synthesis calls "
                        "this pass, cassette has %d entries). This issue needed a second "
                        "LLM call (its first response didn't parse) and the account had no "
                        "headroom left for it -- deferring to a later pass rather than "
                        "burning the rest of this pass's queue on doomed retries. "
                        "Groq error: %s",
                        issue_id, n_synthesis_recorded, cassette.stats()["entries"], exc,
                    )
                    prior = deferred_map.get(issue_id, {})
                    deferred_map[issue_id] = {
                        "reason": "tpd_mid_synthesis",
                        "attempts": prior.get("attempts", 0) + 1,
                        "last_groq_error": str(exc)[:300],
                    }
                    checkpoint["deferred"] = deferred_map
                    save_checkpoint(checkpoint)
                    deferred_ids_this_pass.append(issue_id)
                    budget_exhausted_this_pass = True
                    # Short-circuit: TPD is an account-wide budget, not per-issue -- if THIS
                    # call couldn't get headroom, no other issue's call will either, this pass.
                    break
                if _is_connection_error(exc):
                    logger.error(
                        "STOP: connection lost after %d synthesis calls. "
                        "Cassette has %d entries. Error: %s",
                        n_synthesis_recorded, cassette.stats()["entries"], exc,
                    )
                    # save the FULL checkpoint object (not just {"done": ...}) -- an earlier
                    # version of this line dropped the "deferred" map on every connection
                    # error, silently erasing which issues are known to need >1 LLM call.
                    save_checkpoint(checkpoint)
                    print("\n=== CONNECTION LOST ===")
                    print(f"Synthesis recorded (this pass): {n_synthesis_recorded}")
                    print(f"Judge recorded (this pass): {n_judge_recorded}")
                    print(f"Cassette entries: {cassette.stats()['entries']}")
                    sys.exit(1)  # incomplete recording -- see TPD-exit comment above
                logger.warning("  synthesis FAILED: %s", exc)
                triage_error = str(exc)

            if plan is None:
                results[issue_id] = {"error": triage_error, "plan": None, "judge_score": None}
                checkpoint["done"][issue_id] = results[issue_id]
                save_checkpoint(checkpoint)
                continue

            # --- Judge ---
            time.sleep(JUDGE_DELAY)
            # exclude={"declared_attribution", "abstention_status"}: must match run_eval.py's
            # plan_json exactly (same fields) so the judge cache key computed here at record
            # time is the same one run_eval.py's replay looks up later. Both fields are
            # unconditional on TriagePlan (always serialize, even as None) but never populated
            # by this eval harness -- see run_eval.py's ADR-0020/ADR-0021 comment for detail.
            plan_json = json.dumps(
                plan.model_dump(exclude={"declared_attribution", "abstention_status"}),
                ensure_ascii=False,
            )
            gold = {
                "component": issue["gold_component"],
                "priority": issue["gold_priority"],
                "actual_resolution_days": issue["actual_resolution_days"],
            }

            judge_score = None
            _judge_exc: Exception | None = None
            for _attempt in range(6):
                try:
                    score = judge.score(
                        issue_title=issue["title"],
                        issue_body=issue["body"][:600],
                        triage_plan_json=plan_json,
                        gold=gold,
                    )
                    judge_score = score.model_dump()
                    n_judge_recorded += 1
                    logger.info(
                        "  judge → %d/%d (cassette_entries=%d)",
                        score.total(), sum(DIMENSION_MAX.values()),
                        cassette.stats()["entries"],
                    )
                    _judge_exc = None
                    break
                except Exception as exc:
                    if _is_tpd_error(exc):
                        _judge_exc = exc
                        last_tpd_exc = exc
                        break  # genuine daily limit — stop outer loop below
                    if _is_connection_error(exc):
                        logger.error("STOP: connection lost during judging: %s", exc)
                        save_checkpoint(checkpoint)
                        print("\n=== CONNECTION LOST (during judge) ===")
                        print(f"Synthesis recorded (this pass): {n_synthesis_recorded}")
                        print(f"Judge recorded (this pass): {n_judge_recorded}")
                        sys.exit(1)  # incomplete recording -- see TPD-exit comment above
                    if _is_rate_limit_error(exc):
                        _wait = 20 * (2 ** _attempt)
                        logger.warning(
                            "  judge TPM rate limit (per-minute) — waiting %ds (attempt %d/6): %s",
                            _wait, _attempt + 1, str(exc)[:200],
                        )
                        time.sleep(_wait)
                        continue
                    logger.warning("  judge FAILED: %s", exc)
                    break
            if _judge_exc is not None:
                logger.warning(
                    "  %s deferred: Groq TPD hit during judging (after %d judge calls "
                    "this pass, cassette has %d entries). Synthesis for this issue is "
                    "already cached -- only the judge call will re-run next time. "
                    "Groq error: %s",
                    issue_id, n_judge_recorded, cassette.stats()["entries"], _judge_exc,
                )
                # tpd_hit=True keeps this OUT of done_ids on the next load (so it's retried),
                # while still recording that synthesis succeeded and is cached.
                results[issue_id] = {
                    "plan": plan.model_dump(),
                    "judge_score": None,
                    "tpd_hit": True,
                }
                checkpoint["done"][issue_id] = results[issue_id]
                prior = deferred_map.get(issue_id, {})
                deferred_map[issue_id] = {
                    "reason": "tpd_mid_judge",
                    "attempts": prior.get("attempts", 0) + 1,
                    "last_groq_error": str(_judge_exc)[:300],
                }
                checkpoint["deferred"] = deferred_map
                save_checkpoint(checkpoint)
                deferred_ids_this_pass.append(issue_id)
                budget_exhausted_this_pass = True
                break  # same account-wide-budget short-circuit reasoning as synthesis above

            rec = {
                "plan": plan.model_dump(),
                "judge_score": judge_score,
                "error": triage_error,
            }
            results[issue_id] = rec
            checkpoint["done"][issue_id] = rec
            deferred_map.pop(issue_id, None)  # succeeded -- no longer needs the "hard" lane
            checkpoint["deferred"] = deferred_map
            save_checkpoint(checkpoint)
            recorded_ids_this_pass.append(issue_id)

        total_synthesis_recorded += n_synthesis_recorded
        total_judge_recorded += n_judge_recorded

        # --- Per-pass summary (always printed — real numbers for THIS pass) ---
        print(f"\n=== PASS {pass_num} SUMMARY ===")
        print(f"Recorded this pass:  {len(recorded_ids_this_pass)} {recorded_ids_this_pass}")
        print(f"Deferred this pass:  {len(deferred_ids_this_pass)} {deferred_ids_this_pass}")
        print(f"Budget exhausted this pass: {budget_exhausted_this_pass}")
        print(f"Total done (all-time):     {len(results)}/{len(issues)}")
        print(f"Total deferred (all-time): {len(deferred_map)} {sorted(deferred_map.keys())}")

        completed = [v for v in results.values() if v.get("judge_score") is not None]
        is_fully_complete = len(completed) == len(issues)

        if is_fully_complete:
            break  # done -- fall through to the overall-stats block below

        # Stall detection applies uniformly, regardless of whether this pass hit TPD --
        # see the bug this replaced, below.
        made_progress = cassette.stats()["entries"] > entries_before_pass
        consecutive_stalls = 0 if made_progress else consecutive_stalls + 1
        if consecutive_stalls >= MAX_CONSECUTIVE_STALLS:
            logger.error(
                "%d consecutive passes recorded zero new cassette entries -- something "
                "other than 'budget hasn't recovered yet' is wrong. Stopping rather than "
                "looping forever.", consecutive_stalls,
            )
            sys.exit(1)

        if not budget_exhausted_this_pass:
            # BUG FIXED 2026-08-25: this used to unconditionally `break` here, on the
            # assumption that "this pass never hit TPD" meant "everything not-done
            # failed for a real, non-retryable reason." That's wrong -- a pass only
            # ever works ONE of fresh_queue/hard_queue (see queue-building above), so
            # a pass that fully clears a non-empty fresh_queue without hitting TPD
            # leaves the hard_queue (previously-deferred items) completely untried,
            # and this `break` then reported them as permanently missing a judge score
            # -- a false "INCOMPLETE" on a run that had a fully working key with
            # headroom to spare. Caught live: a fresh Groq key cleared all 4 fresh
            # items with zero TPD hits, and the old code declared the recording
            # incomplete and exited 1 with 26 never-attempted-this-invocation items
            # still sitting in the hard_queue. Loop back immediately instead (no sleep
            # needed -- there's headroom) so the next pass picks up the hard_queue. The
            # stall detection above still catches a genuine dead end: if every
            # remaining item fails for a real non-TPD reason, no pass makes any new
            # cassette-entry progress and MAX_CONSECUTIVE_STALLS fires.
            continue

        wait_s = _parse_tpd_retry_seconds(last_tpd_exc) if last_tpd_exc is not None else None
        if wait_s is None:
            logger.error(
                "TPD budget exhausted but couldn't parse a 'try again in ...' duration "
                "out of Groq's error -- refusing to guess a sleep length. Raw error: %s",
                last_tpd_exc,
            )
            sys.exit(1)

        wait_total = wait_s + TPD_WAIT_BUFFER
        logger.info(
            "Pass %d exhausted the daily token budget (%d recorded, %d deferred this "
            "pass). Groq says headroom recovers in %.0fs; sleeping %.0fs (+%.0fs buffer) "
            "then resuming automatically -- no manual re-invocation needed.",
            pass_num, len(recorded_ids_this_pass), len(deferred_ids_this_pass),
            wait_s, wait_total, TPD_WAIT_BUFFER,
        )
        time.sleep(wait_total)
        # loop continues -> next pass re-derives done_ids/deferred_map/queue from
        # `checkpoint`, which was updated in place by save_checkpoint() calls above.

    # --- Overall stats (whole invocation — every pass, not just the last one) ---
    # Gate the "RECORDING COMPLETE" header on genuine completeness (every issue has a
    # judge score), not just "at least one does" -- callers (including CI or a human
    # grepping the log) look for that exact string as the done signal.
    completed = [v for v in results.values() if v.get("judge_score") is not None]
    is_fully_complete = len(completed) == len(issues)
    if completed:
        scores = [JudgeScore.model_validate(v["judge_score"]) for v in completed]
        total_scores = [s.total() for s in scores]
        print(f"\n=== {'RECORDING COMPLETE' if is_fully_complete else 'RECORDING INCOMPLETE'} ===")
        print(f"Passes run:          {pass_num}")
        print(f"Issues processed:   {len(results)}/{len(issues)}")
        print(f"Skipped (cached at start): {n_skipped_at_start}")
        print(f"Synthesis recorded (this invocation): {total_synthesis_recorded}")
        print(f"Judge recorded (this invocation):     {total_judge_recorded}")
        print(f"Cassette entries:   {cassette.stats()['entries']}")
        print(f"Judge score (mean): {np.mean(total_scores):.2f}/15 ({np.mean(total_scores)/15*100:.0f}%)")

        # Per-repo breakdown
        by_repo: dict[str, list] = {}
        for issue in issues:
            r = results.get(issue["id"], {})
            if r.get("judge_score"):
                by_repo.setdefault(issue["repo"], []).append(
                    JudgeScore.model_validate(r["judge_score"]).total()
                )
        for repo, repo_scores in by_repo.items():
            print(f"  {repo}: n={len(repo_scores)}, mean={np.mean(repo_scores):.2f}/15")
    else:
        print(f"\n=== RECORDING INCOMPLETE — no judge scores ===")
        print(f"Cassette entries: {cassette.stats()['entries']}")

    # --- Completeness assertion ---
    # The pass loop above can silently leave an issue with judge_score=None without ever
    # hitting one of the sys.exit(1) paths above: a non-TPD/non-connection synthesis
    # exception, or a non-TPD/non-connection/non-rate-limit judge failure after 6
    # retries, both just `continue`/`break` to the next issue. Those gaps are invisible
    # here in this script's own output -- they only surface later as a CassetteMissError
    # in run_eval.py's replay, in a different job step (or a different day), with no
    # link back to this run. Assert it here instead: every issue must have a non-None
    # judge_score, or this is not a usable recording.
    missing = [iid for iid in (i["id"] for i in issues) if results.get(iid, {}).get("judge_score") is None]
    if missing:
        print(f"\n=== INCOMPLETE: {len(missing)}/{len(issues)} issues missing a judge score ===")
        print(f"Missing: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        sys.exit(1)


if __name__ == "__main__":
    main()
