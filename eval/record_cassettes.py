from __future__ import annotations

"""One-time recording pass: runs the full TriageIQ pipeline + judge over
eval/eval_set.jsonl and saves ALL LLM interactions to
eval/cassettes/eval_cassette.json.

Run ONCE locally with live GROQ_API_KEY set.  CI never runs this script.

Usage:
    python eval/record_cassettes.py

Exit codes:
    0 — all issues recorded successfully
    0 — recording stopped cleanly due to Groq TPD (tokens-per-day) limit;
        cassette has partial entries, script reports what was done
    1 — unexpected error
"""

import hashlib
import json
import logging
import numpy as np
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
from triage_iq.model_config import TRIAGE_MODEL
from triage_iq.models.component_classifier import load_classifier
from triage_iq.evaluation.triage_eval import DIMENSION_MAX, JudgeScore, TriageJudge
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant, TruncatedCompletionError

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


def _compute_prompt_hash() -> str:
    """Fingerprint of the exact system prompt + few-shot messages this run will send,
    mirroring TriageAssistant._call_llm_verbose's prompt-selection (triage.py) so a prompt
    change is caught by the checkpoint the same way a model change is caught by
    TRIAGE_MODEL. Does not cover use_structured_output's SYSTEM_PROMPT/_PROSE branch --
    that branch is only reachable when TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1, which this
    script never sets."""
    from triage_iq.prompts.triage_prompt import (
        SYSTEM_PROMPT_LEGACY,
        SYSTEM_PROMPT_PROSE,
        build_few_shot_examples,
        build_few_shot_examples_legacy,
    )

    if os.environ.get("TRIAGE_PROMPT_INCLUDE_ATTRIBUTION") == "1":
        system_prompt = SYSTEM_PROMPT_PROSE
        few_shots = build_few_shot_examples()
    else:
        system_prompt = SYSTEM_PROMPT_LEGACY
        few_shots = build_few_shot_examples_legacy()
    payload = json.dumps({"system_prompt": system_prompt, "few_shots": few_shots}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _checkpoint_key(issue_id: str, model: str, prompt_hash: str) -> str:
    return f"{issue_id}::{model}::{prompt_hash}"


def _record_done(checkpoint: dict, issue_id: str, record: dict, model: str, prompt_hash: str) -> dict:
    """Tag and store a done-entry under a composite (issue_id, model, prompt_hash) key so a
    model or prompt change can never be silently mistaken for "already recorded". Fixes the
    2026-08-31 incident: a stale checkpoint recorded under the retired llama-3.1-8b-instant,
    keyed by bare issue_id, was silently accepted as complete for openai/gpt-oss-120b and
    printed "RECORDING COMPLETE" with the old model's judge means after zero live calls."""
    tagged = {**record, "issue_id": issue_id, "model": model, "prompt_hash": prompt_hash}
    checkpoint["done"][_checkpoint_key(issue_id, model, prompt_hash)] = tagged
    return tagged


def load_checkpoint(current_model: str, current_prompt_hash: str) -> tuple[dict, dict[str, dict]]:
    """Load the checkpoint file and partition its done-entries into those matching the
    currently configured (model, prompt_hash) vs. everything else. Any entry missing a
    model/prompt_hash tag (i.e. written before this composite-key fix) is treated as
    untrustworthy and halts the run -- it cannot be proven to belong to the current model,
    which is exactly the silent-reuse failure this keying scheme exists to prevent."""
    if not CHECKPOINT_PATH.exists():
        return {"done": {}}, {}

    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    all_done = data.get("done", {})

    current_done: dict[str, dict] = {}
    stale_by_model: dict[str, int] = {}
    for key, rec in all_done.items():
        rec_model = rec.get("model")
        rec_hash = rec.get("prompt_hash")
        if rec_model is None or rec_hash is None:
            logger.error(
                "STOP: recording_checkpoint.json entry %r has no model/prompt_hash tag -- "
                "it predates the (issue_id, model, prompt_hash) keying fix and cannot be "
                "trusted to belong to the currently configured model (%s). Refusing to "
                "resume silently. Archive or delete this checkpoint file to start fresh "
                "under the current model, or manually re-tag its entries if you can confirm "
                "which model actually recorded them.",
                key, current_model,
            )
            sys.exit(1)
        if rec_model == current_model and rec_hash == current_prompt_hash:
            current_done[rec["issue_id"]] = rec
        else:
            stale_by_model[f"{rec_model}@{rec_hash[:8]}"] = (
                stale_by_model.get(f"{rec_model}@{rec_hash[:8]}", 0) + 1
            )

    logger.info(
        "Checkpoint recorded under configured model=%s prompt_hash=%s: %d issue(s) already done.",
        current_model, current_prompt_hash, len(current_done),
    )
    if stale_by_model:
        logger.warning(
            "Checkpoint also holds %d entries recorded under a DIFFERENT model/prompt -- "
            "ignored for resume, not deleted: %s",
            sum(stale_by_model.values()), stale_by_model,
        )
    return data, current_done


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

    # Load cassette in record mode (strict=False, allow_record=True -- the one sanctioned
    # recording pass; see eval/cassette.py's class docstring for why these are separate).
    cassette = CassettePlayer(CASSETTE_PATH, strict=False, allow_record=True)
    logger.info("Cassette: %d entries already recorded at %s", cassette.stats()["entries"], CASSETTE_PATH)

    current_model = TRIAGE_MODEL
    current_prompt_hash = _compute_prompt_hash()
    logger.info("Configured for this run: model=%s prompt_hash=%s", current_model, current_prompt_hash)

    # Load checkpoint, filtered to entries matching the CURRENT model+prompt only.
    checkpoint, current_done = load_checkpoint(current_model, current_prompt_hash)
    # Exclude tpd_hit entries so they are retried — their synthesis is cached, only the judge reruns.
    done_ids = {k for k, v in current_done.items() if not v.get("tpd_hit")}
    n_tpd_retry = len(current_done) - len(done_ids)
    logger.info("Checkpoint: %d issues already processed under current model+prompt (%d tpd_hit will retry)",
                len(done_ids), n_tpd_retry)

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

    n_synthesis_recorded = 0
    n_judge_recorded = 0
    n_skipped = 0
    results: dict[str, dict] = {}

    for i, issue in enumerate(issues):
        issue_id = issue["id"]
        repo = issue["repo"]

        if issue_id in done_ids:
            logger.info("[%d/%d] %s — skipped (checkpoint)", i + 1, len(issues), issue_id)
            results[issue_id] = current_done[issue_id]
            n_skipped += 1
            continue

        if i > 0 and issue_id not in done_ids:
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
            # 2026-08-30: TruncatedCompletionError is caught INSIDE _call_llm_verbose (Part
            # B3's degrade path, PR #113) and converted to a clean fallback plan before it
            # ever reaches this caller -- confirmed by a zero-quota dry run
            # (scripts/record_cassettes_dry_run_check.py) that this except-TruncatedCompletion
            # -Error block below never fires anymore. Without this check, a truncated (or
            # otherwise degraded) completion would silently proceed to judging and get
            # checkpointed as "done" with no synthesis entry ever reaching the cassette
            # (_call_llm_verbose returns before cache.set() on that path) -- a permanent,
            # silent mismatch between the checkpoint (says done) and the cassette (has
            # nothing for that issue), invisible until some later replay hits a
            # CassetteMissError with no link back to this run. Same failure shape as this
            # engagement's other instrumentation-found-after-the-spend incidents; catch it
            # here, at the one place that knows llm_status, not downstream.
            llm_status = meta.get("llm_status")
            if llm_status not in ("ok", "parse_retry_succeeded"):
                logger.error(
                    "STOP: synthesis degraded (llm_status=%s) after %d synthesis calls. "
                    "Cassette has %d entries. This issue is NOT marked done -- re-running "
                    "will retry it, not skip it.",
                    llm_status, n_synthesis_recorded, cassette.stats()["entries"],
                )
                save_checkpoint({"done": checkpoint.get("done", {})})
                print("\n=== SYNTHESIS DEGRADED (not a genuine completion) ===")
                print(f"Issue: {issue_id}")
                print(f"llm_status={llm_status}")
                print(f"Synthesis recorded before stop: {n_synthesis_recorded}")
                print(f"Cassette entries: {cassette.stats()['entries']}")
                sys.exit(1)
            n_synthesis_recorded += 1
            logger.info("  synthesis → %s (cache_hit=%s)", plan.predicted_component, meta.get("llm_cache_hit"))
        except TruncatedCompletionError as exc:
            # 2026-08-28: a truncated completion means max_tokens is too small for this
            # model -- every subsequent issue is likely to hit the same wall, so this stops
            # the run loudly rather than silently skipping entries (and burning TPD budget)
            # under a config already known to be broken. Raise max_tokens and restart --
            # the resume mechanism picks up from recording_checkpoint.json as usual.
            logger.error(
                "STOP: completion truncated after %d synthesis calls "
                "(completion_tokens=%d, max_tokens=%d). Cassette has %d entries. "
                "Raise max_tokens and re-run -- resume will continue from checkpoint.",
                n_synthesis_recorded, exc.completion_tokens, exc.max_tokens,
                cassette.stats()["entries"],
            )
            save_checkpoint({"done": checkpoint.get("done", {})})
            print("\n=== TRUNCATED COMPLETION ===")
            print(f"Issue: {issue_id}")
            print(f"completion_tokens={exc.completion_tokens} max_tokens={exc.max_tokens}")
            print(f"Synthesis recorded before stop: {n_synthesis_recorded}")
            print(f"Cassette entries: {cassette.stats()['entries']}")
            sys.exit(1)  # incomplete recording -- see TPD-exit comment below
        except Exception as exc:
            # 2026-08-30: was `if _is_tpd_error(exc)` only -- too narrow. _groq_completion
            # already retries a RateLimitError internally (6 attempts, exponential
            # backoff) before this exception ever reaches here, so by the time we see one,
            # it has already survived real backoff and is not a transient blip. But its
            # message does not reliably contain "daily"/"tpd" (Groq's rate-limit body uses
            # 'code': 'rate_limit_exceeded' whether the underlying cause is a per-minute or
            # per-day ceiling -- confirmed directly from this session's captured raw Groq
            # error bodies). Under the old check, a sustained-but-not-explicitly-"daily"
            # rate limit fell through to the generic except branch below and got recorded
            # as a permanent per-issue failure (checkpointed done, never retried) -- the
            # same silent-mismatch shape the llm_status check above this block exists to
            # prevent, just for a different trigger. _is_rate_limit_error covers both.
            if _is_rate_limit_error(exc):
                logger.error(
                    "STOP: Groq rate limit (TPD or sustained TPM) hit after %d synthesis "
                    "calls. Cassette has %d entries. Groq error: %s",
                    n_synthesis_recorded, cassette.stats()["entries"], exc,
                )
                save_checkpoint({"done": checkpoint.get("done", {})})
                print(f"\n=== TPD HIT ===")
                print(f"Synthesis recorded: {n_synthesis_recorded}")
                print(f"Judge recorded: {n_judge_recorded}")
                print(f"Cassette entries: {cassette.stats()['entries']}")
                # exit(1): this recording is INCOMPLETE. exit(0) here previously let a
                # partial cassette look like a clean run to the shell -- CI's "Update
                # eval baseline" step would then hit CassetteMissError on the first
                # un-recorded issue with no indication the recording itself was short.
                # Local resumable use is unaffected: re-running still resumes from
                # recording_checkpoint.json regardless of this process's exit code.
                sys.exit(1)
            if _is_connection_error(exc):
                logger.error(
                    "STOP: connection lost after %d synthesis calls. "
                    "Cassette has %d entries. Error: %s",
                    n_synthesis_recorded, cassette.stats()["entries"], exc,
                )
                save_checkpoint({"done": checkpoint.get("done", {})})
                print("\n=== CONNECTION LOST ===")
                print(f"Synthesis recorded: {n_synthesis_recorded}")
                print(f"Judge recorded: {n_judge_recorded}")
                print(f"Cassette entries: {cassette.stats()['entries']}")
                sys.exit(1)  # incomplete recording -- see TPD-exit comment above
            logger.warning("  synthesis FAILED: %s", exc)
            triage_error = str(exc)

        if plan is None:
            results[issue_id] = {"error": triage_error, "plan": None, "judge_score": None}
            _record_done(checkpoint, issue_id, results[issue_id], current_model, current_prompt_hash)
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
                    break  # genuine daily limit — stop outer loop below
                if _is_connection_error(exc):
                    logger.error("STOP: connection lost during judging: %s", exc)
                    save_checkpoint(checkpoint)
                    print("\n=== CONNECTION LOST (during judge) ===")
                    print(f"Synthesis recorded: {n_synthesis_recorded}")
                    print(f"Judge recorded: {n_judge_recorded}")
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
            logger.error(
                "STOP: Groq TPD (daily quota) hit during judging after %d judge calls. "
                "Cassette has %d entries. Groq error: %s",
                n_judge_recorded, cassette.stats()["entries"], _judge_exc,
            )
            results[issue_id] = {
                "plan": plan.model_dump(),
                "judge_score": None,
                "tpd_hit": True,
            }
            _record_done(checkpoint, issue_id, results[issue_id], current_model, current_prompt_hash)
            save_checkpoint(checkpoint)
            print(f"\n=== TPD HIT (during judge) ===")
            print(f"Synthesis recorded: {n_synthesis_recorded}")
            print(f"Judge recorded: {n_judge_recorded}")
            print(f"Cassette entries: {cassette.stats()['entries']}")
            sys.exit(1)  # incomplete recording -- see TPD-exit comment above

        rec = {
            "plan": plan.model_dump(),
            "judge_score": judge_score,
            "error": triage_error,
        }
        results[issue_id] = rec
        _record_done(checkpoint, issue_id, rec, current_model, current_prompt_hash)
        save_checkpoint(checkpoint)

    # --- Summary ---
    completed = [v for v in results.values() if v.get("judge_score") is not None]
    if completed and n_synthesis_recorded > 0:
        scores = [JudgeScore.model_validate(v["judge_score"]) for v in completed]
        total_scores = [s.total() for s in scores]
        print(f"\n=== RECORDING COMPLETE ===")
        print(f"Model:              {current_model}")
        print(f"Prompt hash:        {current_prompt_hash}")
        print(f"Issues processed:   {len(results)}")
        print(f"Skipped (cached):   {n_skipped}")
        print(f"Synthesis recorded: {n_synthesis_recorded}")
        print(f"Judge recorded:     {n_judge_recorded}")
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
    elif completed and n_synthesis_recorded == 0:
        # All entries came from the checkpoint (0 live synthesis calls this invocation).
        # Never report this as "RECORDING COMPLETE" -- that phrase must mean this run did
        # genuine live work, not that it silently reused whatever a checkpoint claimed. If
        # every issue really is done, inspect the checkpoint/cassette directly rather than
        # trusting this run's exit.
        print(f"\n=== NOT RECORDING COMPLETE (zero live synthesis calls this run) ===")
        print(f"Model:            {current_model}")
        print(f"Prompt hash:      {current_prompt_hash}")
        print(f"Issues skipped (checkpoint): {n_skipped}/{len(issues)}")
        print(f"Cassette entries: {cassette.stats()['entries']}")
        sys.exit(1)
    else:
        print(f"\n=== RECORDING INCOMPLETE — no judge scores ===")
        print(f"Model:            {current_model}")
        print(f"Prompt hash:      {current_prompt_hash}")
        print(f"Cassette entries: {cassette.stats()['entries']}")

    # --- Completeness assertion ---
    # The loop above can silently leave an issue with judge_score=None without ever
    # hitting one of the sys.exit(1) paths above: a non-TPD/non-connection synthesis
    # exception (line ~253) or a non-TPD/non-connection/non-rate-limit judge failure
    # after 6 retries (line ~314) both just `continue`/`break` to the next issue. Those
    # gaps are invisible here in this script's own output -- they only surface later as
    # a CassetteMissError in run_eval.py's replay, in a different job step (or a
    # different day), with no link back to this run. Assert it here instead: every
    # issue must have a non-None judge_score, or this is not a usable recording.
    missing = [iid for iid in (i["id"] for i in issues) if results.get(iid, {}).get("judge_score") is None]
    if missing:
        print(f"\n=== INCOMPLETE: {len(missing)}/{len(issues)} issues missing a judge score ===")
        print(f"Missing: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        sys.exit(1)


if __name__ == "__main__":
    main()
