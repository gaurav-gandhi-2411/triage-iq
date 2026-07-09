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
from triage_iq.evaluation.triage_eval import DIMENSION_MAX, JudgeScore, TriageJudge
from triage_iq.models.component_classifier import TFIDFComponentClassifier
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


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        logger.info("Checkpoint: %d issues already recorded", len(data.get("done", {})))
        return data
    return {"done": {}}


def save_checkpoint(data: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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
            classifier = TFIDFComponentClassifier.load(
                str(models_dir / f"component_classifier_{slug}.pkl")
            )
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
            results[issue_id] = checkpoint["done"][issue_id]
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
            n_synthesis_recorded += 1
            logger.info("  synthesis → %s (cache_hit=%s)", plan.predicted_component, meta.get("llm_cache_hit"))
        except Exception as exc:
            if _is_tpd_error(exc):
                logger.error(
                    "STOP: Groq TPD (daily quota) hit after %d synthesis calls. "
                    "Cassette has %d entries. Groq error: %s",
                    n_synthesis_recorded, cassette.stats()["entries"], exc,
                )
                save_checkpoint({"done": checkpoint.get("done", {})})
                print(f"\n=== TPD HIT ===")
                print(f"Synthesis recorded: {n_synthesis_recorded}")
                print(f"Judge recorded: {n_judge_recorded}")
                print(f"Cassette entries: {cassette.stats()['entries']}")
                sys.exit(0)
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
                sys.exit(0)
            logger.warning("  synthesis FAILED: %s", exc)
            triage_error = str(exc)

        if plan is None:
            results[issue_id] = {"error": triage_error, "plan": None, "judge_score": None}
            checkpoint["done"][issue_id] = results[issue_id]
            save_checkpoint(checkpoint)
            continue

        # --- Judge ---
        time.sleep(JUDGE_DELAY)
        plan_json = json.dumps(plan.model_dump(), ensure_ascii=False)
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
                    sys.exit(0)
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
            checkpoint["done"][issue_id] = results[issue_id]
            save_checkpoint(checkpoint)
            print(f"\n=== TPD HIT (during judge) ===")
            print(f"Synthesis recorded: {n_synthesis_recorded}")
            print(f"Judge recorded: {n_judge_recorded}")
            print(f"Cassette entries: {cassette.stats()['entries']}")
            sys.exit(0)

        rec = {
            "plan": plan.model_dump(),
            "judge_score": judge_score,
            "error": triage_error,
        }
        results[issue_id] = rec
        checkpoint["done"][issue_id] = rec
        save_checkpoint(checkpoint)

    # --- Summary ---
    completed = [v for v in results.values() if v.get("judge_score") is not None]
    if completed:
        scores = [JudgeScore.model_validate(v["judge_score"]) for v in completed]
        total_scores = [s.total() for s in scores]
        print(f"\n=== RECORDING COMPLETE ===")
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
    else:
        print(f"\n=== RECORDING INCOMPLETE — no judge scores ===")
        print(f"Cassette entries: {cassette.stats()['entries']}")


if __name__ == "__main__":
    main()
