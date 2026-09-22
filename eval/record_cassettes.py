from __future__ import annotations

"""One-time recording pass: runs the full TriageIQ pipeline + judge over
eval/eval_set.jsonl and saves ALL LLM interactions to
eval/cassettes/eval_cassette.json.

Run ONCE locally with live GROQ_API_KEY set.  CI never runs this script.

Usage:
    python eval/record_cassettes.py                  # full: synthesis + judge per issue
    python eval/record_cassettes.py --mode synthesis  # Groq only, no Ollama load at all
    python eval/record_cassettes.py --mode judge      # local judge only, replays stored plans

2026-09-06 (ADR-0060): split synthesis from judging because the local Ollama judge (qwen3:8b)
sits resident in RAM+VRAM through every multi-minute Groq quota wait for zero benefit --
~5 GB RAM / ~5.5 GB VRAM held idle on an 8 GB card, for a judge call that takes seconds once
synthesis is actually done. --mode synthesis never imports TriageJudge or touches Ollama at
all; --mode judge does a single batch pass over already-synthesized entries and loads no
classifier/predictor/retriever. --mode full (the default) preserves the original combined
per-issue behavior for any caller that doesn't care about the split.

Exit codes:
    0 — all issues recorded successfully
    0 — recording stopped cleanly due to Groq TPD (tokens-per-day) limit;
        cassette has partial entries, script reports what was done
    1 — unexpected error
"""

import argparse
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

import artifact_fingerprint
from cassette import CassettePlayer
from frozen_retriever import build_frozen_retrievers
from triage_iq.model_config import TRIAGE_MODEL
from triage_iq.models.component_classifier import load_classifier
from triage_iq.evaluation.triage_eval import DIMENSION_MAX, JudgeScore
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant, TruncatedCompletionError

# TriageJudge (and therefore Ollama) is imported lazily, only inside _build_judge() -- a
# --mode synthesis run must never import it, let alone construct one (2026-09-06, ADR-0060).

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
CASSETTE_PATH = ROOT / "eval" / "cassettes" / "eval_cassette.json"
CHECKPOINT_PATH = ROOT / "eval" / "cassettes" / "recording_checkpoint.json"

# ADR-0059: the 2026-09-05 incident that made artifact fingerprinting necessary was this
# exact checkout being resolved as ROOT by a script actually invoked from it (ROOT is always
# `Path(__file__).parent.parent`, so an accidental main-checkout invocation is
# indistinguishable from a legitimate one by any check confined to THIS worktree's own code).
# The general fix is the expected-artifact-hash gate below, which catches ANY wrong-checkout
# or wrong-artifact-version mistake, not just this one -- this constant is a cheap, specific
# tripwire for the exact incident pattern, defense in depth on top of that general gate.
_KNOWN_OTHER_CHECKOUT = Path(r"C:\Users\gaura\ml-projects\triage-iq")

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
    """Fingerprint of the exact system prompt + few-shot messages + wire JSON schema this
    run will send, mirroring TriageAssistant._call_llm_verbose's prompt-selection
    (triage.py) so a prompt OR schema change is caught by the checkpoint the same way a
    model change is caught by TRIAGE_MODEL. Does not cover use_structured_output's
    SYSTEM_PROMPT/_PROSE branch -- that branch is only reachable when
    TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1, which this script never sets.

    2026-09-03 (ADR-0054/0055): the wire schema (TriagePlan's response_format, via
    _build_triage_plan_response_format) was NOT part of this hash until now -- the
    schema-reduction fix changed which fields Groq's strict decoding demands without
    touching the prompt text at all, which this hash would have silently missed,
    letting a resume reuse pre-schema-change checkpoint entries as if they were
    recorded under the new contract. They are not comparable: a synthesis call made
    under the old (18-required-field) schema is a different request than one made
    under the new (11-required-field) schema, even with identical prompt text."""
    from triage_iq.prompts.triage_prompt import (
        SYSTEM_PROMPT_LEGACY,
        SYSTEM_PROMPT_PROSE,
        build_few_shot_examples,
        build_few_shot_examples_legacy,
    )
    from triage_iq.models.triage import _TRIAGE_PLAN_RESPONSE_FORMAT

    if os.environ.get("TRIAGE_PROMPT_INCLUDE_ATTRIBUTION") == "1":
        system_prompt = SYSTEM_PROMPT_PROSE
        few_shots = build_few_shot_examples()
    else:
        system_prompt = SYSTEM_PROMPT_LEGACY
        few_shots = build_few_shot_examples_legacy()
    payload = json.dumps(
        {
            "system_prompt": system_prompt,
            "few_shots": few_shots,
            "wire_schema": _TRIAGE_PLAN_RESPONSE_FORMAT,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _checkpoint_key(issue_id: str, model: str, prompt_hash: str, artifact_hash: str) -> str:
    return f"{issue_id}::{model}::{prompt_hash}::{artifact_hash}"


def _record_done(
    checkpoint: dict, issue_id: str, record: dict, model: str, prompt_hash: str, artifact_hash: str,
) -> dict:
    """Tag and store a done-entry under a composite (issue_id, model, prompt_hash,
    artifact_hash) key so a model, prompt, OR classifier/predictor/index/conformal-store
    change can never be silently mistaken for "already recorded". model/prompt_hash fixes the
    2026-08-31 incident (a stale checkpoint recorded under the retired llama-3.1-8b-instant,
    keyed by bare issue_id, was silently accepted as complete for openai/gpt-oss-120b).
    artifact_hash fixes the 2026-09-05 incident (ADR-0059): a classifier retrain changed what
    every prompt actually said without changing the model name or prompt/schema text at all --
    the two dimensions this key already covered -- so it was invisible to this exact
    resume-safety mechanism until now."""
    tagged = {
        **record, "issue_id": issue_id, "model": model,
        "prompt_hash": prompt_hash, "artifact_hash": artifact_hash,
    }
    checkpoint["done"][_checkpoint_key(issue_id, model, prompt_hash, artifact_hash)] = tagged
    return tagged


def load_checkpoint(
    current_model: str, current_prompt_hash: str, current_artifact_hash: str,
) -> tuple[dict, dict[str, dict]]:
    """Load the checkpoint file and partition its done-entries into those matching the
    currently configured (model, prompt_hash, artifact_hash) vs. everything else. Any entry
    missing one of these three tags (i.e. written before that tag existed) is treated as
    untrustworthy and halts the run -- it cannot be proven to belong to the current
    configuration, which is exactly the silent-reuse failure this keying scheme exists to
    prevent."""
    if not CHECKPOINT_PATH.exists():
        return {"done": {}}, {}

    data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    all_done = data.get("done", {})

    current_done: dict[str, dict] = {}
    stale_by_config: dict[str, int] = {}
    for key, rec in all_done.items():
        rec_model = rec.get("model")
        rec_hash = rec.get("prompt_hash")
        rec_artifact_hash = rec.get("artifact_hash")
        if rec_model is None or rec_hash is None or rec_artifact_hash is None:
            logger.error(
                "STOP: recording_checkpoint.json entry %r is missing a model/prompt_hash/"
                "artifact_hash tag -- it predates the composite-key fix (ADR-0058/ADR-0059) "
                "and cannot be trusted to belong to the currently configured model+artifacts "
                "(%s, artifact_hash=%s). Refusing to resume silently. Archive or delete this "
                "checkpoint file to start fresh, or manually re-tag its entries if you can "
                "confirm what actually recorded them.",
                key, current_model, current_artifact_hash,
            )
            sys.exit(1)
        if (
            rec_model == current_model
            and rec_hash == current_prompt_hash
            and rec_artifact_hash == current_artifact_hash
        ):
            current_done[rec["issue_id"]] = rec
        else:
            config_id = f"{rec_model}@{rec_hash[:8]}@{rec_artifact_hash[:8]}"
            stale_by_config[config_id] = stale_by_config.get(config_id, 0) + 1

    logger.info(
        "Checkpoint recorded under configured model=%s prompt_hash=%s artifact_hash=%s: "
        "%d issue(s) already done.",
        current_model, current_prompt_hash, current_artifact_hash, len(current_done),
    )
    if stale_by_config:
        logger.warning(
            "Checkpoint also holds %d entries recorded under a DIFFERENT model/prompt/"
            "artifact configuration -- ignored for resume, not deleted: %s",
            sum(stale_by_config.values()), stale_by_config,
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


def _resolve_and_verify_artifacts() -> dict[str, str]:
    """ADR-0059: print exactly which checkout and which artifact files this run resolved,
    and refuse to proceed unless they match the committed expected-hash file. This is the
    check that would have caught the 2026-09-05 incident immediately, at the top of the run,
    instead of hours later as an unexplained CassetteMissError on replay."""
    resolved_root = ROOT.resolve()
    logger.info("Resolved ROOT for this invocation: %s", resolved_root)
    try:
        other = _KNOWN_OTHER_CHECKOUT.resolve()
    except OSError:
        other = None
    if other is not None and resolved_root == other:
        logger.error(
            "STOP: this invocation resolved ROOT to %s -- the known OTHER TriageIQ checkout, "
            "not this worktree. That is exactly the 2026-09-05 incident shape (a script "
            "executed against the wrong checkout's classifier/artifacts). Run this script "
            "from the intended worktree, not %s.",
            resolved_root, resolved_root,
        )
        sys.exit(1)

    current_hashes = artifact_fingerprint.compute_artifact_hashes(ROOT)
    logger.info("Model artifacts resolved for this run:")
    for rel, h in sorted(current_hashes.items()):
        abs_path = (ROOT / rel).resolve()
        logger.info("  %s  %s  (%s)", h[:16], rel, abs_path)
        if h == "MISSING":
            logger.error("STOP: artifact missing on disk: %s", abs_path)
            sys.exit(1)

    expected = artifact_fingerprint.load_expected_hashes(ROOT)
    expected_path = artifact_fingerprint.expected_hashes_path(ROOT)
    if expected is None:
        logger.error(
            "STOP: %s does not exist. This file is the explicit, committed statement of "
            "which artifacts this branch intends to record against (ADR-0059) -- a missing "
            "file is refused, never treated as 'no expectation, proceed anyway' (rule 98a: "
            "fail closed, not open). Create it deliberately "
            "(artifact_fingerprint.save_expected_hashes) once you've confirmed the resolved "
            "artifacts above are the ones you actually intend to record against.",
            expected_path,
        )
        sys.exit(1)

    mismatches = artifact_fingerprint.diff_against_expected(current_hashes, expected)
    if mismatches:
        logger.error(
            "STOP: resolved artifacts do not match %s. Refusing to record -- a mismatch "
            "here means either the wrong checkout executed this script, or the artifacts "
            "changed without a deliberate update to the expected-hash file:\n%s",
            expected_path, "\n".join(mismatches),
        )
        sys.exit(1)

    logger.info("Artifact check: resolved artifacts match %s -- proceeding.", expected_path)
    return current_hashes


_JUDGE_EXCLUDED_PLAN_FIELDS = {"declared_attribution", "abstention_status"}
# exclude={"declared_attribution", "abstention_status"}: must match run_eval.py's plan_json
# exactly (same fields) so the judge cache key computed here at record time is the same one
# run_eval.py's replay looks up later. Both fields are unconditional on TriagePlan (always
# serialize, even as None) but never populated by this eval harness -- see run_eval.py's
# ADR-0020/ADR-0021 comment for detail.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode", choices=("full", "synthesis", "judge"), default="full",
        help="full: synthesis+judge per issue (original combined behavior). synthesis: Groq "
             "calls only -- never imports TriageJudge, never touches Ollama. judge: local-"
             "judge-only batch pass over already-synthesized (judge_pending) entries -- loads "
             "no classifier/predictor/retriever, makes zero Groq calls.",
    )
    return parser.parse_args()


def _load_common(mode: str) -> tuple[str, list[dict], CassettePlayer, str, str, str, dict, dict[str, dict]]:
    """Shared setup for every mode: GROQ key (skipped for judge -- it makes no Groq calls),
    artifact verification, eval set, cassette, and the filtered checkpoint."""
    groq_key = ""
    if mode != "judge":
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            logger.error("GROQ_API_KEY not set. Add it to .env or export it.")
            sys.exit(1)

    # ADR-0059: judge mode makes no Groq calls and never touches the classifier, but it
    # still needs to know WHICH checkpoint entries belong to the current configuration --
    # running the same artifact gate here (rather than skipping it) means a classifier
    # retrained between the synthesis pass and the judge pass is caught explicitly instead
    # of the judge pass silently scoring plans whose provenance no longer matches anything
    # "current".
    current_artifact_hashes = _resolve_and_verify_artifacts()
    current_artifact_hash = artifact_fingerprint.combined_hash(current_artifact_hashes)

    issues = load_eval_set()
    logger.info("Eval set: %d issues (%s)",
                len(issues),
                ", ".join(f"{r}: {sum(1 for i in issues if i['repo']==r)}"
                          for r in sorted(set(i['repo'] for i in issues))))

    cassette = CassettePlayer(CASSETTE_PATH, strict=False, allow_record=True)
    logger.info("Cassette: %d entries already recorded at %s", cassette.stats()["entries"], CASSETTE_PATH)

    current_model = TRIAGE_MODEL
    current_prompt_hash = _compute_prompt_hash()
    logger.info(
        "Configured for this run (mode=%s): model=%s prompt_hash=%s artifact_hash=%s",
        mode, current_model, current_prompt_hash, current_artifact_hash,
    )

    checkpoint, current_done = load_checkpoint(current_model, current_prompt_hash, current_artifact_hash)
    return groq_key, issues, cassette, current_model, current_prompt_hash, current_artifact_hash, checkpoint, current_done


def _load_models(groq_key: str, cassette: CassettePlayer) -> dict[str, dict]:
    """Classifier + resolution predictor + train_df + TriageAssistant per repo. Only called
    by synthesis/full mode -- judge mode never needs any of this."""
    try:
        frozen_retrievers = build_frozen_retrievers(EVAL_SET)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

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
            # ADR-0059: only THIS repo's own artifacts, so a cassette entry's provenance
            # names exactly what fed its prompt (not the other repo's classifier too).
            repo_artifact_hashes = artifact_fingerprint.compute_artifact_hashes(ROOT, repo=repo)
            assistant = TriageAssistant(
                repo=repo,
                classifier=classifier,
                detector=frozen_retrievers[repo],  # frozen, not live FAISS
                predictor=predictor,
                train_df=train_df,
                groq_api_key=groq_key,
                cache=cassette,
                artifact_hashes=repo_artifact_hashes,
            )
            models[repo] = {"classifier": classifier, "predictor": predictor, "train_df": train_df, "assistant": assistant}
            logger.info("Models loaded for %s", repo)
        except Exception as exc:
            logger.error("Failed to load models for %s: %s", repo, exc)
            sys.exit(1)
    return models


def _build_judge(cassette: CassettePlayer):
    """Lazy import + construct + warm up TriageJudge (Ollama). Only ever called from
    judge/full mode -- --mode synthesis must never reach this function (ADR-0060)."""
    from triage_iq.evaluation.triage_eval import TriageJudge

    judge = TriageJudge(
        model=JUDGE_MODEL, provider=JUDGE_PROVIDER, temperature=0.0, ollama_seed=42, cache=cassette,
    )
    # Throwaway warm-up call (ADR-0019): the first inference after a fresh Ollama model
    # load produces different output than subsequent calls on the same loaded instance —
    # each mode is independently reproducible, but they differ from each other. Absorbing
    # that one-time cold-start here means every REAL judge call below is uniformly
    # warm-mode, so a from-scratch re-run reproduces the identical cassette.
    logger.info("Ollama judge warm-up call (absorbing cold-start variance)...")
    judge._ollama_completion([{"role": "user", "content": "Reply with just: OK"}])
    logger.info("Warm-up done — judge is now in steady warm-mode for the rest of this run.")
    return judge


def _unload_judge_model() -> None:
    """Phase 3c: explicitly unload JUDGE_MODEL from Ollama (keep_alive=0) the moment a
    judge pass ends, instead of restarting Ollama's server with OLLAMA_KEEP_ALIVE=0/
    OLLAMA_MAX_LOADED_MODELS=1 globally -- this machine's Ollama instance may be serving
    other concurrent sessions/tools, and a global server restart would disrupt those for a
    setting this script only needs scoped to its own model. `_ollama_completion`'s own
    per-call keep_alive=-1 (deliberately: never idle-unload mid-pass, avoiding a cold-start
    reload before every one of the ~64 judge calls in a batch) is left unchanged; this is
    the explicit unload for AFTER the batch, achieving the same net VRAM-freed outcome
    without touching that per-call setting or the shared server's global config."""
    try:
        import ollama as _ollama

        host = __import__("os").environ.get("OLLAMA_HOST", "http://localhost:11434")
        _ollama.Client(host=host).generate(model=JUDGE_MODEL, prompt="", keep_alive=0)
        logger.info("Ollama model %s unloaded (keep_alive=0) after the judge pass.", JUDGE_MODEL)
    except Exception as exc:
        logger.warning("Could not explicitly unload Ollama model %s: %s", JUDGE_MODEL, exc)


def _synthesize_one(
    assistant, issue: dict, issue_id: str, i: int, total: int, current_done: dict, checkpoint: dict,
    current_model: str, current_prompt_hash: str, current_artifact_hash: str,
    cassette: CassettePlayer, n_synthesis_recorded: int,
) -> tuple[object | None, str | None, int, bool]:
    """Run synthesis for one issue. Returns (plan_or_None, triage_error_or_None,
    updated_n_synthesis_recorded, already_finalized). already_finalized=True means this
    function already wrote and saved a checkpoint entry for issue_id (the
    degraded_schema_invalid residual case) -- the caller must NOT also record one, just
    move on to the next issue. Every hard-stop condition calls sys.exit(1) directly, exactly
    as the original combined loop did. Shared by --mode full and --mode synthesis so the two
    can never diverge (this project's "one function, everyone uses it" precedent)."""
    row = pd.Series({
        "title": issue["title"],
        "body_clean": issue["body"],
        "number": issue["number"],
        "created_at": pd.Timestamp(issue["created_at"]) if issue.get("created_at") else pd.Timestamp("now", tz="UTC"),
    })

    plan = None
    triage_error = None
    try:
        plan, meta = assistant.triage_with_metadata(row)
        # 2026-08-30: TruncatedCompletionError is caught INSIDE _call_llm_verbose (Part
        # B3's degrade path, PR #113) and converted to a clean fallback plan before it ever
        # reaches this caller -- confirmed by a zero-quota dry run
        # (scripts/record_cassettes_dry_run_check.py) that this except-TruncatedCompletion
        # -Error block below never fires anymore. Without this check, a truncated (or
        # otherwise degraded) completion would silently proceed and get checkpointed as
        # "done" with no synthesis entry ever reaching the cassette (_call_llm_verbose
        # returns before cache.set() on that path) -- a permanent, silent mismatch between
        # the checkpoint (says done) and the cassette (has nothing for that issue),
        # invisible until some later replay hits a CassetteMissError with no link back to
        # this run. Same failure shape as this engagement's other instrumentation-found-
        # after-the-spend incidents; catch it here, at the one place that knows llm_status.
        llm_status = meta.get("llm_status")
        if llm_status == "degraded_schema_invalid":
            # 2026-09-03 (ADR-0055 Part P1a/2c): a syntactically-complete completion Groq's
            # post-hoc schema validator rejected (missing field or malformed key -- ADR-0055's
            # finding). Empirically non-reproducible on retry (Part A: failed once in 2 live
            # attempts on the same issue) -- treat a FIRST occurrence on a given issue as
            # expected residual, not a systemic problem: log it, checkpoint it excluded from
            # done_ids (retried on the next resume, same mechanism as tpd_hit), and continue
            # to the NEXT issue in THIS run rather than stopping. A SECOND failure of this
            # same kind on the SAME issue (i.e. this checkpoint entry already has
            # schema_invalid_retry=True from a prior run) means it reproduced -- that's no
            # longer "rare residual", stop loudly, matching every other hard-stop's severity.
            prior = current_done.get(issue_id)
            if prior is not None and prior.get("schema_invalid_retry"):
                logger.error(
                    "STOP: schema-validation failure REPRODUCED on #%s (2nd attempt) "
                    "after %d synthesis calls. Cassette has %d entries. This is no "
                    "longer treated as rare residual -- investigate before resuming.",
                    issue_id, n_synthesis_recorded, cassette.stats()["entries"],
                )
                save_checkpoint({"done": checkpoint.get("done", {})})
                print("\n=== SCHEMA VALIDATION FAILURE REPRODUCED (2nd attempt on same issue) ===")
                print(f"Issue: {issue_id}")
                print(f"Synthesis recorded before stop: {n_synthesis_recorded}")
                print(f"Cassette entries: {cassette.stats()['entries']}")
                sys.exit(1)
            logger.warning(
                "Schema-validation failure on #%s (1st occurrence, expected residual "
                "per ADR-0055 Part A) -- logging and continuing to the next issue.",
                issue_id,
            )
            rec = {
                "error": meta.get("groq_error_code", "json_validate_failed"),
                "plan": None, "judge_score": None, "schema_invalid_retry": True,
            }
            _record_done(checkpoint, issue_id, rec, current_model, current_prompt_hash, current_artifact_hash)
            save_checkpoint(checkpoint)
            return None, None, n_synthesis_recorded, True
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
        # 2026-08-28: a truncated completion means max_tokens is too small for this model --
        # every subsequent issue is likely to hit the same wall, so this stops the run
        # loudly rather than silently skipping entries (and burning TPD budget) under a
        # config already known to be broken. Raise max_tokens and restart -- the resume
        # mechanism picks up from recording_checkpoint.json as usual.
        logger.error(
            "STOP: completion truncated after %d synthesis calls "
            "(completion_tokens=%d, max_tokens=%d). Cassette has %d entries. "
            "Raise max_tokens and re-run -- resume will continue from checkpoint.",
            n_synthesis_recorded, exc.completion_tokens, exc.max_tokens, cassette.stats()["entries"],
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
        # already retries a RateLimitError internally (6 attempts, exponential backoff)
        # before this exception ever reaches here, so by the time we see one, it has
        # already survived real backoff and is not a transient blip. But its message does
        # not reliably contain "daily"/"tpd" (Groq's rate-limit body uses 'code':
        # 'rate_limit_exceeded' whether the underlying cause is a per-minute or per-day
        # ceiling -- confirmed directly from this session's captured raw Groq error
        # bodies). Under the old check, a sustained-but-not-explicitly-"daily" rate limit
        # fell through to the generic except branch below and got recorded as a permanent
        # per-issue failure (checkpointed done, never retried) -- the same silent-mismatch
        # shape the llm_status check above this block exists to prevent, just for a
        # different trigger. _is_rate_limit_error covers both.
        if _is_rate_limit_error(exc):
            logger.error(
                "STOP: Groq rate limit (TPD or sustained TPM) hit after %d synthesis "
                "calls. Cassette has %d entries. Groq error: %s",
                n_synthesis_recorded, cassette.stats()["entries"], exc,
            )
            save_checkpoint({"done": checkpoint.get("done", {})})
            print(f"\n=== TPD HIT ===")
            print(f"Synthesis recorded: {n_synthesis_recorded}")
            print(f"Cassette entries: {cassette.stats()['entries']}")
            # exit(1): this recording is INCOMPLETE. exit(0) here previously let a partial
            # cassette look like a clean run to the shell -- CI's "Update eval baseline"
            # step would then hit CassetteMissError on the first un-recorded issue with no
            # indication the recording itself was short. Local resumable use is unaffected:
            # re-running still resumes from recording_checkpoint.json regardless of exit code.
            sys.exit(1)
        if _is_connection_error(exc):
            logger.error(
                "STOP: connection lost after %d synthesis calls. Cassette has %d entries. Error: %s",
                n_synthesis_recorded, cassette.stats()["entries"], exc,
            )
            save_checkpoint({"done": checkpoint.get("done", {})})
            print("\n=== CONNECTION LOST ===")
            print(f"Synthesis recorded: {n_synthesis_recorded}")
            print(f"Cassette entries: {cassette.stats()['entries']}")
            sys.exit(1)  # incomplete recording -- see TPD-exit comment above
        logger.warning("  synthesis FAILED: %s", exc)
        triage_error = str(exc)

    return plan, triage_error, n_synthesis_recorded, False


def _judge_one(judge, issue: dict, plan_dict: dict, n_judge_recorded: int) -> tuple[dict | None, int, bool]:
    """Score one already-synthesized plan (plan_dict already excludes
    _JUDGE_EXCLUDED_PLAN_FIELDS). Returns (judge_score_dict_or_None, updated
    n_judge_recorded, hit_fatal_tpd) -- the caller decides the exit message/checkpoint
    write for hit_fatal_tpd, since that differs between --mode full and --mode judge."""
    time.sleep(JUDGE_DELAY)
    plan_json = json.dumps(plan_dict, ensure_ascii=False)
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
                issue_title=issue["title"], issue_body=issue["body"][:600],
                triage_plan_json=plan_json, gold=gold,
            )
            judge_score = score.model_dump()
            n_judge_recorded += 1
            logger.info("  judge → %d/%d", score.total(), sum(DIMENSION_MAX.values()))
            _judge_exc = None
            break
        except Exception as exc:
            if _is_tpd_error(exc):
                _judge_exc = exc
                break  # genuine daily limit — stop outer loop below (local judge: effectively unreachable, kept for defense in depth)
            if _is_connection_error(exc):
                logger.error("STOP: connection lost during judging: %s", exc)
                print("\n=== CONNECTION LOST (during judge) ===")
                print(f"Judge recorded: {n_judge_recorded}")
                sys.exit(1)
            if _is_rate_limit_error(exc):
                _wait = 20 * (2 ** _attempt)
                logger.warning("  judge TPM rate limit (per-minute) — waiting %ds (attempt %d/6): %s",
                               _wait, _attempt + 1, str(exc)[:200])
                time.sleep(_wait)
                continue
            logger.warning("  judge FAILED: %s", exc)
            break

    if _judge_exc is not None:
        return None, n_judge_recorded, True
    return judge_score, n_judge_recorded, False


def run_full(groq_key, issues, cassette, current_model, current_prompt_hash, current_artifact_hash, checkpoint, current_done) -> None:
    """Original combined behavior: synthesis + judge per issue, one pass. Default mode --
    kept for any caller that doesn't care about the RAM/VRAM split (ADR-0060)."""
    models = _load_models(groq_key, cassette)
    judge = _build_judge(cassette)

    done_ids = {k for k, v in current_done.items() if not v.get("tpd_hit") and not v.get("schema_invalid_retry")}
    n_tpd_retry = len(current_done) - len(done_ids)
    logger.info("Checkpoint: %d issues already processed under current config (%d tpd_hit will retry)",
                len(done_ids), n_tpd_retry)

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
        assistant = models[repo]["assistant"]
        plan, triage_error, n_synthesis_recorded, already_finalized = _synthesize_one(
            assistant, issue, issue_id, i, len(issues), current_done, checkpoint,
            current_model, current_prompt_hash, current_artifact_hash, cassette, n_synthesis_recorded,
        )
        if already_finalized:
            continue
        if plan is None:
            rec = {"error": triage_error, "plan": None, "judge_score": None}
            results[issue_id] = rec
            _record_done(checkpoint, issue_id, rec, current_model, current_prompt_hash, current_artifact_hash)
            save_checkpoint(checkpoint)
            continue

        plan_dict = plan.model_dump(exclude=_JUDGE_EXCLUDED_PLAN_FIELDS)
        judge_score, n_judge_recorded, hit_fatal_tpd = _judge_one(judge, issue, plan_dict, n_judge_recorded)
        if hit_fatal_tpd:
            logger.error("STOP: Groq TPD (daily quota) hit during judging after %d judge calls.", n_judge_recorded)
            rec = {"plan": plan.model_dump(), "judge_score": None, "tpd_hit": True}
            results[issue_id] = rec
            _record_done(checkpoint, issue_id, rec, current_model, current_prompt_hash, current_artifact_hash)
            save_checkpoint(checkpoint)
            print(f"\n=== TPD HIT (during judge) ===")
            print(f"Synthesis recorded: {n_synthesis_recorded}")
            print(f"Judge recorded: {n_judge_recorded}")
            sys.exit(1)

        rec = {"plan": plan.model_dump(), "judge_score": judge_score, "error": triage_error}
        results[issue_id] = rec
        _record_done(checkpoint, issue_id, rec, current_model, current_prompt_hash, current_artifact_hash)
        save_checkpoint(checkpoint)

    _print_summary_full(issues, results, current_model, current_prompt_hash, n_synthesis_recorded, n_skipped, cassette)


def run_synthesis(groq_key, issues, cassette, current_model, current_prompt_hash, current_artifact_hash, checkpoint, current_done) -> None:
    """--mode synthesis: Groq calls only. Never imports TriageJudge, never touches Ollama --
    RAM/VRAM stay free for whatever else needs them (ADR-0060). Checkpoints each success as
    judge_pending (plan present, judge_score None) instead of calling the judge."""
    models = _load_models(groq_key, cassette)

    # "Already done" for synthesis purposes = has a plan already (judged or judge-pending),
    # OR a permanent non-retryable synthesis failure. Identical formula to run_full's --
    # a synthesis-only pass and a full pass must treat "already synthesized" identically.
    done_ids = {k for k, v in current_done.items() if not v.get("tpd_hit") and not v.get("schema_invalid_retry")}
    n_tpd_retry = len(current_done) - len(done_ids)
    logger.info("Checkpoint: %d issues already synthesized (%d retryable will retry)", len(done_ids), n_tpd_retry)

    n_synthesis_recorded = 0
    n_skipped = 0
    n_pending_judge = 0

    for i, issue in enumerate(issues):
        issue_id = issue["id"]
        repo = issue["repo"]

        if issue_id in done_ids:
            logger.info("[%d/%d] %s — skipped (checkpoint)", i + 1, len(issues), issue_id)
            n_skipped += 1
            continue

        if i > 0 and issue_id not in done_ids:
            time.sleep(SYNTHESIS_DELAY)

        logger.info("[%d/%d] %s — synthesizing …", i + 1, len(issues), issue_id)
        assistant = models[repo]["assistant"]
        plan, triage_error, n_synthesis_recorded, already_finalized = _synthesize_one(
            assistant, issue, issue_id, i, len(issues), current_done, checkpoint,
            current_model, current_prompt_hash, current_artifact_hash, cassette, n_synthesis_recorded,
        )
        if already_finalized:
            continue

        rec = {
            "plan": plan.model_dump() if plan is not None else None,
            "judge_score": None,
            "judge_pending": plan is not None,
            "error": triage_error,
        }
        _record_done(checkpoint, issue_id, rec, current_model, current_prompt_hash, current_artifact_hash)
        save_checkpoint(checkpoint)
        if plan is not None:
            n_pending_judge += 1

    total_synthesized = n_skipped + n_synthesis_recorded
    print(f"\n=== SYNTHESIS {'COMPLETE' if total_synthesized >= len(issues) else 'PARTIAL'} ===")
    print(f"Model:               {current_model}")
    print(f"Prompt hash:         {current_prompt_hash}")
    print(f"Artifact hash:       {current_artifact_hash}")
    print(f"Issues synthesized:  {total_synthesized}/{len(issues)} (skipped {n_skipped}, new {n_synthesis_recorded})")
    print(f"Awaiting judge pass: {n_pending_judge} new this run (run --mode judge next)")
    if total_synthesized < len(issues):
        sys.exit(1)


def _pending_judge_entries(current_done: dict[str, dict]) -> dict[str, dict]:
    """Entries that are synthesis-done but NOT fully-done: plan present, judge_score still
    None. Deliberately NOT the same set as run_synthesis's done_ids (which includes
    fully-judged entries too) -- this is the one function that must distinguish
    "synthesis-done" from "fully-done" (Phase 2c), so both run_judge and its test import
    this same definition rather than each re-deriving it."""
    return {
        issue_id: v for issue_id, v in current_done.items()
        if v.get("plan") is not None and v.get("judge_score") is None
    }


def _fully_judged_count(current_done: dict[str, dict]) -> int:
    """Entries with a real judge_score -- the ONLY thing a completion summary may count
    toward "N/64 done". Never count len(current_done) or a synthesis-only count here; that
    is exactly the bug this function exists to make structurally impossible (Phase 2c)."""
    return sum(1 for v in current_done.values() if v.get("judge_score") is not None)


def run_judge(issues, cassette, current_model, current_prompt_hash, current_artifact_hash, checkpoint, current_done) -> None:
    """--mode judge: batch-score every synthesis-done, not-yet-judged entry with the local
    Ollama judge. Loads no classifier/predictor/retriever, makes zero Groq calls -- Ollama
    is the only heavy thing this mode touches, and only for the duration of this pass."""
    by_id = {issue["id"]: issue for issue in issues}

    pending = _pending_judge_entries(current_done)
    already_judged = _fully_judged_count(current_done)
    not_yet_synthesized = len(issues) - len(current_done)

    logger.info(
        "Judge pass: %d already judged, %d pending judge, %d not yet synthesized (need "
        "--mode synthesis for those first).",
        already_judged, len(pending), not_yet_synthesized,
    )
    if not pending:
        print("\n=== NOTHING TO JUDGE ===")
        print(f"Already judged: {already_judged}/{len(issues)}")
        print(f"Not yet synthesized: {not_yet_synthesized}/{len(issues)} (run --mode synthesis first)")
        if not_yet_synthesized:
            sys.exit(1)
        return

    judge = _build_judge(cassette)
    n_judge_recorded = 0

    try:
        for i, (issue_id, rec) in enumerate(pending.items()):
            issue = by_id[issue_id]
            logger.info("[%d/%d] %s — judging …", i + 1, len(pending), issue_id)
            plan_dict = {k: v for k, v in rec["plan"].items() if k not in _JUDGE_EXCLUDED_PLAN_FIELDS}
            judge_score, n_judge_recorded, hit_fatal_tpd = _judge_one(judge, issue, plan_dict, n_judge_recorded)
            if hit_fatal_tpd:
                logger.error("STOP: judge hit a fatal rate-limit signal after %d judge calls "
                             "(unexpected for a local Ollama judge -- investigate).", n_judge_recorded)
                print(f"\n=== JUDGE PASS STOPPED (unexpected rate-limit signal) ===")
                print(f"Judge recorded this run: {n_judge_recorded}")
                sys.exit(1)

            updated = {**rec, "judge_score": judge_score, "judge_pending": False}
            _record_done(checkpoint, issue_id, updated, current_model, current_prompt_hash, current_artifact_hash)
            save_checkpoint(checkpoint)
    finally:
        # Phase 3c: unload the model the moment this pass is done (success, partial, or a
        # sys.exit above) so VRAM is freed immediately rather than waiting for Ollama's own
        # idle-unload timer -- runs even on the early-exit paths, not just the happy path.
        _unload_judge_model()

    # Re-derive from checkpoint["done"] (not the `pending`/`already_judged` snapshot taken
    # before this loop ran) so the final count reflects what THIS run actually wrote.
    current_config_done = {
        v["issue_id"]: v for v in checkpoint["done"].values()
        if v.get("model") == current_model and v.get("prompt_hash") == current_prompt_hash
        and v.get("artifact_hash") == current_artifact_hash
    }
    final_judged = _fully_judged_count(current_config_done)
    print(f"\n=== JUDGE {'COMPLETE' if final_judged >= len(issues) else 'PARTIAL'} ===")
    print(f"Model:            {current_model}")
    print(f"Prompt hash:      {current_prompt_hash}")
    print(f"Artifact hash:    {current_artifact_hash}")
    print(f"Judge recorded:   {n_judge_recorded} this run")
    print(f"Fully judged:     {final_judged}/{len(issues)}")
    if final_judged < len(issues):
        sys.exit(1)


def _print_summary_full(issues, results, current_model, current_prompt_hash, n_synthesis_recorded, n_skipped, cassette) -> None:
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
        print(f"Cassette entries:   {cassette.stats()['entries']}")
        print(f"Judge score (mean): {np.mean(total_scores):.2f}/15 ({np.mean(total_scores)/15*100:.0f}%)")

        by_repo: dict[str, list] = {}
        for issue in issues:
            r = results.get(issue["id"], {})
            if r.get("judge_score"):
                by_repo.setdefault(issue["repo"], []).append(JudgeScore.model_validate(r["judge_score"]).total())
        for repo, repo_scores in by_repo.items():
            print(f"  {repo}: n={len(repo_scores)}, mean={np.mean(repo_scores):.2f}/15")
    elif completed and n_synthesis_recorded == 0:
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
    # The loop above can silently leave an issue with judge_score=None without ever hitting
    # one of the sys.exit(1) paths above. Those gaps are invisible here in this script's own
    # output -- they only surface later as a CassetteMissError in run_eval.py's replay, in a
    # different job step, with no link back to this run. Assert it here instead: every issue
    # must have a non-None judge_score, or this is not a usable recording.
    missing = [iid for iid in (i["id"] for i in issues) if results.get(iid, {}).get("judge_score") is None]
    if missing:
        print(f"\n=== INCOMPLETE: {len(missing)}/{len(issues)} issues missing a judge score ===")
        print(f"Missing: {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        sys.exit(1)


def main() -> None:
    args = parse_args()
    (
        groq_key, issues, cassette, current_model, current_prompt_hash,
        current_artifact_hash, checkpoint, current_done,
    ) = _load_common(args.mode)

    if args.mode == "synthesis":
        run_synthesis(groq_key, issues, cassette, current_model, current_prompt_hash, current_artifact_hash, checkpoint, current_done)
    elif args.mode == "judge":
        run_judge(issues, cassette, current_model, current_prompt_hash, current_artifact_hash, checkpoint, current_done)
    else:
        run_full(groq_key, issues, cassette, current_model, current_prompt_hash, current_artifact_hash, checkpoint, current_done)


if __name__ == "__main__":
    main()
