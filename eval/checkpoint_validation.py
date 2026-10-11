"""Checkpoint validation for eval/record_cassettes.py (ADR-0062 addendum).

Problem: recording_checkpoint.json keys an entry by (issue, model, prompt_hash, artifact_hash).
None of those four change when a CODE change alters the request a given issue produces -- e.g.
#150 made k8s resolution features depend on the retrieval query embedding, which changed the
resolution numbers embedded in the synthesis prompt, hence the cassette key, while the
checkpoint still said "done". A plain resume would have re-recorded nothing and replay would
then have raised CassetteMissError on 53 issues. The coordinator deleted 53 keys by hand.

Fix: before trusting any "done" entry, recompute its request through the CURRENT code -- the
exact replay machinery eval/run_eval.py uses (frozen retriever + classifier + predictor +
prompt builder + a strict, read-only CassettePlayer) -- and require a cassette HIT:

    valid            synthesis request hits the cassette (and, if a judge score is recorded,
                     the judge request built from the replayed plan hits too)
    stale_synthesis  the synthesis request misses (or its key differs from the checkpoint's
                     recorded synthesis_cache_key): the entry must be re-synthesized
    stale_judge      synthesis still hits but the judge request -- which depends on the plan --
                     misses: only the judge must be redone (the plan is refreshed from replay)

Entries without a plan (permanent synthesis failures, tpd_hit, schema_invalid_retry) record no
cassette-backed request, so there is nothing to validate; the recorder's existing dead/retry
logic governs them and they are omitted from the result.

Fail closed (rule 98a): if validation cannot run (missing artifacts, missing cassette, replay
raising anything other than a cassette miss) it raises CheckpointValidationUnavailable. Callers
must stop, never fall back to trusting the checkpoint.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from cassette import CassetteMissError, CassettePlayer  # noqa: E402

logger = logging.getLogger(__name__)

VALID = "valid"
STALE_SYNTHESIS = "stale_synthesis"
STALE_JUDGE = "stale_judge"

# Same replay-time constants as eval/run_eval.py (the replay tests); validation must use the
# identical request-construction parameters or a "valid" verdict would not predict a replay hit.
_JUDGE_EXCLUDED_PLAN_FIELDS = {"declared_attribution", "abstention_status"}
_OK_LLM_STATUSES = ("ok", "parse_retry_succeeded")


class CheckpointValidationUnavailable(RuntimeError):
    """Validation could not run. Callers must stop rather than trust the checkpoint."""


@dataclass(frozen=True)
class ValidationResult:
    status: str
    detail: str = ""
    # Plan recomputed by the current code (model_dump), only for entries whose synthesis hit.
    replayed_plan: dict[str, Any] | None = None


def issue_to_row(issue: dict) -> pd.Series:
    """The row the pipeline is called with -- identical to run_eval.compute_scores."""
    return pd.Series(
        {
            "title": issue["title"],
            "body_clean": issue["body"],
            "number": issue["number"],
            "created_at": (
                pd.Timestamp(issue["created_at"])
                if issue.get("created_at")
                else pd.Timestamp("now", tz="UTC")
            ),
        }
    )


def build_replay_context(
    cassette_path: Path | None = None,
    eval_set_path: Path | None = None,
    *,
    need_judge: bool = True,
) -> tuple[list[dict], CassettePlayer, dict[str, Any], Any]:
    """Load issues, a STRICT read-only cassette, per-repo TriageAssistants and (optionally) a
    cache-only TriageJudge, exactly as run_eval.compute_scores does.

    need_judge=False never imports TriageJudge (--mode synthesis must not touch it, ADR-0060).
    Raises CheckpointValidationUnavailable on any missing/unloadable input.
    """
    import run_eval  # light import; shares MODELS_DIR / REPO_MAP / constants with replay tests

    cassette_path = cassette_path or run_eval.CASSETTE_PATH
    eval_set_path = eval_set_path or run_eval.EVAL_SET_PATH
    for p in (cassette_path, eval_set_path):
        if not Path(p).exists():
            raise CheckpointValidationUnavailable(f"required input missing: {p}")
    try:
        # strict=True, allow_record=False: a miss raises, nothing can ever be written.
        cassette = CassettePlayer(cassette_path, strict=True, allow_record=False)
        issues = run_eval._load_eval_set(eval_set_path)
        from frozen_retriever import build_frozen_retrievers

        frozen = build_frozen_retrievers(eval_set_path)
        assistants = {
            repo: run_eval._load_models(repo, slug, cassette, frozen)["assistant"]
            for repo, slug in run_eval.REPO_MAP.items()
        }
        judge = None
        if need_judge:
            from triage_iq.evaluation.triage_eval import TriageJudge

            judge = TriageJudge(
                groq_api_key=run_eval.CI_API_KEY,
                model=run_eval.JUDGE_MODEL,
                provider=run_eval.JUDGE_PROVIDER,
                temperature=0.0,
                ollama_seed=42,
                cache=cassette,
            )
    except CheckpointValidationUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - any load failure means "cannot validate"
        raise CheckpointValidationUnavailable(
            f"could not build the replay context ({type(exc).__name__}: {exc}). Required: "
            "data/models + data/processed artifacts, eval/eval_set.jsonl with frozen "
            "similar_issues, the cassette."
        ) from exc
    return issues, cassette, assistants, judge


def _entries(checkpoint: Mapping[str, Any]) -> dict[str, dict]:
    """Accept either {"done": {key: rec}} (raw file) or {issue_id: rec} (config-filtered)."""
    done = checkpoint.get("done") if "done" in checkpoint else None
    if isinstance(done, dict):
        return {rec["issue_id"]: rec for rec in done.values()}
    return dict(checkpoint)  # type: ignore[arg-type]


def validate_checkpoint_detailed(
    issues: list[dict],
    cassette: CassettePlayer,
    checkpoint: Mapping[str, Any],
    *,
    assistants: Mapping[str, Any],
    judge: Any | None = None,
) -> dict[str, ValidationResult]:
    """Core validator. `assistants` maps repo -> TriageAssistant wired to `cassette` (strict);
    `judge` is a cache-only TriageJudge or None (None skips the judge check entirely).
    """
    by_id = {i["id"]: i for i in issues}
    results: dict[str, ValidationResult] = {}
    for issue_id, rec in _entries(checkpoint).items():
        if rec.get("plan") is None:
            continue  # no cassette-backed request to validate (see module docstring)
        issue = by_id.get(issue_id)
        if issue is None or issue["repo"] not in assistants:
            results[issue_id] = ValidationResult(
                STALE_SYNTHESIS, "issue not in the eval set / no assistant for its repo"
            )
            continue
        try:
            plan, meta = assistants[issue["repo"]].triage_with_metadata(issue_to_row(issue))
        except CassetteMissError as exc:
            results[issue_id] = ValidationResult(STALE_SYNTHESIS, f"synthesis miss: {str(exc)[:80]}")
            continue
        except Exception as exc:  # noqa: BLE001
            raise CheckpointValidationUnavailable(
                f"replay of {issue_id} raised {type(exc).__name__}: {exc}"
            ) from exc

        llm_status = meta.get("llm_status")
        key = meta.get("synthesis_cache_key")
        if llm_status not in _OK_LLM_STATUSES or key is None:
            results[issue_id] = ValidationResult(
                STALE_SYNTHESIS, f"replay degraded (llm_status={llm_status}, key={key})"
            )
            continue
        recorded_key = rec.get("synthesis_cache_key")
        if recorded_key is not None and recorded_key != key:
            results[issue_id] = ValidationResult(
                STALE_SYNTHESIS,
                f"synthesis key changed: checkpoint {recorded_key[:12]} != current {key[:12]}",
            )
            continue

        plan_dump = plan.model_dump()
        if judge is None or rec.get("judge_score") is None:
            results[issue_id] = ValidationResult(VALID, replayed_plan=plan_dump)
            continue

        plan_json = json.dumps(
            plan.model_dump(exclude=_JUDGE_EXCLUDED_PLAN_FIELDS), ensure_ascii=False
        )
        gold = {
            "component": issue["gold_component"],
            "priority": issue["gold_priority"],
            "actual_resolution_days": issue["actual_resolution_days"],
        }
        try:
            judge.score(
                issue_title=issue["title"],
                issue_body=issue["body"][:600],
                triage_plan_json=plan_json,
                gold=gold,
            )
        except CassetteMissError as exc:
            results[issue_id] = ValidationResult(
                STALE_JUDGE, f"judge miss: {str(exc)[:80]}", replayed_plan=plan_dump
            )
            continue
        except Exception as exc:  # noqa: BLE001
            raise CheckpointValidationUnavailable(
                f"judge replay of {issue_id} raised {type(exc).__name__}: {exc}"
            ) from exc
        results[issue_id] = ValidationResult(VALID, replayed_plan=plan_dump)
    return results


def validate_checkpoint(
    issues: list[dict],
    cassette: CassettePlayer,
    checkpoint: Mapping[str, Any],
    *,
    assistants: Mapping[str, Any],
    judge: Any | None = None,
) -> dict[str, str]:
    """{issue_id: 'valid' | 'stale_synthesis' | 'stale_judge'} -- see module docstring."""
    detailed = validate_checkpoint_detailed(
        issues, cassette, checkpoint, assistants=assistants, judge=judge
    )
    return {issue_id: r.status for issue_id, r in detailed.items()}


def validate_from_disk(
    checkpoint: Mapping[str, Any],
    *,
    check_judge: bool,
    cassette_path: Path | None = None,
    eval_set_path: Path | None = None,
) -> dict[str, ValidationResult]:
    """Build the replay context from the repo's real artifacts and validate `checkpoint`."""
    issues, cassette, assistants, judge = build_replay_context(
        cassette_path, eval_set_path, need_judge=check_judge
    )
    return validate_checkpoint_detailed(
        issues, cassette, checkpoint, assistants=assistants, judge=judge
    )


def reconcile_current_done(
    current_done: dict[str, dict], results: Mapping[str, ValidationResult], mode: str
) -> dict[str, dict]:
    """Return a copy of `current_done` with stale entries treated as pending.

    * stale_synthesis -> dropped (re-synthesized live by synthesis/full).
    * stale_judge     -> full: dropped (re-synthesis is a cassette hit, then a fresh judge);
                         judge: kept but judge_score cleared, judge_pending set and the plan
                         refreshed from replay, so --mode judge scores the CURRENT plan;
                         synthesis: unaffected (synthesis-done does not depend on the judge).
    """
    out = dict(current_done)
    for issue_id, res in results.items():
        if issue_id not in out or res.status == VALID:
            continue
        if res.status == STALE_SYNTHESIS:
            del out[issue_id]
        elif res.status == STALE_JUDGE:
            if mode == "full":
                del out[issue_id]
            elif mode == "judge":
                out[issue_id] = {
                    **out[issue_id],
                    "plan": res.replayed_plan,
                    "judge_score": None,
                    "judge_pending": True,
                }
    return out


def summarize(results: Mapping[str, ValidationResult]) -> str:
    counts: dict[str, int] = {}
    for r in results.values():
        counts[r.status] = counts.get(r.status, 0) + 1
    head = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no plan-bearing entries"
    lines = [f"Checkpoint validation: {len(results)} entries: {head}"]
    for issue_id, r in sorted(results.items()):
        if r.status != VALID:
            lines.append(f"  {r.status:<16} {issue_id}  ({r.detail})")
    return "\n".join(lines)
