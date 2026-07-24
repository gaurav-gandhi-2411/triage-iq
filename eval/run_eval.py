from __future__ import annotations

"""Eval runner: replays the full TriageIQ pipeline over eval_set.jsonl using the frozen cassette.

Importable as a module (compute_scores) or runnable as a script.

Usage:
    python eval/run_eval.py                  # print scores
    python eval/run_eval.py --update-baseline  # print scores + write reports/eval_baseline.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from cassette import CassettePlayer
from frozen_retriever import build_frozen_retrievers
from triage_iq.models.component_classifier import load_classifier
from triage_iq.evaluation.triage_eval import DIMENSION_MAX, JudgeScore, TriageJudge
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant

MODELS_DIR = ROOT / "data" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"
CASSETTE_PATH = ROOT / "eval" / "cassettes" / "eval_cassette.json"
BASELINE_PATH = ROOT / "reports" / "eval_baseline.json"

REPO_MAP: dict[str, str] = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}

# Local judge (ADR-0019): zero-cost, reproducible without a live key. Cassette
# replay never calls Ollama live -- CassettePlayer(strict=True) serves every
# call from the committed cassette -- but the model/provider must match what
# was recorded, since the cache key includes both.
JUDGE_MODEL = "qwen3:8b"
JUDGE_PROVIDER = "ollama"
CI_API_KEY = "ci-replay-only"


def _file_sha256(path: Path) -> str:
    """Return hex SHA-256 digest of a file's bytes."""
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def _load_eval_set(path: Path) -> list[dict]:
    """Read JSONL eval set, returning one dict per non-empty line."""
    issues: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                issues.append(json.loads(line))
    return issues


def _load_models(
    repo: str,
    slug: str,
    cassette: CassettePlayer,
    frozen_retrievers: dict,
) -> dict[str, Any]:
    """Load all per-repo models and return a dict containing the TriageAssistant.

    Uses FrozenRetriever instead of live FAISS so synthesis prompts are
    deterministic on any hardware. Production /triage is unchanged.
    """
    # load_classifier() dispatches on the pkl's model_kind marker (ADR-0036: multi-label
    # OvR vs legacy single-label) -- same loader the live API uses, not hardcoded to one class.
    classifier = load_classifier(MODELS_DIR, slug)
    predictor = ResolutionTimePredictor.load(
        str(MODELS_DIR / f"resolution_predictor_{slug}.pkl")
    )
    train_df = pd.read_parquet(PROCESSED_DIR / f"{slug}_temporal_train.parquet")
    assistant = TriageAssistant(
        repo=repo,
        classifier=classifier,
        detector=frozen_retrievers[repo],  # frozen, not live FAISS
        predictor=predictor,
        train_df=train_df,
        groq_api_key=CI_API_KEY,
        cache=cassette,
    )
    return {
        "classifier": classifier,
        "predictor": predictor,
        "train_df": train_df,
        "assistant": assistant,
    }


def compute_scores(
    cassette_path: Path = CASSETTE_PATH,
    eval_set_path: Path = EVAL_SET_PATH,
) -> dict:
    """Run the full pipeline in replay-only mode and return aggregate scores.

    Returns a dict with keys:
        eval_set_hash  — SHA-256 of eval_set.jsonl
        cassette_hash  — SHA-256 of eval_cassette.json
        per_repo       — {repo: {n, mean, dimensions: {dim: float}}}
        overall        — {n, mean}

    All LLM calls are served from the cassette (strict=True). Raises
    CassetteMissError if any call is absent from the cassette.
    """
    cassette = CassettePlayer(cassette_path, strict=True)
    issues = _load_eval_set(eval_set_path)
    frozen_retrievers = build_frozen_retrievers(eval_set_path)

    models: dict[str, dict] = {}
    for repo, slug in REPO_MAP.items():
        models[repo] = _load_models(repo, slug, cassette, frozen_retrievers)

    judge = TriageJudge(
        groq_api_key=CI_API_KEY,
        model=JUDGE_MODEL,
        provider=JUDGE_PROVIDER,
        temperature=0.0,
        ollama_seed=42,
        cache=cassette,
    )

    dim_keys = list(DIMENSION_MAX.keys())
    repo_scores: dict[str, list[dict]] = {repo: [] for repo in REPO_MAP}
    # ADR-0028 Phase B3: the judge never sees classifier_top3 / retrieved_numbers, so it
    # cannot detect fabrication in principle (a known hallucination scored ABOVE its
    # repo's mean). plan.grounding_status is already computed deterministically inside
    # triage_with_metadata -- reading it off here adds zero LLM calls. floor_fail_rate
    # similarly reuses dimension scores the judge already produced.
    repo_grounded: dict[str, list[bool]] = {repo: [] for repo in REPO_MAP}

    for issue in issues:
        repo = issue["repo"]
        assistant = models[repo]["assistant"]

        row = pd.Series({
            "title": issue["title"],
            "body_clean": issue["body"],
            "number": issue["number"],
            "created_at": (
                pd.Timestamp(issue["created_at"])
                if issue.get("created_at")
                else pd.Timestamp("now", tz="UTC")
            ),
        })

        plan, _meta = assistant.triage_with_metadata(row)
        repo_grounded[repo].append(plan.grounding_status.all_grounded)

        # ADR-0019: the cassette was re-recorded from scratch against current TriagePlan
        # (grounding + grounding_status included) — no exclusion needed for those anymore.
        # ADR-0020 reintroduced the same problem for declared_attribution: TriagePlan gained
        # the field unconditionally regardless of the attribution prompt flag, so
        # plan.model_dump() always emits it (as None) even on the legacy path, and the old
        # cassette's judge calls were recorded against plan_json that never had this key.
        # ADR-0021 does the same for abstention_status -- also unconditional on TriagePlan
        # (computed in app.py, not here, but the field still exists and always serializes).
        # Both excluded so the judge cache key keeps matching the clean cassette; this eval
        # harness never populates either field anyway (run_eval.py calls
        # triage_with_metadata() directly, bypassing app.py's post-processing).
        #
        # ADR-0025: resolution_bucket / resolution_confidence_pct are deliberately NOT
        # excluded, unlike the three fields above. Those three never existed in the
        # original recording, so excluding them restores an exact text match. These two
        # DID exist at record time with real values -- excluding them now would change
        # plan_json's text relative to the ORIGINAL recording for EVERY issue (not just
        # microsoft/vscode), breaking kubernetes/kubernetes too even though its value is
        # unchanged by BUCKET_CLASSIFIER_TRUSTED (confirmed directly: excluding these two
        # caused a cache miss on the very first k8s issue). The correct fix is a targeted
        # partial re-record of microsoft/vscode's judge entries only (its plan_json value
        # for these fields genuinely changed), not an exclusion. See ADR-0025.
        plan_json = json.dumps(
            plan.model_dump(exclude={"declared_attribution", "abstention_status"}),
            ensure_ascii=False,
        )
        gold = {
            "component": issue["gold_component"],
            "priority": issue["gold_priority"],
            "actual_resolution_days": issue["actual_resolution_days"],
        }

        score = judge.score(
            issue_title=issue["title"],
            issue_body=issue["body"][:600],
            triage_plan_json=plan_json,
            gold=gold,
        )
        repo_scores[repo].append(score.model_dump())

    per_repo: dict[str, dict] = {}
    all_totals: list[float] = []

    for repo, score_dicts in repo_scores.items():
        judge_scores = [JudgeScore.model_validate(s) for s in score_dicts]
        totals = [float(s.total()) for s in judge_scores]
        all_totals.extend(totals)

        dim_means: dict[str, float] = {}
        for key in dim_keys:
            dim_means[key] = float(np.mean([getattr(s, key) for s in judge_scores]))

        # Floor-fail: judge's own worst-band semantics (component_match==0 or
        # similar_issues_relevance==0) -- a correctness-critical miss that a passable
        # mean can hide (see docs/investigations/gold-set-leakage.md Sec 5, ADR-0028).
        floor_fails = [
            s.component_match == 0 or s.similar_issues_relevance == 0 for s in judge_scores
        ]
        n = len(totals)
        fabrication_rate = float(np.mean([not g for g in repo_grounded[repo]])) if n else 0.0

        per_repo[repo] = {
            "n": n,
            "mean": round(float(np.mean(totals)), 4),
            "dimensions": dim_means,
            "floor_fail_rate": round(float(np.mean(floor_fails)), 4) if n else 0.0,
            "fabrication_rate": round(fabrication_rate, 4),
        }

    overall: dict[str, Any] = {
        "n": len(all_totals),
        "mean": round(float(np.mean(all_totals)), 4),
    }

    return {
        "eval_set_hash": _file_sha256(eval_set_path),
        "cassette_hash": _file_sha256(cassette_path),
        "per_repo": per_repo,
        "overall": overall,
    }


# ADR-0019: byte-identical re-recording is not achievable with this project's inference
# stack (Groq has replica-level nondeterminism even with an explicit seed; Ollama has
# ~9% sequence-state divergence at full-run scale that survives strict decoding). Neither
# is a config bug to fix -- both were tested directly. The cassette REPLAY invariant stays
# exact (same committed bytes -> same score, always -- see test_cassette_hash_matches_baseline).
# What moves to a tolerance band is the RE-RECORD-reproducibility comparison: per-repo MEAN
# vs. baseline, banded by 2x the measured per-issue jitter's standard error.
#
# Measured directly, not guessed: two independent full local-judge (qwen3:8b) recordings
# were compared on their 57 common issues. Per-issue total-score jitter: std=0.748 (on the
# /15 scale), mean diff ~0 (symmetric noise, no systematic bias). SEM = std / sqrt(n),
# using each repo's actual comparison sample size (k8s n=46, vscode n=11 from that
# reproduction check). Band = 2 x SEM.
_MEASURED_JITTER_STD = 0.748
_JITTER_SEM_N = {"kubernetes/kubernetes": 46, "microsoft/vscode": 11}


def _tolerance_band(repo: str) -> float:
    """2x SEM band for `repo`, derived from the measured re-record jitter (ADR-0019)."""
    n = _JITTER_SEM_N[repo]
    sem = _MEASURED_JITTER_STD / (n ** 0.5)
    return round(2 * sem, 2)


def _write_baseline(scores: dict, path: Path = BASELINE_PATH) -> None:
    """Write scores to the baseline JSON file, adding threshold metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    per_repo_band = {repo: _tolerance_band(repo) for repo in scores["per_repo"]}
    payload = {
        "schema_version": "v2",
        "eval_set_hash": scores["eval_set_hash"],
        "cassette_hash": scores["cassette_hash"],
        "judge": {
            "model": JUDGE_MODEL,
            "provider": JUDGE_PROVIDER,
            "note": (
                "Local judge (ADR-0019) -- zero-cost, reproducible without a live key. "
                "Replaces llama-3.3-70b-versatile (Groq) used for the prior n=60 baseline. "
                "Means below are NOT comparable to that prior baseline: both the gold set "
                "(train-contamination removed, ADR-0018) and the judge model changed "
                "simultaneously -- this is a new baseline, not a corrected old one."
            ),
        },
        "per_repo": scores["per_repo"],
        "overall": scores["overall"],
        "threshold": {
            "method": (
                "One-directional per-repo mean regression check: fires only if "
                "new_mean < baseline_mean - band. Improvements above the band never trip it."
            ),
            "measured_jitter": {
                "source": (
                    "Two independent full local-judge re-recordings (attempt1 n=65, "
                    "attempt2 n=57 before an unrelated Groq synthesis TPD stop), compared "
                    "on 57 issues common to both."
                ),
                "std_per_issue_total_score": _MEASURED_JITTER_STD,
                "note": "Empirically measured, not guessed -- see ADR-0019.",
            },
            "per_repo_band": {
                repo: {
                    "band": band,
                    "sem": round(_MEASURED_JITTER_STD / (_JITTER_SEM_N[repo] ** 0.5), 4),
                    "n_used_for_sem": _JITTER_SEM_N[repo],
                }
                for repo, band in per_repo_band.items()
            },
            "vscode_note": (
                "microsoft/vscode's band (0.45) is wider than kubernetes/kubernetes's (0.22) "
                "because n=11 makes its mean statistically less stable -- this is the honest "
                "signal of the vscode data-ceiling finding (ADR-0017), not a fudged tolerance. "
                "Its gate is genuinely coarser."
            ),
            "derivation": "band = 2 x SEM = 2 x (measured_std / sqrt(n))",
        },
        "synthesis_quality_floor": {
            "note": (
                "ADR-0028 Phase B3: the mean-band gate above is a REGRESSION detector, not a "
                "quality floor -- it cannot catch fabrication (the judge never sees "
                "classifier_top3/retrieved_numbers) and averages away a correctness-critical "
                "floor-fail rate. per_repo[repo].fabrication_rate and .floor_fail_rate are "
                "additive first-class metrics, computed with zero new LLM calls (reused from "
                "plan.grounding_status and the judge dimension scores already produced)."
            ),
            "fabrication_rate": {
                "definition": "fraction of plans where plan.grounding_status.all_grounded is False "
                "(predicted_component not in classifier_top3, or any cited similar-issue number "
                "not in the retrieved set for that request).",
                "gate": "INFORMATIONAL ONLY (GG decision, 2026-07-11): eval/test_quality_regression.py "
                "asserts fabrication_rate == 0 per repo, but that file's CI job is "
                "continue-on-error:true -- visible, non-blocking, pending a real-world "
                "observation window before promoting to a hard gate.",
            },
            "floor_fail_rate": {
                "definition": "fraction of plans where component_match==0 OR "
                "similar_issues_relevance==0 (the judge's own worst-band signal).",
                "gate": "REPORT ONLY, no gate (GG decision, 2026-07-11): vscode's n=11 gives a "
                "[21,72]% CI on this proportion -- too underpowered to gate reliably. Surfaced "
                "here and in README for visibility instead.",
            },
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run eval pipeline in replay mode and print scores."
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=f"Write scores to {BASELINE_PATH} after printing.",
    )
    args = parser.parse_args()

    scores = compute_scores()

    print("\n=== EVAL SCORES ===")
    print(f"Eval set hash : {scores['eval_set_hash'][:16]}…")
    print(f"Cassette hash : {scores['cassette_hash'][:16]}…")
    print()
    for repo, data in scores["per_repo"].items():
        print(f"  {repo}: n={data['n']}, mean={data['mean']:.4f}/15")
        for dim, val in data["dimensions"].items():
            print(f"    {dim}: {val:.4f}")
        print(
            f"    floor_fail_rate: {data['floor_fail_rate']:.4f}   "
            f"fabrication_rate: {data['fabrication_rate']:.4f}"
        )
    print()
    print(f"  overall: n={scores['overall']['n']}, mean={scores['overall']['mean']:.4f}/15")

    if args.update_baseline:
        _write_baseline(scores)
        print(f"\nBaseline written to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
