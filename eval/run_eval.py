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

from cassette import CassetteMissError, CassettePlayer
from frozen_retriever import build_frozen_retrievers
from triage_iq.evaluation.triage_eval import DIMENSION_MAX, JudgeScore, TriageJudge
from triage_iq.models.component_classifier import TFIDFComponentClassifier
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

JUDGE_MODEL = "llama-3.3-70b-versatile"
JUDGE_PROVIDER = "groq"
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
    classifier = TFIDFComponentClassifier.load(
        str(MODELS_DIR / f"component_classifier_{slug}.pkl")
    )
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
        cache=cassette,
    )

    dim_keys = list(DIMENSION_MAX.keys())
    repo_scores: dict[str, list[dict]] = {repo: [] for repo in REPO_MAP}

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

        try:
            plan, _meta = assistant.triage_with_metadata(row)
        except CassetteMissError:
            # TEMPORARY DIAGNOSTIC — see ADR-0011 cassette-key-divergence investigation.
            # Will be reverted once the CI-vs-local mismatch is root-caused.
            signals = assistant._collect_signals(row)
            print(
                "CASSETTE_MISS_DEBUG="
                + json.dumps(
                    {
                        "index": issues.index(issue),
                        "repo": repo,
                        "number": issue["number"],
                        "classifier_top3": signals["classifier_top3"],
                        "pred_days": signals["pred_days"],
                        "lo_days": signals["lo_days"],
                        "hi_days": signals["hi_days"],
                        "resolution_bucket": signals["resolution_bucket"],
                        "resolution_conf_pct": signals["resolution_conf_pct"],
                    },
                    sort_keys=True,
                )
            )
            raise

        plan_json = json.dumps(plan.model_dump(), ensure_ascii=False)
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

        per_repo[repo] = {
            "n": len(totals),
            "mean": round(float(np.mean(totals)), 4),
            "dimensions": dim_means,
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


def _write_baseline(scores: dict, path: Path = BASELINE_PATH) -> None:
    """Write scores to the baseline JSON file, adding threshold metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "v1",
        "eval_set_hash": scores["eval_set_hash"],
        "cassette_hash": scores["cassette_hash"],
        "per_repo": scores["per_repo"],
        "overall": scores["overall"],
        "threshold": {
            "absolute_drop": 0.0,
            "epsilon": 1e-4,
            "per_repo": True,
            "note": (
                "Gate fails if EITHER repo's mean drops below "
                "(baseline_mean - absolute_drop - epsilon)."
            ),
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
    print()
    print(f"  overall: n={scores['overall']['n']}, mean={scores['overall']['mean']:.4f}/15")

    if args.update_baseline:
        _write_baseline(scores)
        print(f"\nBaseline written to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
