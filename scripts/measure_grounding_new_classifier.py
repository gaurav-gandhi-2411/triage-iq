from __future__ import annotations
"""Phase 2 (ADR-0057 follow-on): re-check grounding against the NEWLY-PROMOTED classifier,
without spending any live LLM quota.

Why this can't just be `scripts/measure_grounding.py` unmodified: that script's assistant
calls `_collect_signals()` -> `_call_llm_verbose()`, and the LLM's prompt embeds
`classifier_top3` as text. Since the classifier was promoted (ADR-0057), that text now
differs from what's in the committed cassette (recorded under the OLD classifier's top3),
so every cache lookup misses -- cleanly failing closed
(`cassette.CassetteMissError`), not silently returning wrong data.

This script decouples the two things the working-agreement Phase 2 question is actually
about: (1) the LLM's own prediction, which doesn't need a new call -- it's already recorded
under the old classifier's prompt, and the model's synthesis judgment doesn't change just
because a downstream feature classifier was swapped; (2) which `classifier_top3` that
prediction gets checked against for grounding, which the working agreement wants
re-evaluated under the NEW classifier. So: build the request/get the cached plan using an
assistant wired to the OLD (archived) classifier -- guarantees a cassette hit, identical to
what actually shipped -- then separately compute the NEW classifier's top-3 for the same
issue text and re-run `compute_grounding_status` with that. Zero live calls either way.

Usage:
    .venv/Scripts/python.exe scripts/measure_grounding_new_classifier.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from cassette import CassettePlayer  # noqa: E402
from frozen_retriever import build_frozen_retrievers  # noqa: E402
from triage_iq.models.component_classifier import load_classifier  # noqa: E402
from triage_iq.models.grounding import compute_grounding_status  # noqa: E402
from triage_iq.models.resolution import ResolutionTimePredictor  # noqa: E402
from triage_iq.models.triage import TriageAssistant  # noqa: E402

MODELS_DIR = ROOT / "data" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"
CASSETTE_PATH = ROOT / "eval" / "cassettes" / "eval_cassette.json"
REPORT_PATH = ROOT / "reports" / "grounding_measurement_new_classifier.json"

REPO_MAP: dict[str, str] = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}
CI_API_KEY = "ci-replay-only"


def _load_eval_set(path: Path) -> list[dict]:
    issues: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                issues.append(json.loads(line))
    return issues


def _new_classifier_top3(classifier: Any, title: str, body: str) -> list[dict]:
    """Mirrors TriageAssistant._collect_signals' classifier_top3 construction exactly
    (src/triage_iq/models/triage.py:767-778), against a caller-supplied classifier
    instead of the assistant's own -- lets the LLM's already-recorded plan (from the OLD
    classifier's cassette entry) be checked against a DIFFERENT classifier's top-3."""
    text = f"{title}. {body}"
    proba = classifier.predict_proba_calibrated(pd.Series([text]))
    classes = classifier.classes_()
    top_idx = np.argsort(proba[0])[::-1][:3]
    return [{"label": classes[i], "confidence": float(proba[0][i])} for i in top_idx]


def compute_grounding_reports_new_classifier() -> list[dict]:
    cassette = CassettePlayer(CASSETTE_PATH, strict=True)
    issues = _load_eval_set(EVAL_SET_PATH)
    frozen_retrievers = build_frozen_retrievers(EVAL_SET_PATH)

    old_classifiers: dict[str, Any] = {}
    new_classifiers: dict[str, Any] = {}
    assistants: dict[str, TriageAssistant] = {}
    for repo, slug in REPO_MAP.items():
        # OLD classifier: reproduces the exact prompt the cassette was recorded against,
        # so _call_llm_verbose() gets a cache hit and returns the real recorded plan.
        old_clf = load_classifier(MODELS_DIR, f"{slug}_PRE_RETRAIN_2026-09-04")
        old_classifiers[repo] = old_clf
        # NEW classifier: the one now promoted to the deployed path (ADR-0057) --
        # what grounding is re-evaluated against.
        new_classifiers[repo] = load_classifier(MODELS_DIR, slug)

        predictor = ResolutionTimePredictor.load(str(MODELS_DIR / f"resolution_predictor_{slug}.pkl"))
        train_df = pd.read_parquet(PROCESSED_DIR / f"{slug}_temporal_train.parquet")
        assistants[repo] = TriageAssistant(
            repo=repo, classifier=old_clf, detector=frozen_retrievers[repo],
            predictor=predictor, train_df=train_df, groq_api_key=CI_API_KEY, cache=cassette,
        )

    cases: list[dict] = []
    for issue in issues:
        repo = issue["repo"]
        assistant = assistants[repo]

        row = pd.Series({
            "title": issue["title"],
            "body_clean": issue["body"],
            "number": issue["number"],
            "created_at": (
                pd.Timestamp(issue["created_at"]) if issue.get("created_at")
                else pd.Timestamp("now", tz="UTC")
            ),
        })

        signals = assistant._collect_signals(row)  # noqa: SLF001 -- OLD-classifier prompt, matches cassette
        plan, _raw, usage, llm_status, _cache_hit = assistant._call_llm_verbose(signals)  # noqa: SLF001

        retrieved_numbers = {s["number"] for s in signals["similar_raw"]}
        old_top3 = signals["classifier_top3"]
        new_top3 = _new_classifier_top3(new_classifiers[repo], str(issue["title"]), str(issue["body"]))

        resolved_new = compute_grounding_status(
            plan, new_top3, retrieved_numbers,
            enable_validated_override_rescue=False,  # working agreement 2d: not enabled
            issue_title=str(issue["title"]), issue_body=str(issue["body"]),
        )

        cases.append({
            "issue_number": issue["number"],
            "repo": repo,
            "predicted_component": plan.predicted_component,
            "old_classifier_top3_labels": [e["label"] for e in old_top3],
            "new_classifier_top3_labels": [e["label"] for e in new_top3],
            "component_grounded_new": resolved_new.component_grounded,
            "component_reason_new": resolved_new.component_reason,
            "all_grounded_new": resolved_new.all_grounded,
            "override_applied_new": resolved_new.override_applied,
            "retrieved_numbers": sorted(retrieved_numbers),
            "llm_status": llm_status,
            "finish_reason": usage.get("finish_reason") if isinstance(usage, dict) else None,
        })

    return cases


def main() -> None:
    cases = compute_grounding_reports_new_classifier()

    per_repo: dict[str, dict] = {}
    overall_n = 0
    overall_ungrounded = 0
    ungrounded_cases: list[dict] = []

    for repo in REPO_MAP:
        repo_cases = [c for c in cases if c["repo"] == repo]
        n = len(repo_cases)
        ungrounded = [c for c in repo_cases if not c["component_grounded_new"]]
        per_repo[repo] = {
            "n": n,
            "component_ungrounded_count": len(ungrounded),
            "component_ungrounded_pct": round(100 * len(ungrounded) / n, 2) if n else 0.0,
        }
        overall_n += n
        overall_ungrounded += len(ungrounded)
        ungrounded_cases.extend(ungrounded)

    result = {
        "per_repo": per_repo,
        "overall": {
            "n": overall_n,
            "component_ungrounded_count": overall_ungrounded,
            "component_ungrounded_pct": round(100 * overall_ungrounded / overall_n, 2) if overall_n else 0.0,
        },
        "ungrounded_cases": ungrounded_cases,
    }

    print("=== Grounding re-check against the newly-promoted classifier ===")
    print("(LLM plans unchanged -- reused from the committed cassette, zero live calls;")
    print(" only the classifier_top3 they're checked against has changed.)\n")
    for repo, r in per_repo.items():
        print(f"{repo}: {r['component_ungrounded_count']}/{r['n']} component-ungrounded "
              f"({r['component_ungrounded_pct']}%)")
    print(f"overall: {overall_ungrounded}/{overall_n} ({result['overall']['component_ungrounded_pct']}%)")
    print("\nUngrounded issues:")
    for c in ungrounded_cases:
        print(f"  #{c['issue_number']} ({c['repo']}): predicted={c['predicted_component']!r} "
              f"new_top3={c['new_classifier_top3_labels']}")

    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
