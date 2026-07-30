from __future__ import annotations

"""Zero-judge, ~10-call probe for the ADR-0037 primary-pick anchoring fix.

The label-drift regression (kubectl-style generic-label drift under the confidence-framing
prompt fix) is measurable from synthesis output alone -- predicted_component vs. the
classifier's own top-1 -- so it doesn't need a full 128-call judge recording to test. This
script runs synthesis ONLY (no judge, no cassette) on a small, hand-picked set of k8s issues
that regressed under the first (un-anchored) prompt fix, live against Groq, and reports
whether the sharpened "primary pick by default" wording restores alignment with the
classifier's rank-1 pick.

Run locally with GROQ_API_KEY set (fetched from Secret Manager here to match CI's pattern).
Not part of the eval suite; not run in CI. One-off diagnostic per ADR-0037.

Superseded for prompt-ITERATION use by scripts/probe_prompt_structure_local.py (zero Groq
quota, local Ollama). This script (or the pattern it follows) is for the final
production-model confirmation of a structural signal found locally -- not for iterating on
wording, which is what actually burned quota here (two 9-call Groq runs, v2 and v3, when
only v3's needed to be on Groq).
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np
import pandas as pd

from frozen_retriever import build_frozen_retrievers
from triage_iq.models.component_classifier import load_classifier
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"

# The 9 k8s issues that changed predicted_component between the no-fix and first-fix
# recordings (2026-07-30): 4 clean correct->wrong flips vs. gold (12703, 12737, 14895,
# 14363 -- 3 of which drifted specifically to "kubectl"), the 1 issue that improved
# (13784, included as a control -- must NOT regress back), and 4 more label-changed
# issues where neither prediction was gold-correct but the drift pattern is still
# diagnostic (14550, 14935, 14477, 12665).
TARGET_ISSUE_IDS = [
    "k8s-12703", "k8s-12737", "k8s-14895", "k8s-14363",  # correct(no-fix) -> wrong(fix)
    "k8s-13784",  # control: improved under the un-anchored fix, must not regress
    "k8s-14550", "k8s-14935", "k8s-14477", "k8s-12665",  # other label drift
]

NOFIX_FIXED_LABELS = {
    # From the two already-recorded runs (2026-07-30 07:18 and 11:40 UTC), for reference
    # in the printed comparison. predicted_component only.
    "k8s-12703": ("ui", "kubectl"),
    "k8s-12737": ("apiserver", "api"),
    "k8s-14895": ("client-libraries", "kubectl"),
    "k8s-14363": ("cloudprovider", "kubectl"),
    "k8s-13784": ("cloudprovider", "kubelet"),
    "k8s-14550": ("nodecontroller", "test-infra"),
    "k8s-14935": ("kubelet", "isolation"),
    "k8s-14477": ("kube-proxy", "test-infra"),
    "k8s-12665": ("app-lifecycle", "usability"),
}


def _get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "GROQ_API_KEY not set. Fetch it and export it before running this script, e.g.:\n"
            "  GROQ_API_KEY=$(gcloud secrets versions access latest "
            "--secret=groq-api-key --project=triageiq-portfolio-495022)"
        )
    return key


def main() -> None:
    issues_by_id = {}
    with open(EVAL_SET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                issues_by_id[rec["id"]] = rec

    missing = [iid for iid in TARGET_ISSUE_IDS if iid not in issues_by_id]
    if missing:
        raise SystemExit(f"Issue IDs not found in eval_set.jsonl: {missing}")

    groq_key = _get_groq_key()

    models_dir = ROOT / "data" / "models"
    processed_dir = ROOT / "data" / "processed"
    classifier = load_classifier(models_dir, "kubernetes_kubernetes")
    predictor = ResolutionTimePredictor.load(
        str(models_dir / "resolution_predictor_kubernetes_kubernetes.pkl")
    )
    train_df = pd.read_parquet(processed_dir / "kubernetes_kubernetes_temporal_train.parquet")
    frozen_retrievers = build_frozen_retrievers(EVAL_SET)

    assistant = TriageAssistant(
        repo="kubernetes/kubernetes",
        classifier=classifier,
        detector=frozen_retrievers["kubernetes/kubernetes"],
        predictor=predictor,
        train_df=train_df,
        groq_api_key=groq_key,
        cache=None,  # no cassette -- always live, this is a one-off probe
    )

    # Success metric is GOLD accuracy, not clf_top1-matching -- an earlier probe run (v2,
    # "primary pick is the prediction, deviate only with concrete reason") showed 9/9
    # clf_top1-matching but 0/9 change vs. the v1 (un-anchored) recording already on file,
    # because the classifier's own top-1 is frequently wrong on this specific hand-picked
    # regression subset (by construction -- these are the issues where v1's LLM output
    # regressed). Matching clf_top1 is not the same as being right.
    print(f"{'issue':12} {'gold':20} {'clf_top1':20} {'no-fix':20} {'v1(un-anchored)':20} {'v3(this probe)':20}")
    n_calls = 0
    matches_gold = 0
    matches_v1 = 0
    for iid in TARGET_ISSUE_IDS:
        issue = issues_by_id[iid]
        row = pd.Series({
            "title": issue["title"],
            "body_clean": issue["body"],
            "number": issue["number"],
            "created_at": pd.Timestamp(issue["created_at"]) if issue.get("created_at") else pd.Timestamp("now", tz="UTC"),
        })

        text = f"{issue['title']}. {issue['body']}"
        proba = classifier.predict_proba_calibrated(pd.Series([text]))
        classes = classifier.classes_()
        clf_top1 = classes[int(np.argmax(proba[0]))]

        plan, meta = assistant.triage_with_metadata(row)
        n_calls += 1
        probe_label = plan.predicted_component

        gold = issue["gold_component"]
        nofix_label, v1_label = NOFIX_FIXED_LABELS[iid]
        tag = "GOLD" if probe_label == gold else ("same-as-v1" if probe_label == v1_label else "")
        if probe_label == gold:
            matches_gold += 1
        if probe_label == v1_label:
            matches_v1 += 1
        print(f"{iid:12} {gold:20} {clf_top1:20} {nofix_label:20} {v1_label:20} {probe_label:20} [{tag}]")

    nofix_gold_matches = sum(
        1 for iid in TARGET_ISSUE_IDS if NOFIX_FIXED_LABELS[iid][0] == issues_by_id[iid]["gold_component"]
    )
    v1_gold_matches = sum(
        1 for iid in TARGET_ISSUE_IDS if NOFIX_FIXED_LABELS[iid][1] == issues_by_id[iid]["gold_component"]
    )
    print(f"\n{n_calls} live Groq calls made.")
    print(f"Matches gold: {matches_gold}/{len(TARGET_ISSUE_IDS)}  "
          f"(no-fix baseline: {nofix_gold_matches}/{len(TARGET_ISSUE_IDS)}, "
          f"v1/un-anchored baseline: {v1_gold_matches}/{len(TARGET_ISSUE_IDS)})")
    print(f"Identical to v1 (un-anchored) output: {matches_v1}/{len(TARGET_ISSUE_IDS)}")


if __name__ == "__main__":
    main()
