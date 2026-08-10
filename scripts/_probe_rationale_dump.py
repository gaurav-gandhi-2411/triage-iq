from __future__ import annotations

"""One-off helper (not part of the investigation deliverable): re-runs a small set of
(issue, variant) pairs from probe_confidence_structural_variants.py and prints the FULL
plan JSON + judge rationale text, so the investigation writeup can quote concrete examples
instead of just a hedge-phrase word count. Zero Groq cost, local Ollama only. Deterministic
(temperature=0.0, seed=42) so results should match the full run exactly."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from frozen_retriever import build_frozen_retrievers
from triage_iq.evaluation.triage_eval import TriageJudge
from triage_iq.models.component_classifier import load_classifier
from triage_iq.models.grounding import verify_plan_grounding
from triage_iq.models.resolution import ResolutionTimePredictor, engineer_features
from triage_iq.models.triage import TriageAssistant

import probe_confidence_structural_variants as P

# (issue_id, variant_name) pairs to inspect closely
TARGETS = [
    ("k8s-14756", "baseline_v3"),
    ("k8s-14756", "candidate1_no_numbers"),
    ("k8s-14756", "candidate2_top1_only"),
    ("k8s-14363", "baseline_v3"),
    ("k8s-14363", "candidate2_top1_only"),  # the ungrounded fabrication case
    ("k8s-12703", "candidate2_top1_only"),
    ("k8s-12665", "candidate2_top1_only"),
    ("k8s-12224", "candidate1_no_numbers"),
]

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"


def main() -> None:
    issues_by_id = {}
    with open(EVAL_SET, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                issues_by_id[rec["id"]] = rec

    models_dir = ROOT / "data" / "models"
    processed_dir = ROOT / "data" / "processed"
    classifier = load_classifier(models_dir, "kubernetes_kubernetes")
    predictor = ResolutionTimePredictor.load(str(models_dir / "resolution_predictor_kubernetes_kubernetes.pkl"))
    train_df = pd.read_parquet(processed_dir / "kubernetes_kubernetes_temporal_train.parquet")
    frozen_retrievers = build_frozen_retrievers(EVAL_SET)
    detector = frozen_retrievers["kubernetes/kubernetes"]

    judge = TriageJudge(model=P.JUDGE_MODEL, provider="ollama", temperature=0.0, ollama_seed=P.OLLAMA_SEED, cache=None)
    judge._ollama_completion([{"role": "user", "content": "Reply with just: OK"}])

    for iid, variant_name in TARGETS:
        issue = issues_by_id[iid]
        system_prompt, few_shots, prompt_builder = P.VARIANTS[variant_name]

        text = f"{issue['title']}. {issue['body']}"
        proba = classifier.predict_proba_calibrated(pd.Series([text]))
        classes = classifier.classes_()
        top_idx = np.argsort(proba[0])[::-1][:3]
        classifier_top3 = [{"label": classes[i], "confidence": float(proba[0][i])} for i in top_idx]

        similar_raw = detector.retrieve(text, k=5, exclude_number=issue["number"])
        retrieved_numbers = {s["number"] for s in similar_raw}

        row_df = pd.DataFrame([{
            "title": issue["title"],
            "body_clean": issue["body"],
            "number": issue["number"],
            "created_at": pd.Timestamp(issue["created_at"]) if issue.get("created_at") else pd.Timestamp("now", tz="UTC"),
        }])
        feats, _ = engineer_features(row_df, train_df=train_df)
        for col in predictor.feature_names:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats[predictor.feature_names]
        pred_hrs = predictor.predict(feats)[0]
        lo_hrs, hi_hrs = predictor.predict_intervals(feats)
        pred_days, lo_days, hi_days = pred_hrs / 24.0, float(lo_hrs[0]) / 24.0, float(hi_hrs[0]) / 24.0

        user_prompt = prompt_builder(
            issue["title"], issue["body"], classifier_top3, similar_raw,
            pred_days, lo_days, hi_days, "kubernetes/kubernetes",
        )
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(few_shots)
        messages.append({"role": "user", "content": user_prompt})

        raw = P._ollama_synthesize(messages)
        plan = TriageAssistant._parse_plan(raw)
        report = verify_plan_grounding(plan, classifier_top3, retrieved_numbers)

        plan_json = json.dumps(plan.model_dump(exclude={"declared_attribution", "abstention_status"}), ensure_ascii=False)
        judge_score = judge.score(
            issue_title=issue["title"],
            issue_body=issue["body"][:600],
            triage_plan_json=plan_json,
            gold={
                "component": issue["gold_component"],
                "priority": issue["gold_priority"],
                "actual_resolution_days": issue["actual_resolution_days"],
            },
        )

        print(f"\n{'='*100}")
        print(f"{iid} / {variant_name}")
        print(f"{'='*100}")
        print(f"classifier_top3 (actual, unaffected by prompt text): {classifier_top3}")
        print(f"predicted_component: {plan.predicted_component!r}  grounded={report.component_grounded} ({report.component_reason})")
        print(f"expected_resolution_summary: {plan.expected_resolution_summary!r}")
        print(f"expected_resolution_lower_days/upper_days: {plan.expected_resolution_lower_days} / {plan.expected_resolution_upper_days}")
        print(f"suggested_next_steps: {plan.suggested_next_steps}")
        print(f"judge total: {judge_score.total()}/15  res={judge_score.resolution_estimate_reasonableness} next={judge_score.next_steps_actionability} comp={judge_score.component_match}")
        print(f"judge_rationale: {judge_score.judge_rationale}")


if __name__ == "__main__":
    main()
