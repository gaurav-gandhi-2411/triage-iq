from __future__ import annotations

"""Zero-Groq-quota probe for STRUCTURAL prompt-behavior questions (ADR-0037 and future
prompt iterations): does the LLM anchor to the classifier's top-1? does it hedge on
resolution/next-steps when confidence scores cluster? These are questions about how a
chat LLM responds to this prompt's structure and framing, not questions that require
production's exact model -- so this script builds the IDENTICAL messages
(SYSTEM_PROMPT_LEGACY + the four few-shot examples + build_triage_prompt()'s user turn,
via the same functions production imports) and sends them to a LOCAL Ollama model
instead of Groq. Also runs the SAME judge production/CI already uses (qwen3:8b via
Ollama, ADR-0019) against each plan -- the judge was already zero-cost, so there was no
reason the label-drift-only version of this script left hedging tone (the
resolution_estimate_reasonableness / next_steps_actionability dimensions, and the judge's
own rationale text) unmeasured. Both checks this script was built for -- top-1 adherence
and hedging tone -- are covered here for zero Groq cost.

*** IMPORTANT CAVEAT, read before trusting a result from this script -- THIS IS NOT A
VALIDATION GATE, IT HAS ALREADY BEEN WRONG ONCE ***
Results here indicate STRUCTURAL prompt behavior (anchoring, hedging, label drift) under
a locally-run model. They are NOT production-identical output -- production synthesis is
served by Groq's llama-3.1-8b-instant, a specific hosted model/quantization/serving stack
that this script does not reproduce. Default model below (llama3.1:8b via Ollama) is the
closest local match by base architecture, but is still a different weights/serving path.
Treat a "fixed here" result as, at most, a weak prior that MIGHT justify spending the real
128-call Groq recording to confirm -- never as a substitute for that recording, and never
as grounds to write a new eval baseline.

CONFIRMED FALSE POSITIVE (2026-08-05, ADR-0037 v3): a 24-issue run of this script against
the v3 prompt (labels-removed variant) showed zero hedging language in judge rationales and
partial label recovery on the hardest subset -- a clean "go" signal. The subsequent full
64-issue LIVE Groq + real production classifier recording showed the opposite: k8s mean
regressed -0.62 (2.8x the project's own ±0.22 noise band), and de-hedging did NOT hold --
resolution_estimate_reasonableness and next_steps_actionability both landed BELOW the OLD
pre-cutover baseline, worse than v1 achieved on that same axis. Root cause of the mismatch
is not fully understood, but the practical lesson is unambiguous either way: local synthesis
(llama3.1:8b via Ollama) did NOT predict production synthesis (Groq llama-3.1-8b-instant)
behavior for this class of confidence-framing/hedging prompt question, at a scale (24 vs 64
issues) and cost (zero vs ~215K tokens) that made the discrepancy expensive to discover late
rather than cheap to discover early. See ADR-0037's "Local-probe invalidation" section for
the full writeup. Do not resume trusting this script's signal as gating evidence without
first re-reading that section.

The eval baseline is defined by what production actually serves; this script exists so
prompt iteration doesn't have to pay Groq's TPD budget on every attempt, only once, at the
end, to confirm -- treat "once, at the end, to confirm" as load-bearing, not optional.

Ported from scripts/probe_label_anchoring_fix.py (2026-07-30) after two Groq-based cheap
probes (18 calls total) were spent testing prompt-wording variants that could have been
tested here for zero quota cost -- see ADR-0037's cost accounting.

Requires a local Ollama server (`ollama serve`) with the target model pulled
(`ollama pull llama3.1:8b`). Not part of the eval suite; not run in CI.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np
import pandas as pd

from frozen_retriever import build_frozen_retrievers
from triage_iq.evaluation.triage_eval import DIMENSION_MAX, TriageJudge
from triage_iq.models.component_classifier import load_classifier
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant
from triage_iq.prompts.triage_prompt import SYSTEM_PROMPT_LEGACY, build_few_shot_examples_legacy, build_triage_prompt

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"

# Closest local base-architecture match to production's llama-3.1-8b-instant (Groq). Still
# a different weights/quantization/serving stack -- see the module docstring caveat.
LOCAL_MODEL = "llama3.1:8b"
OLLAMA_SEED = 42
JUDGE_MODEL = "qwen3:8b"  # same model production/CI use for judging (ADR-0019) -- this part
# IS production-identical, since the judge already runs on local Ollama everywhere.

# The original 9-issue regression set (see scripts/probe_label_anchoring_fix.py's comment
# for how it was chosen) plus 15 more k8s issues sampled (seed=42, from the 44 not in that
# original 9) to broaden coverage for a v3-validation pass beyond the cherry-picked worst
# cases -- includes k8s-13270/14756 (the hedging examples from ADR-0037's diagnosis, one
# recovered under v1, one didn't) and k8s-14281 (the single worst score drop / near-tie
# label flip). Default here is the full 24; pass a smaller list at the call site if you
# only need the quick 9-issue check.
TARGET_ISSUE_IDS = [
    "k8s-12224", "k8s-12248", "k8s-12287", "k8s-12665", "k8s-12703", "k8s-12737",
    "k8s-12828", "k8s-13257", "k8s-13270", "k8s-13435", "k8s-13508", "k8s-13784",
    "k8s-14125", "k8s-14135", "k8s-14190", "k8s-14281", "k8s-14363", "k8s-14477",
    "k8s-14550", "k8s-14711", "k8s-14756", "k8s-14762", "k8s-14895", "k8s-14935",
]

# All 53 k8s issues' predicted_component from the two Groq-recorded full runs (no-fix:
# 2026-07-30T07:18 UTC, v1/un-anchored fix: 2026-07-30T11:40 UTC) -- for reference only,
# not required (TARGET_ISSUE_IDS can be any subset; issues without an entry here just print
# "-" in those columns instead of blocking).
NOFIX_V1_LABELS = {
    "k8s-12122": ("cloudprovider", "cloudprovider"),
    "k8s-12123": ("kubectl", "kubectl"),
    "k8s-12224": ("example", "example"),
    "k8s-12248": ("kubectl", "kubectl"),
    "k8s-12254": ("test-infra", "test-infra"),
    "k8s-12277": ("nodecontroller", "nodecontroller"),
    "k8s-12284": ("cloudprovider", "cloudprovider"),
    "k8s-12287": ("cloudprovider", "cloudprovider"),
    "k8s-12461": ("kubectl", "kubectl"),
    "k8s-12477": ("apiserver", "apiserver"),
    "k8s-12587": ("os/coreos", "os/coreos"),
    "k8s-12615": ("api", "api"),
    "k8s-12665": ("app-lifecycle", "usability"),
    "k8s-12703": ("ui", "kubectl"),
    "k8s-12737": ("apiserver", "api"),
    "k8s-12784": ("apiserver", "apiserver"),
    "k8s-12818": ("test-infra", "test-infra"),
    "k8s-12828": ("kubectl", "kubectl"),
    "k8s-12853": ("os/coreos", "os/coreos"),
    "k8s-13034": ("cloudprovider", "cloudprovider"),
    "k8s-13057": ("api", "usability"),
    "k8s-13062": ("test-infra", "test-infra"),
    "k8s-13096": ("api", "api"),
    "k8s-13257": ("test-infra", "test-infra"),
    "k8s-13270": ("kube-proxy", "kube-proxy"),
    "k8s-13276": ("ui", "ui"),
    "k8s-13435": ("usability", "usability"),
    "k8s-13470": ("kubelet", "kubelet"),
    "k8s-13508": ("cloudprovider", "cloudprovider"),
    "k8s-13784": ("cloudprovider", "kubelet"),
    "k8s-13995": ("test-infra", "test-infra"),
    "k8s-14009": ("monitoring", "monitoring"),
    "k8s-14054": ("kubelet", "kubelet"),
    "k8s-14078": ("test-infra", "test-infra"),
    "k8s-14125": ("test-infra", "test-infra"),
    "k8s-14135": ("test-infra", "test-infra"),
    "k8s-14190": ("api", "api"),
    "k8s-14281": ("kubectl", "kubectl"),
    "k8s-14363": ("cloudprovider", "kubectl"),
    "k8s-14424": ("test-infra", "test-infra"),
    "k8s-14477": ("kube-proxy", "test-infra"),
    "k8s-14550": ("nodecontroller", "test-infra"),
    "k8s-14552": ("test-infra", "test-infra"),
    "k8s-14553": ("platform/mesos", "platform/mesos"),
    "k8s-14557": ("kubelet", "kubelet"),
    "k8s-14711": ("introspection", "introspection"),
    "k8s-14723": ("security", "security"),
    "k8s-14756": ("kube-proxy", "kube-proxy"),
    "k8s-14762": ("kubelet", "introspection"),
    "k8s-14785": ("build-release", "build-release"),
    "k8s-14835": ("build-release", "build-release"),
    "k8s-14895": ("client-libraries", "kubectl"),
    "k8s-14935": ("kubelet", "isolation"),
}


def _ollama_synthesize(messages: list[dict]) -> str:
    """Same client pattern as TriageJudge._ollama_completion (triage_eval.py) -- local
    server, no rate limit, retry only covers model-loading/connection races."""
    import time as _time

    try:
        import ollama as _ollama
    except ImportError as e:
        raise ImportError("pip install ollama") from e

    client = _ollama.Client()
    backoff = 3.0
    for attempt in range(6):
        try:
            resp = client.chat(
                model=LOCAL_MODEL,
                messages=messages,
                keep_alive=-1,
                options={"temperature": 0.0, "seed": OLLAMA_SEED},
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            if attempt < 5:
                print(f"  Ollama attempt {attempt + 1}/6 failed: {e} -- retrying in {backoff:.1f}s")
                _time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            else:
                raise
    raise RuntimeError("Ollama completion failed after 6 attempts")


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

    models_dir = ROOT / "data" / "models"
    processed_dir = ROOT / "data" / "processed"
    classifier = load_classifier(models_dir, "kubernetes_kubernetes")
    predictor = ResolutionTimePredictor.load(
        str(models_dir / "resolution_predictor_kubernetes_kubernetes.pkl")
    )
    train_df = pd.read_parquet(processed_dir / "kubernetes_kubernetes_temporal_train.parquet")
    frozen_retrievers = build_frozen_retrievers(EVAL_SET)
    detector = frozen_retrievers["kubernetes/kubernetes"]

    few_shots = build_few_shot_examples_legacy()

    judge = TriageJudge(model=JUDGE_MODEL, provider="ollama", temperature=0.0, ollama_seed=OLLAMA_SEED, cache=None)
    print("Ollama judge warm-up call (absorbing cold-start variance, ADR-0019)...")
    judge._ollama_completion([{"role": "user", "content": "Reply with just: OK"}])

    print(f"\n*** LOCAL MODEL ({LOCAL_MODEL} via Ollama) -- structural probe, NOT production-identical ***\n")
    print(f"{'issue':12} {'gold':20} {'clf_top1':20} {'no-fix':20} {'v1(un-anchored)':20} {'local probe':20}")
    n_calls = 0
    matches_gold = 0
    matches_v1 = 0
    judge_totals = []
    resolution_est_scores = []
    next_steps_scores = []
    component_match_scores = []
    for iid in TARGET_ISSUE_IDS:
        issue = issues_by_id[iid]

        text = f"{issue['title']}. {issue['body']}"
        proba = classifier.predict_proba_calibrated(pd.Series([text]))
        classes = classifier.classes_()
        top_idx = np.argsort(proba[0])[::-1][:3]
        classifier_top3 = [{"label": classes[i], "confidence": float(proba[0][i])} for i in top_idx]
        clf_top1 = classifier_top3[0]["label"]

        similar_raw = detector.retrieve(text, k=5, exclude_number=issue["number"])

        # Mirrors TriageAssistant._collect_signals exactly (triage.py) -- engineer_features
        # is driven by title/body/number/created_at, not gold labels (production has no gold
        # labels at inference time; the gold_component/gold_priority rename branch in
        # _collect_signals is for a different caller and is a no-op on this exact row shape).
        row_df = pd.DataFrame([{
            "title": issue["title"],
            "body_clean": issue["body"],
            "number": issue["number"],
            "created_at": pd.Timestamp(issue["created_at"]) if issue.get("created_at") else pd.Timestamp("now", tz="UTC"),
        }])
        from triage_iq.models.resolution import engineer_features
        feats, _ = engineer_features(row_df, train_df=train_df)
        for col in predictor.feature_names:
            if col not in feats.columns:
                feats[col] = 0.0
        feats = feats[predictor.feature_names]
        pred_hrs = predictor.predict(feats)[0]
        lo_hrs, hi_hrs = predictor.predict_intervals(feats)
        pred_days, lo_days, hi_days = pred_hrs / 24.0, float(lo_hrs[0]) / 24.0, float(hi_hrs[0]) / 24.0

        user_prompt = build_triage_prompt(
            issue_title=issue["title"],
            issue_body=issue["body"],
            classifier_top3=classifier_top3,
            similar_issues=similar_raw,
            resolution_point_days=pred_days,
            resolution_lower_days=lo_days,
            resolution_upper_days=hi_days,
            repo="kubernetes/kubernetes",
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT_LEGACY}]
        messages.extend(few_shots)
        messages.append({"role": "user", "content": user_prompt})

        raw = _ollama_synthesize(messages)
        n_calls += 1
        try:
            plan = TriageAssistant._parse_plan(raw)
            probe_label = plan.predicted_component
        except Exception as e:
            probe_label = f"<parse failed: {e}>"
            plan = None

        gold = issue["gold_component"]
        nofix_label, v1_label = NOFIX_V1_LABELS.get(iid, ("-", "-"))
        tag = "GOLD" if probe_label == gold else ("same-as-v1" if probe_label == v1_label else "")
        if probe_label == gold:
            matches_gold += 1
        if probe_label == v1_label:
            matches_v1 += 1
        print(f"{iid:12} {gold:20} {clf_top1:20} {nofix_label:20} {v1_label:20} {probe_label:20} [{tag}]")

        if plan is None:
            print("  (judge skipped -- plan failed to parse)")
            continue

        # Local judge, zero Groq cost -- exclude fields to match run_eval.py/record_cassettes.py's
        # plan_json construction (ADR-0020/ADR-0021), though this is a probe, not a cassette entry.
        plan_json = json.dumps(
            plan.model_dump(exclude={"declared_attribution", "abstention_status"}),
            ensure_ascii=False,
        )
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
        judge_totals.append(judge_score.total())
        resolution_est_scores.append(judge_score.resolution_estimate_reasonableness)
        next_steps_scores.append(judge_score.next_steps_actionability)
        component_match_scores.append(judge_score.component_match)
        print(
            f"  judge: {judge_score.total()}/{sum(DIMENSION_MAX.values())}  "
            f"resolution_est={judge_score.resolution_estimate_reasonableness}  "
            f"next_steps={judge_score.next_steps_actionability}  "
            f"component_match={judge_score.component_match}"
        )
        print(f"  rationale: {judge_score.judge_rationale}")

    print(f"\n{n_calls} local Ollama synthesis calls + {len(judge_totals)} local Ollama judge calls made. Zero Groq tokens spent.")
    print(f"Matches gold: {matches_gold}/{len(TARGET_ISSUE_IDS)}  ({matches_gold/len(TARGET_ISSUE_IDS):.1%})")
    print(f"Identical to v1 (un-anchored, Groq) output: {matches_v1}/{len(TARGET_ISSUE_IDS)}")
    if judge_totals:
        print(f"Local-judge mean total: {np.mean(judge_totals):.2f}/{sum(DIMENSION_MAX.values())}")
        print(f"Local-judge mean resolution_estimate_reasonableness: {np.mean(resolution_est_scores):.2f}/{DIMENSION_MAX['resolution_estimate_reasonableness']}  "
              f"(the primary de-hedging signal from ADR-0037)")
        print(f"Local-judge mean next_steps_actionability: {np.mean(next_steps_scores):.2f}/{DIMENSION_MAX['next_steps_actionability']}  "
              f"(the other de-hedging signal)")
        print(f"Local-judge mean component_match: {np.mean(component_match_scores):.2f}/{DIMENSION_MAX['component_match']}")
    print(
        "\nReminder: this is a structural signal on a different model than production. "
        "A promising result here justifies spending the Groq recording to confirm; "
        "it does not replace it."
    )


if __name__ == "__main__":
    main()
