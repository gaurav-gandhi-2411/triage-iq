from __future__ import annotations

"""Zero-Groq-quota structural probe for the ADR-0037 "untested lever": not a fifth prompt-WORDING
variant (four of those were tried and failed, see ADR-0037), but a structural change to HOW the
classifier's top-3 confidence is REPRESENTED in the prompt. Tests two candidates against the
current live (v3) prompt as the local baseline:

  - baseline (v3, live today): "N. label (confidence: 0.XXX)" x3, framed as independent
    per-component probabilities.
  - candidate1: rank-ordered list, NO numbers at all -- "N. label" x3.
  - candidate2: top-1 only, single calibrated confidence -- no alternates shown to the LLM at all.
    Internally, classifier_top3 (the full top-3) is UNCHANGED and still passed to grounding
    verification exactly as today -- only the LLM-facing prompt TEXT differs. This probe verifies
    that by running verify_plan_grounding() with the real (unedited) classifier_top3 against each
    candidate's plan, independent of what the LLM was shown.

*** SAME CAVEAT AS scripts/probe_prompt_structure_local.py -- READ IT BEFORE TRUSTING A RESULT ***
This is a LOCAL Ollama probe (llama3.1:8b), not production (Groq llama-3.1-8b-instant). ADR-0037's
v3 iteration showed a clean local "go" signal that inverted on the real 64-issue Groq recording
(k8s mean regressed -0.62, de-hedging did NOT hold). That was a WORDING variant; this probe tests
STRUCTURAL variants instead, which is a materially different, more mechanical question (does the
model treat "no numbers shown" or "one number shown" differently, vs. "does prose elsewhere
override an anchoring label" -- the question v2's probe asked, which DID replicate cleanly). But
that distinction is a hypothesis, not a proven exemption from the same local/production gap -- this
script is a CHEAP FILTER, not a validation gate. A clean result here is grounds to recommend
spending the real Groq confirmation recording sooner; it is never a substitute for it.

Uses the SAME 12-issue k8s subset (of the docstring's TARGET_ISSUE_IDS in
probe_prompt_structure_local.py) for cost -- includes both hedging examples named in ADR-0037
(k8s-13270, k8s-14756), the single worst score-drop/near-tie label flip (k8s-14281), two issues
that recovered under v3 (k8s-12703, k8s-14363), and the one confirmed classifier-error case (not a
prompt problem, k8s-14895), plus 6 more for general spread. Not the full 24 -- three prompt variants
x 12 issues x (1 synthesis + 1 judge call) = 72 local calls is already a meaningful time cost for a
cheap filter; the investigation writeup states this choice explicitly.

Requires a local Ollama server (`ollama serve`) with llama3.1:8b and qwen3:8b pulled. Not part of
the eval suite; not run in CI.
"""

import json
import sys
import time as _time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np
import pandas as pd

from frozen_retriever import build_frozen_retrievers
from triage_iq.evaluation.triage_eval import DIMENSION_MAX, TriageJudge
from triage_iq.models.component_classifier import load_classifier
from triage_iq.models.grounding import verify_plan_grounding
from triage_iq.models.resolution import ResolutionTimePredictor, engineer_features
from triage_iq.models.triage import TriageAssistant

EVAL_SET = ROOT / "eval" / "eval_set.jsonl"

LOCAL_MODEL = "llama3.1:8b"
OLLAMA_SEED = 42
JUDGE_MODEL = "qwen3:8b"

# 12-issue subset of probe_prompt_structure_local.py's 24-issue TARGET_ISSUE_IDS -- see module
# docstring for the selection rationale.
TARGET_ISSUE_IDS = [
    "k8s-13270",  # ADR-0037 hedging example 1 (gold=kube-proxy, correct both times, judge dinged
                  # resolution/next-steps anyway)
    "k8s-14756",  # ADR-0037 hedging example 2 (same pattern)
    "k8s-14281",  # single worst score drop (-5), near-tie label flip (app-lifecycle vs kubectl)
    "k8s-12703",  # label flip that recovered under v3 (ui->kubectl under v1, back to ui under v3)
    "k8s-14363",  # same, recovered under v3 (cloudprovider->kubectl under v1)
    "k8s-14895",  # confirmed classifier error, not a prompt problem (classifier's own top-1 is
                  # wrong; GG's call was not to force an override)
    "k8s-12224", "k8s-12665", "k8s-12828", "k8s-13435", "k8s-14135", "k8s-14711",  # spread
]

# ---------------------------------------------------------------------------
# System-prompt variants. Only the CLASSIFIER CONFIDENCE GUIDANCE paragraph
# differs between variants; everything else in SYSTEM_PROMPT_LEGACY is unchanged.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_HEAD = """\
You are an expert GitHub issue triager for large open-source software projects.
You will be given an issue along with signals from automated classification and retrieval systems.
Your job is to produce a structured triage plan in valid JSON.

Output ONLY valid JSON matching the schema below. Do not include any prose before or after the JSON block.

PRIORITY GUIDELINES:
1. low — cosmetic or non-blocking; affects an edge case or niche workflow; trivial workaround exists; or long-standing minor annoyance.
2. medium — reproducible regression with a workaround available; feature gap affecting an active workflow; or visual inconsistency in a core feature.
3. high — crash, data loss, auth failure, or broken core workflow with NO workaround for any user.
4. Resource-leak / memory-leak framing does NOT automatically imply high. Assign medium unless the leak also causes a crash or completely blocks usage.
5. Empty or image-only body: assign priority based on the title alone. If the title is ambiguous and does not indicate a crash or data loss, default to medium.

"""

_SCHEMA_TAIL = """
Schema:
{
  "predicted_component": "string — the single best component label for this issue",
  "component_confidence": "number 0.0–1.0 — your confidence in the component assignment",
  "similar_issues": [
    {
      "number": "integer — issue number",
      "similarity": "number 0.0–1.0 — semantic similarity score",
      "relevance_note": "string — one sentence on why this is related"
    }
  ],
  "expected_resolution_summary": "string — human-readable estimate (e.g., '2–7 days typical for this component')",
  "expected_resolution_lower_days": "number — optimistic estimate in days",
  "expected_resolution_upper_days": "number — conservative estimate in days",
  "priority_guess": "one of: low | medium | high",
  "priority_rationale": "string — 1–2 sentences explaining priority assignment",
  "suggested_assignee_class": "string — team or role best suited (e.g., 'core-runtime team', 'documentation team', 'first-time-contributor friendly')",
  "suggested_next_steps": ["string — ordered list of 2–4 actionable next steps"],
  "triage_summary": "string — 2–3 sentence executive summary of the issue and recommended action"
}
"""

SYSTEM_PROMPT_BASELINE_V3 = (
    _SYSTEM_PROMPT_HEAD
    + """\
CLASSIFIER CONFIDENCE GUIDANCE:
The component classifier in SYSTEM 1 reports independent per-component probabilities, not a single
normalized distribution, so several components scoring similarly is expected and does not mean the
classifier is unsure. Weigh it together with the issue text and the similar issues below, the same
way you always would — a close spread is additional context, not an instruction toward or away from
any particular entry. The resolution estimate and next steps are produced by separate, independent
models — do not soften or hedge them because the classifier's scores happen to be close together.
"""
    + _SCHEMA_TAIL
)

SYSTEM_PROMPT_CANDIDATE1_NO_NUMBERS = (
    _SYSTEM_PROMPT_HEAD
    + """\
CLASSIFIER CONFIDENCE GUIDANCE:
The component classifier in SYSTEM 1 reports a ranked list of candidate components, most to least
likely, with no numeric scores shown. Treat #1 as the classifier's lead assessment. The resolution
estimate and next steps are produced by separate, independent models — do not soften or hedge them
based on how the classifier ranked its component guesses.
"""
    + _SCHEMA_TAIL
)

SYSTEM_PROMPT_CANDIDATE2_TOP1_ONLY = (
    _SYSTEM_PROMPT_HEAD
    + """\
CLASSIFIER CONFIDENCE GUIDANCE:
The component classifier in SYSTEM 1 reports its single top prediction with a calibrated confidence
score. Weigh it together with the issue text and the similar issues below, the same way you always
would. The resolution estimate and next steps are produced by separate, independent models — do not
soften or hedge them because of the classifier's confidence value.
"""
    + _SCHEMA_TAIL
)


# ---------------------------------------------------------------------------
# User-turn builders (mirror build_triage_prompt() in triage_prompt.py, but with the
# SYSTEM 1 section rendered per-candidate). Signature-compatible callers pass the full
# classifier_top3 in every case -- ONLY the rendered text differs.
# ---------------------------------------------------------------------------


def _common_sections(
    issue_title: str,
    issue_body: str,
    similar_issues: list[dict],
    resolution_point_days: float,
    resolution_lower_days: float,
    resolution_upper_days: float,
    repo: str,
) -> tuple[str, str, str]:
    body_preview = issue_body[:800].strip() if issue_body else "(no body)"
    similar_lines = "\n".join(
        f"  #{s['number']} (similarity: {s['score']:.3f}): {s['text'][:120]}..."
        for s in similar_issues[:5]
    )
    header = f"Repository: {repo}\n\n--- ISSUE ---\nTitle: {issue_title}\nBody:\n{body_preview}\n"
    system2 = f"\n--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---\n{similar_lines}\n"
    system3 = (
        f"\n--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---\n"
        f"Point estimate: {resolution_point_days:.1f} days\n"
        f"80% prediction interval: [{resolution_lower_days:.1f}d, {resolution_upper_days:.1f}d]\n"
        "Note: These estimates are trained on historical data and may not account for current team velocity or issue priority changes.\n"
    )
    task = (
        "\n--- TASK ---\n"
        "Produce a triage plan as valid JSON matching the schema in the system prompt.\n"
        "Use the classifier signals, similar issues, and resolution estimate to inform your plan.\n"
        "Be specific and actionable. Do not hallucinate issue numbers not listed above.\n"
    )
    return header, system2, system3 + task


def build_prompt_baseline_v3(issue_title, issue_body, classifier_top3, similar_issues, *args) -> str:
    header, system2, tail = _common_sections(issue_title, issue_body, similar_issues, *args)
    lines = "\n".join(
        f"  {i+1}. {c['label']} (confidence: {c['confidence']:.3f})" for i, c in enumerate(classifier_top3)
    )
    system1 = (
        "\n--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---\n"
        "These are independent per-component probabilities, not a single normalized distribution: each score\n"
        'answers "does this component apply?" on its own, so it is normal and expected for two or three\n'
        "components to score similarly when an issue plausibly touches more than one area — this does not by\n"
        "itself mean the classifier is unsure. Weigh these scores together with the issue text and the similar\n"
        "issues below, the same way you always would.\n"
        f"Top-3 predictions:\n{lines}\n"
    )
    return header + system1 + system2 + tail


def build_prompt_candidate1_no_numbers(issue_title, issue_body, classifier_top3, similar_issues, *args) -> str:
    header, system2, tail = _common_sections(issue_title, issue_body, similar_issues, *args)
    lines = "\n".join(f"  {i+1}. {c['label']}" for i, c in enumerate(classifier_top3))
    system1 = (
        "\n--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---\n"
        "Ranked component predictions (most to least likely), from an automated classifier. Rank order is\n"
        "the classifier's signal here — treat #1 as its lead assessment.\n"
        f"{lines}\n"
    )
    return header + system1 + system2 + tail


def build_prompt_candidate2_top1_only(issue_title, issue_body, classifier_top3, similar_issues, *args) -> str:
    header, system2, tail = _common_sections(issue_title, issue_body, similar_issues, *args)
    top1 = classifier_top3[0]
    system1 = (
        "\n--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---\n"
        f"Predicted component (automated classifier): {top1['label']} (confidence: {top1['confidence']:.3f})\n"
    )
    return header + system1 + system2 + tail


# ---------------------------------------------------------------------------
# Few-shot examples. The three frozen (ADR-0020) LOW/MEDIUM/HIGH examples are reused
# byte-identical across all three variants (untouched, as production would keep them).
# Only the appended 4th ("clustered confidence") example's user turn is rewritten per
# variant, mirroring exactly what ADR-0037's v3 iteration did when it changed the live
# format (see triage_prompt.py's build_few_shot_examples_legacy() module comment).
# ---------------------------------------------------------------------------

from triage_iq.prompts.triage_prompt import build_few_shot_examples_legacy

_FROZEN_THREE = build_few_shot_examples_legacy()[:6]  # 3 user/assistant pairs, untouched

_FOURTH_ASSISTANT_TURN = """\
{
  "predicted_component": "settings-sync",
  "component_confidence": 0.579,
  "similar_issues": [
    {"number": 15210, "similarity": 0.742, "relevance_note": "Same failure mode -- extension-contributed settings not restored after a profile switch."},
    {"number": 14889, "similarity": 0.681, "relevance_note": "Same intermittent pattern isolated to extension-contributed keys specifically, not global settings."}
  ],
  "expected_resolution_summary": "Cross-cutting sync timing bug between profile switch and extension settings restore; typically 1-3 weeks once the race condition is isolated.",
  "expected_resolution_lower_days": 2.8,
  "expected_resolution_upper_days": 21.6,
  "priority_guess": "medium",
  "priority_rationale": "Reproducible regression with real user impact (extension settings silently drop), but a manual re-sync works around it and the failure rate (~1 in 5) is partial, not universal.",
  "suggested_assignee_class": "settings-sync team",
  "suggested_next_steps": [
    "Reproduce with a minimal profile containing a single extension with custom settings, sync, then switch profiles 10x to isolate the race window.",
    "Check whether extension-contributed settings restore is awaited before the profile switch reports complete.",
    "Confirm #15210 and #14889 are duplicates of this or distinct manifestations of the same race."
  ],
  "triage_summary": "Settings Sync drops extension-specific settings on roughly 1 in 5 profile switches while global settings sync reliably every time, pointing to a timing race in the extension-settings restore path specifically. Two closely related prior reports confirm this is a recurring pattern, not a one-off. Assign to the settings-sync team; medium priority given the available workaround."
}"""

_FOURTH_USER_HEADER = """\
Repository: microsoft/vscode

--- ISSUE ---
Title: Settings Sync intermittently drops extension-specific settings after profile switch
Body:
When switching between user profiles with Settings Sync enabled, per-extension settings (e.g. formatter configuration, linter rules) occasionally fail to reapply after the switch completes. Reproducible roughly 1 in 5 switches on VS Code 1.88.0. Global (non-extension) settings sync correctly every time. Re-triggering sync manually usually fixes it.
"""

_FOURTH_TAIL = """
--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
  #15210 (similarity: 0.742): Extension settings not restored after switching profiles on sync...
  #14889 (similarity: 0.681): Intermittent settings sync failure specific to extension-contributed keys...

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: 9.4 days
80% prediction interval: [2.8d, 21.6d]
Note: These estimates are trained on historical data and may not account for current team velocity or issue priority changes.

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
Use the classifier signals, similar issues, and resolution estimate to inform your plan.
Be specific and actionable. Do not hallucinate issue numbers not listed above.
"""


def few_shots_baseline_v3() -> list[dict]:
    user = (
        _FOURTH_USER_HEADER
        + "\n--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---\n"
        "These are independent per-component probabilities, not a single normalized distribution: each score\n"
        'answers "does this component apply?" on its own, so it is normal and expected for two or three\n'
        "components to score similarly when an issue plausibly touches more than one area — this does not by\n"
        "itself mean the classifier is unsure. Weigh these scores together with the issue text and the similar\n"
        "issues below, the same way you always would.\n"
        "Top-3 predictions:\n"
        "  1. settings-sync (confidence: 0.579)\n"
        "  2. extensions (confidence: 0.531)\n"
        "  3. profiles (confidence: 0.492)\n"
        + _FOURTH_TAIL
    )
    return _FROZEN_THREE + [
        {"role": "user", "content": user},
        {"role": "assistant", "content": _FOURTH_ASSISTANT_TURN},
    ]


def few_shots_candidate1_no_numbers() -> list[dict]:
    user = (
        _FOURTH_USER_HEADER
        + "\n--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---\n"
        "Ranked component predictions (most to least likely), from an automated classifier. Rank order is\n"
        "the classifier's signal here — treat #1 as its lead assessment.\n"
        "  1. settings-sync\n"
        "  2. extensions\n"
        "  3. profiles\n"
        + _FOURTH_TAIL
    )
    return _FROZEN_THREE + [
        {"role": "user", "content": user},
        {"role": "assistant", "content": _FOURTH_ASSISTANT_TURN},
    ]


def few_shots_candidate2_top1_only() -> list[dict]:
    user = (
        _FOURTH_USER_HEADER
        + "\n--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---\n"
        "Predicted component (automated classifier): settings-sync (confidence: 0.579)\n"
        + _FOURTH_TAIL
    )
    return _FROZEN_THREE + [
        {"role": "user", "content": user},
        {"role": "assistant", "content": _FOURTH_ASSISTANT_TURN},
    ]


VARIANTS = {
    "baseline_v3": (SYSTEM_PROMPT_BASELINE_V3, few_shots_baseline_v3(), build_prompt_baseline_v3),
    "candidate1_no_numbers": (
        SYSTEM_PROMPT_CANDIDATE1_NO_NUMBERS,
        few_shots_candidate1_no_numbers(),
        build_prompt_candidate1_no_numbers,
    ),
    "candidate2_top1_only": (
        SYSTEM_PROMPT_CANDIDATE2_TOP1_ONLY,
        few_shots_candidate2_top1_only(),
        build_prompt_candidate2_top1_only,
    ),
}


def _ollama_synthesize(messages: list[dict]) -> str:
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


HEDGE_PHRASES = [
    "imprecise", "overly optimistic", "uncertain", "unclear", "may not be accurate",
    "hard to say", "difficult to determine", "not entirely clear", "somewhat unclear",
    "lacks confidence", "low confidence", "not fully confident", "hedge",
]


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
    predictor = ResolutionTimePredictor.load(str(models_dir / "resolution_predictor_kubernetes_kubernetes.pkl"))
    train_df = pd.read_parquet(processed_dir / "kubernetes_kubernetes_temporal_train.parquet")
    frozen_retrievers = build_frozen_retrievers(EVAL_SET)
    detector = frozen_retrievers["kubernetes/kubernetes"]

    judge = TriageJudge(model=JUDGE_MODEL, provider="ollama", temperature=0.0, ollama_seed=OLLAMA_SEED, cache=None)
    print("Ollama judge warm-up call (absorbing cold-start variance, ADR-0019)...")
    judge._ollama_completion([{"role": "user", "content": "Reply with just: OK"}])

    all_results: dict[str, dict] = {}

    for variant_name, (system_prompt, few_shots, prompt_builder) in VARIANTS.items():
        print(f"\n{'='*100}\nVARIANT: {variant_name}\n{'='*100}")
        judge_totals, resolution_scores, next_steps_scores, component_scores = [], [], [], []
        hedge_hits = []
        grounding_ok = []
        rows = []

        for iid in TARGET_ISSUE_IDS:
            issue = issues_by_id[iid]
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

            raw = _ollama_synthesize(messages)
            try:
                plan = TriageAssistant._parse_plan(raw)
            except Exception as e:
                print(f"  {iid}: PARSE FAILED: {e}")
                rows.append((iid, "<parse failed>", None, None))
                continue

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
            judge_totals.append(judge_score.total())
            resolution_scores.append(judge_score.resolution_estimate_reasonableness)
            next_steps_scores.append(judge_score.next_steps_actionability)
            component_scores.append(judge_score.component_match)
            grounding_ok.append(report.component_grounded)

            rationale_lower = (judge_score.judge_rationale or "").lower()
            hits = [p for p in HEDGE_PHRASES if p in rationale_lower]
            hedge_hits.append(hits)

            gold_match = "GOLD" if plan.predicted_component == issue["gold_component"] else ""
            print(
                f"  {iid:12} pred={plan.predicted_component:20} clf_top1={classifier_top3[0]['label']:15} "
                f"[{gold_match}] grounded={report.component_grounded} "
                f"judge={judge_score.total()}/15 res={judge_score.resolution_estimate_reasonableness} "
                f"next={judge_score.next_steps_actionability} comp={judge_score.component_match} "
                f"hedge={hits if hits else '-'}"
            )
            rows.append((iid, plan.predicted_component, judge_score.total(), report.component_grounded))

        n_hedge_issues = sum(1 for h in hedge_hits if h)
        summary = {
            "n": len(judge_totals),
            "mean_total": float(np.mean(judge_totals)) if judge_totals else None,
            "mean_resolution_estimate_reasonableness": float(np.mean(resolution_scores)) if resolution_scores else None,
            "mean_next_steps_actionability": float(np.mean(next_steps_scores)) if next_steps_scores else None,
            "mean_component_match": float(np.mean(component_scores)) if component_scores else None,
            "grounding_pass_rate": float(np.mean(grounding_ok)) if grounding_ok else None,
            "n_issues_with_hedge_language": n_hedge_issues,
            "hedge_details": [(iid, h) for iid, h in zip(TARGET_ISSUE_IDS, hedge_hits) if h],
        }
        all_results[variant_name] = summary
        print(f"\n--- {variant_name} SUMMARY ---")
        for k, v in summary.items():
            if k != "hedge_details":
                print(f"  {k}: {v}")
        if summary["hedge_details"]:
            print("  hedge_details:")
            for iid, h in summary["hedge_details"]:
                print(f"    {iid}: {h}")

    print(f"\n{'='*100}\nCROSS-VARIANT COMPARISON\n{'='*100}")
    header = f"{'metric':40}" + "".join(f"{name:24}" for name in VARIANTS)
    print(header)
    for key in [
        "mean_total", "mean_resolution_estimate_reasonableness", "mean_next_steps_actionability",
        "mean_component_match", "grounding_pass_rate", "n_issues_with_hedge_language",
    ]:
        row = f"{key:40}" + "".join(f"{all_results[name][key]!s:24}" for name in VARIANTS)
        print(row)

    out_path = ROOT / "reports" / "probe_confidence_structural_variants_result.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nFull results written to {out_path}")
    print(
        "\nReminder: this is a LOCAL structural signal on a different model than production "
        "(llama3.1:8b vs Groq llama-3.1-8b-instant). A promising result here is grounds to "
        "recommend spending the real Groq confirmation recording; it does not replace it."
    )


if __name__ == "__main__":
    main()
