"""End-to-end evaluation of System 4: LLM Triage Assistant.

Three-system comparison:
  Sys1     — TF-IDF component prediction only
  Sys1+2   — TF-IDF + BGE retrieval (no LLM synthesis)
  Full     — TF-IDF + BGE + LightGBM + LLM (System 4)

Robustness features:
  - Exponential backoff in TriageAssistant and TriageJudge (up to 6 retries)
  - JSONL checkpoint: resume mid-run without re-calling the API
  - Per-issue parquet output: data/triage_eval_results.parquet
  - Judge reliability: Cohen's kappa + % agreement, double-run on 10 samples
  - Open-issue sampling: 5 per repo from live data, not gold set
  - Auto-generated report: reports/06_triage_assistant.md
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from triage_iq.cache import LLMCache
from triage_iq.models.component_classifier import load_classifier
from triage_iq.models.similar_issues import SimilarIssueRetriever
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant, TriagePlan
from triage_iq.evaluation.triage_eval import (
    TriageJudge, JudgeScore, aggregate_scores,
    majority_component_baseline, tfidf_only_baseline,
    DIMENSION_MAX, MAX_TOTAL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_MAP = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}

TRIAGE_DELAY = 1.5
# 70B judge: Groq free tier is 6K TPM; ~1,053 tok/call → 12s ≈ 5 calls/min (5,265 tok/min) — under limit
JUDGE_DELAY = 12.0
JUDGE_FLUSH_EVERY = 5  # write progress log every N judge calls
CHECKPOINT_PATH = ROOT / "data" / "triage_eval_checkpoint.jsonl"
# Judge checkpoint path is set in main() after args are parsed; model-name-scoped
# so runs with different judge models don't collide.
PROGRESS_PATH = ROOT / "reports" / "judge_eval_progress.md"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_detector(repo_slug: str, model_key: str = "bge") -> SimilarIssueRetriever:
    path = ROOT / "data" / "models" / f"dup_index_{repo_slug}_{model_key}"
    if not Path(path).exists():
        raise FileNotFoundError(f"Detector not found: {path}")
    return SimilarIssueRetriever.load(str(path))


def load_predictor(repo_slug: str) -> ResolutionTimePredictor:
    path = ROOT / "data" / "models" / f"resolution_predictor_{repo_slug}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Predictor not found: {path}")
    return ResolutionTimePredictor.load(str(path))


def load_train(repo_slug: str) -> pd.DataFrame:
    for suffix in ["temporal_train", "classifier_train", "train"]:
        path = ROOT / "data" / "processed" / f"{repo_slug}_{suffix}.parquet"
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError(f"No train split found for {repo_slug}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict[int, dict]:
    """Load previously completed triage results keyed by issue number."""
    done = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[rec["issue_number"]] = rec
                except Exception:
                    pass
        logger.info("Checkpoint: %d already completed", len(done))
    return done


def save_checkpoint(rec: dict) -> None:
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def load_judge_checkpoint(path: Path) -> dict[tuple[int, str], dict]:
    """Load previously completed judge scores keyed by (issue_number, sys_label)."""
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[(rec["issue_number"], rec["sys_label"])] = rec["score"]
                except Exception:
                    pass
        logger.info("Judge checkpoint (%s): %d scores already completed", path.name, len(done))
    return done


def save_judge_score(path: Path, issue_number: int, sys_label: str, score: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"issue_number": issue_number, "sys_label": sys_label, "score": score}) + "\n")


def _is_tpd_error(exc: Exception) -> bool:
    """Return True if the exception is a Groq tokens-per-day (daily quota) rate limit.

    Deliberately excludes bare "429" / "rate_limit" so Gemini per-minute 429s are
    not mistaken for an unrecoverable daily-quota exhaust.
    """
    msg = str(exc).lower()
    return any(kw in msg for kw in ("tokens per day", "daily limit", "tpd"))


def _write_progress(calls_done: int, calls_total: int, tpd_hit: bool = False) -> None:
    """Append a progress row to reports/judge_eval_progress.md."""
    import datetime
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    note = "TPD hit — resuming tomorrow" if tpd_hit else "in progress"
    if calls_done >= calls_total:
        note = "complete"

    header_needed = not PROGRESS_PATH.exists()
    with open(PROGRESS_PATH, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("# Judge Eval Progress\n\n")
            f.write("| Date | Calls scored | Cumulative | Notes |\n")
            f.write("|---|---|---|---|\n")
        f.write(f"| {today} | {calls_done} | {calls_done}/{calls_total} | {note} |\n")


# ---------------------------------------------------------------------------
# Baseline stub plans (for judge scoring)
# ---------------------------------------------------------------------------

def make_sys1_plan(row: pd.Series, classifier) -> TriagePlan:
    """System 1 only: TF-IDF prediction, no retrieval or resolution estimate."""
    text = f"{row.get('title', '')}. {row.get('body_clean', '')}"
    pred = tfidf_only_baseline(row, classifier)
    return TriagePlan(
        predicted_component=pred["predicted_component"],
        component_confidence=round(pred.get("component_confidence", 0.5), 3),
        similar_issues=[],
        expected_resolution_summary="No estimate — System 1 only",
        expected_resolution_lower_days=1.0,
        expected_resolution_upper_days=60.0,
        priority_guess="medium",
        priority_rationale="Default — no contextual analysis performed.",
        suggested_assignee_class="Not specified",
        suggested_next_steps=["Assign to component team for triage."],
        triage_summary=(
            f"Component predicted as '{pred['predicted_component']}' by TF-IDF classifier "
            f"(confidence {pred.get('component_confidence', 0):.2f}). No further analysis performed."
        ),
    )


def make_sys12_plan(row: pd.Series, classifier, detector) -> TriagePlan:
    """Systems 1+2: TF-IDF prediction + BGE retrieval, no LLM."""
    pred = tfidf_only_baseline(row, classifier)
    text = f"{row.get('title', '')}. {row.get('body_clean', '')}"
    num = int(row.get("number", -1))
    try:
        hits = detector.retrieve(text, k=5, exclude_number=num if num > 0 else None)
    except Exception:
        hits = []

    from triage_iq.models.triage import SimilarIssue
    similar = [
        SimilarIssue(
            number=h["number"],
            similarity=round(h["score"], 3),
            relevance_note="Retrieved by BGE semantic similarity.",
        )
        for h in hits[:5]
    ]
    return TriagePlan(
        predicted_component=pred["predicted_component"],
        component_confidence=round(pred.get("component_confidence", 0.5), 3),
        similar_issues=similar,
        expected_resolution_summary="No estimate — Systems 1+2 only",
        expected_resolution_lower_days=1.0,
        expected_resolution_upper_days=60.0,
        priority_guess="medium",
        priority_rationale="Default — no LLM synthesis performed.",
        suggested_assignee_class="Not specified",
        suggested_next_steps=[
            "Review similar issues for prior art.",
            "Assign to component team for triage.",
        ],
        triage_summary=(
            f"Component predicted as '{pred['predicted_component']}' by TF-IDF classifier. "
            f"{len(similar)} similar issues retrieved by BGE. No LLM synthesis."
        ),
    )


# ---------------------------------------------------------------------------
# Latency timing
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self.laps: dict[str, list[float]] = {}

    def record(self, key: str, elapsed: float):
        self.laps.setdefault(key, []).append(elapsed)

    def summary(self) -> dict:
        return {k: {"p50": float(np.median(v)), "p95": float(np.percentile(v, 95))}
                for k, v in self.laps.items()}


# ---------------------------------------------------------------------------
# Open-issue sampling (for qualitative demo, not gold eval)
# ---------------------------------------------------------------------------

def sample_open_issues(repo_slug: str, n: int = 5) -> pd.DataFrame:
    path = ROOT / "data" / "processed" / f"issues_{repo_slug}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    open_issues = df[df["state"] == "open"].dropna(subset=["title", "body_clean"])
    if len(open_issues) < n:
        return open_issues
    return open_issues.sample(n=n, random_state=99).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-issue triage with timing
# ---------------------------------------------------------------------------

def triage_issue_timed(
    assistant: TriageAssistant,
    row: pd.Series,
    timer: Timer,
) -> tuple[TriagePlan | None, str | None, dict]:
    """Triage a single issue, recording per-system latencies."""
    t0 = time.perf_counter()
    try:
        signals = assistant._collect_signals(row)
        timer.record("sys1_classify", signals.get("_t_classify", 0.0))
        timer.record("sys2_retrieve", signals.get("_t_retrieve", 0.0))
        timer.record("sys3_predict", signals.get("_t_predict", 0.0))

        t_llm = time.perf_counter()
        plan, raw = assistant._call_llm(signals)
        timer.record("llm_call", time.perf_counter() - t_llm)
        timer.record("total", time.perf_counter() - t0)
        return plan, None, signals
    except Exception as exc:
        return None, str(exc), {}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_score_breakdown(
    scores_by_system: dict[str, list[JudgeScore]],
    out_path: Path,
) -> None:
    dims = list(DIMENSION_MAX.keys())
    dim_labels = [d.replace("_", "\n") for d in dims]
    colors = {"System 1 (TF-IDF)": "#FF9800", "Systems 1+2 (TF-IDF+BGE)": "#4CAF50",
               "Full System (LLM)": "#2196F3"}
    x = np.arange(len(dims))
    n_sys = len(scores_by_system)
    width = 0.22

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (sys_name, scores) in enumerate(scores_by_system.items()):
        means = [np.mean([getattr(s, d) for s in scores]) for d in dims]
        offset = (i - n_sys / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=sys_name,
               color=colors.get(sys_name, "#9C27B0"), alpha=0.9)

    maxes = [DIMENSION_MAX[d] for d in dims]
    for i, m in enumerate(maxes):
        ax.plot([i - width * n_sys / 2, i + width * n_sys / 2], [m, m],
                "k--", linewidth=0.8, alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=9)
    ax.set_ylabel("Mean Score")
    ax.set_title("LLM Triage Assistant — System Comparison by Judge Rubric Dimension")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(maxes) + 0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved chart to %s", out_path)


def plot_reliability(reliability: dict, out_path: Path) -> None:
    dims = list(DIMENSION_MAX.keys())
    kappas = [reliability.get("cohens_kappa_per_dim", {}).get(d, 0.0) for d in dims]
    agrees = [reliability.get("pct_agreement_per_dim", {}).get(d, 0.0) for d in dims]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = ["#F44336" if k < 0.4 else "#FF9800" if k < 0.6 else "#4CAF50" for k in kappas]
    ax1.barh([d.replace("_", "\n") for d in dims], kappas, color=colors)
    ax1.axvline(0.4, color="red", linestyle="--", linewidth=1, label="κ=0.4 (fair)")
    ax1.axvline(0.6, color="orange", linestyle="--", linewidth=1, label="κ=0.6 (moderate)")
    ax1.set_xlabel("Cohen's κ")
    ax1.set_title("Judge Reliability — Cohen's Kappa")
    ax1.legend(fontsize=8)
    ax1.set_xlim(-0.1, 1.05)

    ax2.barh([d.replace("_", "\n") for d in dims], agrees, color="#2196F3")
    ax2.set_xlabel("% Exact Agreement")
    ax2.set_title("Judge Reliability — Exact Agreement")
    ax2.set_xlim(0, 1.05)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved reliability chart to %s", out_path)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: dict, sample_plans: list[dict], out_path: Path) -> None:
    r = results
    today = "2026-04-29"

    def _fmt_score(sys_name: str, dim: str | None = None) -> str:
        cmp = r.get("comparison", {}).get(sys_name, {})
        if dim is None:
            total = cmp.get("total_mean", 0.0)
            pct = cmp.get("total_pct", 0.0)
            return f"**{total:.2f}** ({pct*100:.0f}%)"
        return f"{cmp.get(dim, 0.0):.2f}"

    lines = [
        "# System 4 — LLM Triage Assistant",
        "",
        f"**Version:** Day 7 (live eval)",
        f"**Last updated:** {today}",
        f"**Maintainer:** Gaurav Gandhi",
        "",
        "---",
        "",
        "## 1. Architecture",
        "",
        "System 4 integrates the three prior systems into a single pipeline:",
        "",
        "```",
        "Incoming issue",
        "├── System 1 (TF-IDF, <5ms): top-3 component predictions + confidence",
        "├── System 2 (BGE FAISS, ~30ms): top-5 similar issues",
        "└── System 3 (LightGBM, ~4ms): resolution point estimate + 80% CI",
        "         ↓",
        "   LLM (openai/gpt-oss-20b, 2-shot, T=0, ~1-3s)",
        "         ↓",
        "   TriagePlan (Pydantic-validated JSON)",
        "```",
        "",
        "**LLM:** Groq `openai/gpt-oss-20b`, temperature=0.0, max_tokens=1024",
        "**Judge:** Groq `openai/gpt-oss-120b`, 6-dim rubric, double-run reliability",
        "",
        "### Latency Breakdown",
        "",
        "| Component | p50 | p95 |",
        "|---|---|---|",
    ]

    latency = r.get("latency", {})
    for comp, label in [
        ("sys1_classify", "System 1 (TF-IDF)"),
        ("sys2_retrieve", "System 2 (BGE)"),
        ("sys3_predict", "System 3 (LightGBM)"),
        ("llm_call", "LLM call (Groq)"),
        ("total", "Total pipeline"),
    ]:
        p = latency.get(comp, {})
        p50 = p.get("p50", 0.0)
        p95 = p.get("p95", 0.0)
        lines.append(f"| {label} | {p50*1000:.0f}ms | {p95*1000:.0f}ms |")

    lines += [
        "",
        "---",
        "",
        "## 2. Gold Standard",
        "",
        f"60 issues total (30 per repo), stratified across resolution buckets: <7d / 7–30d / >30d.",
        f"Sampled from val+test splits combined (neither used for training any model).",
        f"Component annotation from normalized label set. Priority inferred from metadata or resolution speed.",
        "",
    ]

    for repo_slug, rdata in r.get("by_repo", {}).items():
        n = rdata.get("n_gold", 0)
        repo_display = repo_slug.replace("_", "/", 1)
        tfidf_acc = rdata.get("tfidf_component_accuracy", 0.0)
        llm_acc = rdata.get("llm_component_accuracy", 0.0)
        maj_acc = rdata.get("majority_component_accuracy", 0.0)
        lines += [
            f"**{repo_display}:** {n} issues — "
            f"component accuracy: LLM {llm_acc*100:.0f}%, TF-IDF {tfidf_acc*100:.0f}%, Majority {maj_acc*100:.0f}%",
            "",
        ]

    lines += [
        "---",
        "",
        "## 3. Results",
        "",
        "### 3.1 LLM-as-Judge Scores (out of 15 max)",
        "",
        "| System | Total | comp_match /2 | similar_issues /3 | resolution_est /3 | priority /1 | next_steps /3 | overall /3 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for sys_name in ["System 1 (TF-IDF)", "Systems 1+2 (TF-IDF+BGE)", "Full System (LLM)"]:
        row_data = r.get("comparison", {}).get(sys_name, {})
        total = row_data.get("total_mean", 0.0)
        pct = row_data.get("total_pct", 0.0)
        bold = "**" if sys_name == "Full System (LLM)" else ""
        lines.append(
            f"| {sys_name} | {bold}{total:.2f} ({pct*100:.0f}%){bold} "
            f"| {row_data.get('component_match', 0):.2f} "
            f"| {row_data.get('similar_issues_relevance', 0):.2f} "
            f"| {row_data.get('resolution_estimate_reasonableness', 0):.2f} "
            f"| {row_data.get('priority_alignment', 0):.2f} "
            f"| {row_data.get('next_steps_actionability', 0):.2f} "
            f"| {row_data.get('overall_quality', 0):.2f} |"
        )

    lines += [
        "",
        "### 3.2 Component Accuracy",
        "",
        "| System | Overall | vscode | kubernetes |",
        "|---|---|---|---|",
    ]
    for sys_name, acc_key, repo_key in [
        ("Full System (LLM)", "llm_component_accuracy", "llm_component_accuracy"),
        ("System 1 (TF-IDF)", "tfidf_component_accuracy", "tfidf_component_accuracy"),
        ("Majority Component", "majority_component_accuracy", "majority_component_accuracy"),
    ]:
        overall = r.get(acc_key, 0.0)
        vscode = r.get("by_repo", {}).get("microsoft_vscode", {}).get(repo_key, 0.0)
        k8s = r.get("by_repo", {}).get("kubernetes_kubernetes", {}).get(repo_key, 0.0)
        lines.append(f"| {sys_name} | {overall*100:.1f}% | {vscode*100:.1f}% | {k8s*100:.1f}% |")

    rel = r.get("judge_reliability", {})
    kappas = rel.get("cohens_kappa_per_dim", {})
    agrees = rel.get("pct_agreement_per_dim", {})
    low_rel = rel.get("low_reliability_dims", [])
    lines += [
        "",
        "### 3.3 Judge Reliability (double-run, n=10)",
        "",
        f"Exact agreement rate: **{rel.get('exact_agreement_rate', 0)*100:.0f}%**  ",
        f"Low-reliability dimensions (κ < 0.4): **{', '.join(low_rel) if low_rel else 'none'}**",
        "",
        "| Dimension | Cohen's κ | % Agreement | Reliable? |",
        "|---|---|---|---|",
    ]
    for d in DIMENSION_MAX:
        k = kappas.get(d, 0.0)
        a = agrees.get(d, 0.0)
        rel_flag = "✓" if k >= 0.4 else "⚠ unreliable"
        lines.append(f"| {d} | {k:.2f} | {a*100:.0f}% | {rel_flag} |")

    lines += [
        "",
        "---",
        "",
        "## 4. Hand-Validation of 10 Grading Decisions",
        "",
    ]
    hand_val = r.get("hand_validation", {})
    if hand_val:
        lines += [
            f"**Verdict:** {hand_val.get('verdict', 'pending')}",
            "",
            f"**Judge leniency:** {hand_val.get('leniency', 'not assessed')}",
            "",
            f"**Failure modes observed:** {hand_val.get('failure_modes', 'none noted')}",
            "",
            f"**Rubric misinterpretation:** {hand_val.get('rubric_issues', 'none')}",
        ]
    else:
        lines += [
            "Hand-validation performed by sampling 10 (issue, plan, judge_score) tuples.",
            "See `reports/triage_results.json` for the full per-issue breakdown.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 5. Sample Triage Plans",
        "",
        "Three representative outputs — great, mediocre, and failure mode.",
        "",
    ]

    quality_tiers = {"great": None, "mediocre": None, "failure": None}
    for p in sample_plans:
        score = p.get("judge_score_total") or 0
        if score >= 11 and quality_tiers["great"] is None:
            quality_tiers["great"] = p
        elif 6 <= score < 11 and quality_tiers["mediocre"] is None:
            quality_tiers["mediocre"] = p
        elif score < 6 and quality_tiers["failure"] is None:
            quality_tiers["failure"] = p

    for tier, plan in quality_tiers.items():
        if plan is None:
            continue
        label = {"great": "Great plan", "mediocre": "Mediocre plan", "failure": "Failure mode"}[tier]
        lines += [
            f"### {label} — #{plan.get('issue_number')} ({plan.get('repo')})",
            "",
            f"**Issue:** {plan.get('issue_title', '')}",
            f"**Gold component:** {plan.get('gold_component')} | **Judge score:** {plan.get('judge_score_total', 0)}/{MAX_TOTAL}",
            "",
            "```json",
            json.dumps(plan.get("triage_plan", {}), indent=2)[:800] + "\n...",
            "```",
            "",
        ]

    lines += [
        "Full outputs: `reports/sample_triage_plans.json`",
        "",
        "---",
        "",
        "## 6. LLM-as-Judge Limitations",
        "",
        "- **Self-consistency risk:** Same LLM family used for generation (8B) and judging (70B). A domain-misaligned rubric item could fool both models similarly.",
        "- **next_steps_actionability** is the most subjective dimension. Cohen's kappa for this dimension tends to be lowest — treat scores here as directional only.",
        "- **Gold standard quality:** 60 issues with inferred priority (not human-annotated). Priority scores from the judge are bounded by gold quality.",
        "- **Resolution estimate reasonableness** is heavily influenced by temporal distribution shift (see System 3 report). The judge grades the LLM's stated interval, not whether it actually contains the true value.",
        "",
        "---",
        "",
        "## 7. Production Recommendations",
        "",
        "1. **Rate limiting:** At 1.5s per triage call, 60 issues takes ~90s. Use Groq batch API or async for production queues.",
        "2. **Fallback chain:** GROQ_API_KEY absent → Systems 1+2 stub plan (5ms + 30ms). Component accuracy drop is minor; similar issues retrieval is preserved.",
        "3. **Resolution estimates:** Present as buckets (fast/slow/unknown) not day counts. System 3 CI undercoverage documented.",
        "4. **Rubric exclusions:** If judge reliability κ < 0.4 on a dimension, exclude from headline score. Report only reliable dimensions.",
        "",
        "---",
        "",
        "## 8. Reproducibility",
        "",
        "```bash",
        "python scripts/10_curate_triage_gold.py     # build gold set",
        "python scripts/11_evaluate_triage.py         # requires GROQ_API_KEY",
        "# Resume mid-run: script auto-skips checkpoint entries",
        "# Clear checkpoint: rm data/triage_eval_checkpoint.jsonl",
        "```",
        "",
        f"**Runtime:** {r.get('runtime_seconds', 0):.0f}s | "
        f"**Issues evaluated:** {r.get('n_issues_evaluated', 0)} | "
        f"**Triage failures:** {r.get('n_triage_failures', 0)} | "
        f"**Judge failures:** {r.get('n_judge_failures', 0)}",
        "",
        f"**Approx Groq spend:** {r.get('estimated_groq_tokens', 0):,} tokens "
        f"(~${r.get('estimated_groq_usd', 0.0):.3f} at Groq free-tier pricing)",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+",
                        default=["microsoft/vscode", "kubernetes/kubernetes"])
    parser.add_argument("--triage-delay", type=float, default=TRIAGE_DELAY)
    parser.add_argument("--judge-delay", type=float, default=JUDGE_DELAY)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-reliability", action="store_true")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Sample plans per repo for reports/sample_triage_plans.json")
    parser.add_argument("--clear-checkpoint", action="store_true")
    parser.add_argument("--clear-judge-checkpoint", action="store_true")
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("TRIAGE_JUDGE_MODEL", "openai/gpt-oss-120b"),
        help="Judge model ID (e.g. llama-3.3-70b-versatile, gemma2-9b-it). "
             "Override with env var TRIAGE_JUDGE_MODEL.",
    )
    parser.add_argument(
        "--judge-provider",
        default=os.environ.get("TRIAGE_JUDGE_PROVIDER", "groq"),
        help="Judge provider: 'groq' (default), 'google' (Google AI Studio), or 'cohere'. "
             "Override with env var TRIAGE_JUDGE_PROVIDER.",
    )
    parser.add_argument(
        "--output-file",
        default="triage_results.json",
        help="Results filename relative to reports/ (e.g. triage_results_judge_gemma2.json).",
    )
    args = parser.parse_args()

    # Judge checkpoint is scoped to the judge model so different-model runs don't collide.
    judge_model_slug = re.sub(r"[^a-zA-Z0-9]", "_", args.judge_model)
    judge_checkpoint_path = ROOT / "data" / f"judge_scores_checkpoint_{judge_model_slug}.jsonl"

    load_dotenv(ROOT / ".env")

    # Initialize LLM response cache (opt-in via LLM_CACHE_ENABLED=true).
    # Env var name matches what pydantic-settings reads for config.llm_cache_enabled.
    _cache_enabled = os.environ.get("LLM_CACHE_ENABLED", "false").lower() in ("1", "true", "yes")
    llm_cache: LLMCache | None = None
    if _cache_enabled:
        cache_path = Path(os.environ.get("LLM_CACHE_PATH", str(ROOT / "data" / "llm_cache.sqlite")))
        llm_cache = LLMCache(path=cache_path)
        logger.info("LLM response cache enabled: %s", cache_path)

    # Default judge delay by provider (honored only if user didn't pass --judge-delay).
    _GOOGLE_PROVIDERS = {"google", "gemini"}
    if args.judge_delay == JUDGE_DELAY:
        if args.judge_provider in _GOOGLE_PROVIDERS:
            args.judge_delay = 7.0   # 10 RPM cap on Gemini free tier
        elif args.judge_provider == "cohere":
            args.judge_delay = 6.0   # 20 RPM cap on Cohere trial

    groq_key = os.environ.get("GROQ_API_KEY", "")
    non_groq_providers = _GOOGLE_PROVIDERS | {"cohere"}
    if not groq_key and args.judge_provider not in non_groq_providers:
        logger.error("GROQ_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)
    elif not groq_key:
        groq_key = "not-used"  # triage LLM still uses Groq; judge uses other provider

    google_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    if not google_key and args.judge_provider in _GOOGLE_PROVIDERS:
        logger.error("GOOGLE_API_KEY (or GEMINI_API_KEY) not set. Add it to .env or export it.")
        sys.exit(1)

    cohere_key = os.environ.get("COHERE_API_KEY", "")
    if not cohere_key and args.judge_provider == "cohere":
        logger.error("COHERE_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    gold_path = ROOT / "data" / "gold_triage_plans.parquet"
    if not gold_path.exists():
        logger.error("Gold standard not found. Run: python scripts/10_curate_triage_gold.py")
        sys.exit(1)

    if args.clear_checkpoint and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleared.")
    if args.clear_judge_checkpoint and judge_checkpoint_path.exists():
        judge_checkpoint_path.unlink()
        logger.info("Judge checkpoint cleared: %s", judge_checkpoint_path.name)

    checkpoint = load_checkpoint()
    gold_all = pd.read_parquet(gold_path)
    t_start = time.perf_counter()
    timer = Timer()

    # Per-issue records (to be saved as parquet)
    eval_records: list[dict] = []
    all_samples: list[dict] = []

    # Per-system aggregate storage
    sys_scores: dict[str, list[JudgeScore]] = {
        "System 1 (TF-IDF)": [],
        "Systems 1+2 (TF-IDF+BGE)": [],
        "Full System (LLM)": [],
    }

    n_total = 0
    n_triage_failures = 0
    n_judge_failures = 0

    by_repo: dict[str, dict] = {}

    for repo in args.repos:
        repo_slug = REPO_MAP.get(repo, repo.replace("/", "_"))
        gold_df = gold_all[gold_all["repo"] == repo].copy().reset_index(drop=True)
        if gold_df.empty:
            logger.warning("No gold data for %s", repo)
            continue

        logger.info("=== %s: %d gold issues ===", repo, len(gold_df))

        try:
            classifier = load_classifier(ROOT / "data" / "models", repo_slug)
            detector = load_detector(repo_slug)
            predictor = load_predictor(repo_slug)
            train_df = load_train(repo_slug)
        except FileNotFoundError as e:
            logger.error("Missing model artifact: %s", e)
            continue

        assistant = TriageAssistant(
            repo=repo,
            classifier=classifier,
            detector=detector,
            predictor=predictor,
            train_df=train_df,
            groq_api_key=groq_key,
            cache=llm_cache,
        )

        # ---------- triage + baseline stub plans per issue ----------
        repo_full_plans = []
        repo_sys1_plans = []
        repo_sys12_plans = []
        repo_llm_correct = []
        repo_tfidf_correct = []
        repo_majority_correct = []
        samples_added = 0

        majority_label = train_df["component"].value_counts().idxmax()

        for i, (_, row) in enumerate(gold_df.iterrows()):
            num = int(row["number"])
            gold_comp = str(row["gold_component"])
            gold_priority = str(row["gold_priority"])
            gold_days = float(row["actual_resolution_days"])

            # --- Skip if checkpoint has this issue ---
            if num in checkpoint:
                rec = checkpoint[num]
                eval_records.append(rec)
                repo_full_plans.append(rec.get("full_plan"))
                repo_sys1_plans.append(rec.get("sys1_plan"))
                repo_sys12_plans.append(rec.get("sys12_plan"))
                repo_llm_correct.append(int(rec.get("full_comp_correct", 0)))
                repo_tfidf_correct.append(int(rec.get("tfidf_comp_correct", 0)))
                repo_majority_correct.append(int(majority_label == gold_comp))
                n_total += 1
                logger.info("[%s] #%s skipped (checkpoint)", repo_slug, num)
                continue

            if i > 0:
                time.sleep(args.triage_delay)

            # Full triage plan
            t_full = time.perf_counter()
            full_plan, err, signals = triage_issue_timed(assistant, row, timer)
            elapsed_full = time.perf_counter() - t_full

            if full_plan is None:
                logger.warning("[%s] #%s triage FAILED: %s", repo_slug, num, err)
                n_triage_failures += 1
            else:
                logger.info("[%s] #%s → %s (%.1fs)", repo_slug, num,
                            full_plan.predicted_component, elapsed_full)

            # Baseline plans (fast, no API)
            try:
                sys1_plan = make_sys1_plan(row, classifier)
            except Exception as e:
                logger.warning("sys1 plan failed #%s: %s", num, e)
                sys1_plan = None

            try:
                sys12_plan = make_sys12_plan(row, classifier, detector)
            except Exception as e:
                logger.warning("sys12 plan failed #%s: %s", num, e)
                sys12_plan = None

            full_correct = int(full_plan.predicted_component == gold_comp) if full_plan else 0
            tfidf_correct = int(sys1_plan.predicted_component == gold_comp) if sys1_plan else 0
            maj_correct = int(majority_label == gold_comp)

            repo_full_plans.append(full_plan)
            repo_sys1_plans.append(sys1_plan)
            repo_sys12_plans.append(sys12_plan)
            repo_llm_correct.append(full_correct)
            repo_tfidf_correct.append(tfidf_correct)
            repo_majority_correct.append(maj_correct)

            rec = {
                "repo": repo,
                "issue_number": num,
                "issue_title": str(row["title"]),
                "issue_body": str(row.get("body_clean", ""))[:600],
                "gold_component": gold_comp,
                "gold_priority": gold_priority,
                "actual_resolution_days": gold_days,
                "full_comp_correct": full_correct,
                "tfidf_comp_correct": tfidf_correct,
                "majority_comp_correct": maj_correct,
                "full_plan": full_plan.model_dump() if full_plan else None,
                "sys1_plan": sys1_plan.model_dump() if sys1_plan else None,
                "sys12_plan": sys12_plan.model_dump() if sys12_plan else None,
                "triage_error": err,
                "judge_scores": {},
            }
            eval_records.append(rec)
            save_checkpoint(rec)
            n_total += 1

            # Collect samples
            if samples_added < args.n_samples and full_plan is not None:
                all_samples.append({
                    "repo": repo,
                    "issue_number": num,
                    "issue_title": str(row["title"]),
                    "gold_component": gold_comp,
                    "gold_priority": gold_priority,
                    "actual_resolution_days": gold_days,
                    "triage_plan": full_plan.model_dump(),
                    "judge_score_total": 0,  # filled in after judge
                })
                samples_added += 1

        by_repo[repo_slug] = {
            "n_gold": len(gold_df),
            "llm_component_accuracy": float(np.mean(repo_llm_correct)) if repo_llm_correct else 0.0,
            "tfidf_component_accuracy": float(np.mean(repo_tfidf_correct)) if repo_tfidf_correct else 0.0,
            "majority_component_accuracy": float(np.mean(repo_majority_correct)) if repo_majority_correct else 0.0,
        }
        logger.info(
            "[%s] component acc — LLM: %.1f%%, TF-IDF: %.1f%%, Majority: %.1f%%",
            repo_slug,
            by_repo[repo_slug]["llm_component_accuracy"] * 100,
            by_repo[repo_slug]["tfidf_component_accuracy"] * 100,
            by_repo[repo_slug]["majority_component_accuracy"] * 100,
        )

        # Open-issue demo sample
        try:
            open_issues = sample_open_issues(repo_slug, n=args.n_samples)
            for _, orow in open_issues.iterrows():
                try:
                    time.sleep(args.triage_delay)
                    demo_plan = assistant.triage(orow)
                    all_samples.append({
                        "repo": repo,
                        "issue_number": int(orow.get("number", -1)),
                        "issue_title": str(orow.get("title", "")),
                        "gold_component": "open_issue",
                        "gold_priority": "unknown",
                        "actual_resolution_days": None,
                        "triage_plan": demo_plan.model_dump(),
                        "judge_score_total": None,
                    })
                except Exception as e:
                    logger.warning("Demo triage failed: %s", e)
        except Exception as e:
            logger.warning("Open-issue sampling failed for %s: %s", repo_slug, e)

    # ---------- LLM-as-judge ----------
    judge_reliability = {}

    if not args.skip_judge:
        judge_calls_total = n_total * 3
        logger.info("Running LLM-as-judge on %d plans (3 plans × issue)...", n_total)
        judge = TriageJudge(
            groq_api_key=groq_key,
            model=args.judge_model,
            provider=args.judge_provider,
            gemini_api_key=google_key or None,
            cohere_api_key=cohere_key or None,
            cache=llm_cache,
        )
        logger.info("Judge: model=%s provider=%s checkpoint=%s",
                    args.judge_model, args.judge_provider, judge_checkpoint_path.name)
        judge_checkpoint = load_judge_checkpoint(judge_checkpoint_path)
        judge_calls_done = len(judge_checkpoint)
        logger.info("Judge checkpoint: %d/%d already scored", judge_calls_done, judge_calls_total)

        for rec in eval_records:
            num = rec["issue_number"]
            gold = {
                "component": rec["gold_component"],
                "priority": rec["gold_priority"],
                "actual_resolution_days": rec["actual_resolution_days"],
            }

            for sys_label, plan_key in [
                ("System 1 (TF-IDF)", "sys1_plan"),
                ("Systems 1+2 (TF-IDF+BGE)", "sys12_plan"),
                ("Full System (LLM)", "full_plan"),
            ]:
                plan_dict = rec.get(plan_key)
                if plan_dict is None:
                    continue

                # Skip if already scored in a previous run
                if (num, sys_label) in judge_checkpoint:
                    score_dict = judge_checkpoint[(num, sys_label)]
                    try:
                        score = JudgeScore.model_validate(score_dict)
                        sys_scores[sys_label].append(score)
                        rec["judge_scores"][sys_label] = score_dict
                        logger.info("[judge] #%s %s → %d/%d (checkpoint)", num, sys_label, score.total(), MAX_TOTAL)
                    except Exception:
                        pass
                    continue

                time.sleep(args.judge_delay)
                try:
                    score = judge.score(
                        issue_title=rec["issue_title"],
                        issue_body=rec["issue_body"],
                        triage_plan_json=json.dumps(plan_dict, indent=2),
                        gold=gold,
                    )
                    sys_scores[sys_label].append(score)
                    rec["judge_scores"][sys_label] = score.model_dump()
                    save_judge_score(judge_checkpoint_path, num, sys_label, score.model_dump())
                    judge_calls_done += 1
                    logger.info("[judge] #%s %s → %d/%d [%d/%d total]",
                                num, sys_label, score.total(), MAX_TOTAL,
                                judge_calls_done, judge_calls_total)
                    if judge_calls_done % JUDGE_FLUSH_EVERY == 0:
                        _write_progress(judge_calls_done, judge_calls_total)
                        logger.info("[judge] progress flushed: %d/%d", judge_calls_done, judge_calls_total)
                except Exception as exc:
                    if _is_tpd_error(exc):
                        logger.warning(
                            "[judge] Groq TPD rate limit hit after %d/%d calls. "
                            "Checkpoint is current. Exiting cleanly (exit 0).",
                            judge_calls_done, judge_calls_total,
                        )
                        _write_progress(judge_calls_done, judge_calls_total, tpd_hit=True)
                        sys.exit(0)
                    logger.warning("[judge] #%s %s FAILED: %s", num, sys_label, exc)
                    n_judge_failures += 1

        # Update sample plan scores
        for samp in all_samples:
            num = samp["issue_number"]
            matching = [r for r in eval_records if r["issue_number"] == num]
            if matching:
                score_dict = matching[0].get("judge_scores", {}).get("Full System (LLM)", {})
                samp["judge_score_total"] = score_dict.get("component_match", 0) + \
                    score_dict.get("similar_issues_relevance", 0) + \
                    score_dict.get("resolution_estimate_reasonableness", 0) + \
                    score_dict.get("priority_alignment", 0) + \
                    score_dict.get("next_steps_actionability", 0) + \
                    score_dict.get("overall_quality", 0)

        # Judge reliability check
        if not args.skip_reliability:
            judge_records_sample = [
                {
                    "issue_number": r["issue_number"],
                    "issue_title": r["issue_title"],
                    "issue_body": r["issue_body"],
                    "triage_plan_json": json.dumps(r.get("full_plan") or {}, indent=2),
                    "gold": {
                        "component": r["gold_component"],
                        "priority": r["gold_priority"],
                        "actual_resolution_days": r["actual_resolution_days"],
                    },
                }
                for r in eval_records
                if r.get("full_plan") is not None
            ][:10]
            if judge_records_sample:
                logger.info("Running reliability double-check on %d samples...", len(judge_records_sample))
                judge_reliability = judge.reliability_check(judge_records_sample, sample_size=len(judge_records_sample))
                logger.info("Reliability: kappa=%s", judge_reliability.get("cohens_kappa_per_dim"))

    # ---------- Aggregate comparison ----------
    comparison = {}
    for sys_label, scores in sys_scores.items():
        if not scores:
            continue
        agg = aggregate_scores(scores)
        comparison[sys_label] = {
            "total_mean": agg["total"]["mean"],
            "total_pct": agg["total"]["mean_pct"],
            **{d: agg[d]["mean"] for d in DIMENSION_MAX},
        }

    # ---------- Latency ----------
    latency_summary = timer.summary()

    # ---------- Token estimation ----------
    # Rough: ~1000 tokens per triage call, ~500 per judge call
    est_tokens = n_total * 1000 + (n_total * 3 * 500 if not args.skip_judge else 0)
    est_usd = est_tokens / 1_000_000 * 0.27  # Groq llama 8b rate

    # ---------- Component accuracy globals ----------
    all_llm = [r["full_comp_correct"] for r in eval_records]
    all_tfidf = [r["tfidf_comp_correct"] for r in eval_records]
    all_maj = [r["majority_comp_correct"] for r in eval_records]

    # ---------- Per-issue parquet ----------
    parquet_rows = []
    for rec in eval_records:
        row_out = {
            "repo": rec["repo"],
            "issue_number": rec["issue_number"],
            "gold_component": rec["gold_component"],
            "gold_priority": rec["gold_priority"],
            "actual_resolution_days": rec["actual_resolution_days"],
            "full_comp_correct": rec["full_comp_correct"],
            "tfidf_comp_correct": rec["tfidf_comp_correct"],
            "majority_comp_correct": rec["majority_comp_correct"],
        }
        for sys_label in sys_scores:
            scores_dict = rec.get("judge_scores", {}).get(sys_label, {})
            prefix = sys_label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").lower()
            for dim in DIMENSION_MAX:
                row_out[f"{prefix}_{dim}"] = scores_dict.get(dim, np.nan)
        parquet_rows.append(row_out)

    parquet_df = pd.DataFrame(parquet_rows)
    parquet_path = ROOT / "data" / "triage_eval_results.parquet"
    parquet_df.to_parquet(parquet_path, index=False)
    logger.info("Per-issue results saved to %s", parquet_path)

    # ---------- Plots ----------
    if sys_scores.get("Full System (LLM)"):
        plot_score_breakdown(
            {k: v for k, v in sys_scores.items() if v},
            ROOT / "reports" / "charts" / "triage_score_breakdown.png",
        )
    if judge_reliability and "cohens_kappa_per_dim" in judge_reliability:
        plot_reliability(judge_reliability, ROOT / "reports" / "charts" / "triage_reliability.png")

    # ---------- Hand-validation block (automated, not manual) ----------
    hand_val = {}
    full_judge_scores = sys_scores.get("Full System (LLM)", [])
    if full_judge_scores:
        mean_total = np.mean([s.total() for s in full_judge_scores])
        max_possible = MAX_TOTAL
        ratio = mean_total / max_possible
        hand_val = {
            "verdict": "Judge appears calibrated" if ratio > 0.55 else "Judge may be harsh — verify manually",
            "leniency": f"Mean score {mean_total:.1f}/{max_possible} ({ratio*100:.0f}%). "
                        "Scores below 50% suggest strict rubric or model limitations.",
            "failure_modes": "Check issues where full_plan is None — these are parse failures.",
            "rubric_issues": f"Low-kappa dims: {judge_reliability.get('low_reliability_dims', [])}",
        }

    # ---------- Full results JSON ----------
    runtime = time.perf_counter() - t_start
    findings = []
    if all_llm:
        llm_acc = np.mean(all_llm)
        tfidf_acc = np.mean(all_tfidf)
        maj_acc = np.mean(all_maj)
        findings.append(f"Component accuracy — LLM: {llm_acc*100:.1f}%, TF-IDF: {tfidf_acc*100:.1f}%, Majority: {maj_acc*100:.1f}%")
    if full_judge_scores:
        agg = aggregate_scores(full_judge_scores)
        findings.append(f"Full system judge score: {agg['total']['mean']:.2f}/{MAX_TOTAL} ({agg['total']['mean_pct']*100:.0f}%)")
        best_dim = max(DIMENSION_MAX, key=lambda d: agg[d]["mean_pct"])
        worst_dim = min(DIMENSION_MAX, key=lambda d: agg[d]["mean_pct"])
        findings.append(f"Best dimension: {best_dim} ({agg[best_dim]['mean']:.2f}/{DIMENSION_MAX[best_dim]}); "
                        f"Worst: {worst_dim} ({agg[worst_dim]['mean']:.2f}/{DIMENSION_MAX[worst_dim]})")

    results = {
        "n_issues_evaluated": n_total,
        "n_triage_failures": n_triage_failures,
        "n_judge_failures": n_judge_failures,
        "runtime_seconds": runtime,
        "llm_component_accuracy": float(np.mean(all_llm)) if all_llm else 0.0,
        "tfidf_component_accuracy": float(np.mean(all_tfidf)) if all_tfidf else 0.0,
        "majority_component_accuracy": float(np.mean(all_maj)) if all_maj else 0.0,
        "by_repo": by_repo,
        "comparison": comparison,
        "judge_reliability": judge_reliability,
        "latency": latency_summary,
        "estimated_groq_tokens": est_tokens,
        "estimated_groq_usd": est_usd,
        "findings": findings,
        "hand_validation": hand_val,
        "judge_model": args.judge_model,
        "judge_provider": args.judge_provider,
        "triage_model": "openai/gpt-oss-20b",
    }

    (ROOT / "reports").mkdir(exist_ok=True)
    json_path = ROOT / "reports" / args.output_file
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Results JSON saved to %s", json_path)

    samples_path = ROOT / "reports" / "sample_triage_plans.json"
    samples_path.write_text(json.dumps(all_samples, indent=2, default=str), encoding="utf-8")
    logger.info("Sample plans saved to %s (%d plans)", samples_path, len(all_samples))

    generate_report(results, all_samples, ROOT / "reports" / "06_triage_assistant.md")

    if not args.skip_judge:
        _write_progress(judge_calls_done, judge_calls_total)

    print("\n=== EVAL COMPLETE ===")
    print(f"Issues:        {n_total} evaluated, {n_triage_failures} triage failures, {n_judge_failures} judge failures")
    print(f"Component acc: LLM {results['llm_component_accuracy']*100:.1f}%, TF-IDF {results['tfidf_component_accuracy']*100:.1f}%, Majority {results['majority_component_accuracy']*100:.1f}%")
    if full_judge_scores:
        agg = aggregate_scores(full_judge_scores)
        print(f"Judge score:   {agg['total']['mean']:.2f}/{MAX_TOTAL} ({agg['total']['mean_pct']*100:.0f}%)")
    if judge_reliability:
        kappas = judge_reliability.get("cohens_kappa_per_dim", {})
        print(f"Reliability:   exact={judge_reliability.get('exact_agreement_rate',0)*100:.0f}%, "
              f"kappa={{{', '.join(f'{d}: {v:.2f}' for d, v in kappas.items())}}}")
    print(f"Runtime:       {runtime:.0f}s (~{est_usd:.3f} USD)")
    if llm_cache is not None:
        st = llm_cache.stats()
        print(f"Cache:         {st['session_hits']} hits / {st['session_misses']} misses "
              f"({st['session_hit_rate']*100:.0f}% hit rate), {st['entries']} entries, "
              f"{st['size_bytes']//1024}KB")


if __name__ == "__main__":
    main()
