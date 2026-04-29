"""LLM-as-judge evaluation for System 4 triage plans.

Judge model: llama-3.1-70b-versatile (Groq) or any sufficiently capable model.
Rubric applied per triage plan; results averaged across the gold set.
"""

import json
import logging
import os
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rubric schema
# ---------------------------------------------------------------------------

RUBRIC_DESCRIPTION = """
Evaluate the triage plan against the gold standard on 6 dimensions.
Return ONLY valid JSON with integer scores.

Dimensions:
1. component_match (0–2)
   0 = wrong component
   1 = plausible but not the gold label
   2 = correct component label

2. similar_issues_relevance (0–3)
   0 = similar issues are unrelated or hallucinated numbers
   1 = some related issues retrieved but mostly noise
   2 = mostly relevant, at least 2 on-topic issues
   3 = highly relevant; retrieval surface clearly related prior art

3. resolution_estimate_reasonableness (0–3)
   0 = estimate wildly off (>10× from gold)
   1 = order-of-magnitude correct but imprecise
   2 = estimate within 2× of gold actual resolution time
   3 = estimate within same bucket (hours/days/weeks/months) AND interval contains actual

4. priority_alignment (0–1)
   0 = priority clearly wrong given issue severity and gold context
   1 = priority matches or is defensibly close to gold priority

5. next_steps_actionability (0–3)
   0 = next steps are vague or generic boilerplate
   1 = some actionable steps but missing key follow-ups
   2 = concrete steps that a triager could act on immediately
   3 = precise, ordered, and repo-appropriate — addresses root cause, assigns, and verifies

6. overall_quality (0–3)
   0 = triage plan is unhelpful or misleading
   1 = marginally useful; significant gaps
   2 = solid plan a triager would act on with minor edits
   3 = excellent; could ship directly to triage queue

Respond ONLY with JSON:
{"component_match": int, "similar_issues_relevance": int, "resolution_estimate_reasonableness": int,
 "priority_alignment": int, "next_steps_actionability": int, "overall_quality": int,
 "judge_rationale": "string — 1–2 sentences explaining the overall_quality score"}
"""

DIMENSION_MAX = {
    "component_match": 2,
    "similar_issues_relevance": 3,
    "resolution_estimate_reasonableness": 3,
    "priority_alignment": 1,
    "next_steps_actionability": 3,
    "overall_quality": 3,
}
MAX_TOTAL = sum(DIMENSION_MAX.values())  # 15


class JudgeScore(BaseModel):
    component_match: int = Field(ge=0, le=2)
    similar_issues_relevance: int = Field(ge=0, le=3)
    resolution_estimate_reasonableness: int = Field(ge=0, le=3)
    priority_alignment: int = Field(ge=0, le=1)
    next_steps_actionability: int = Field(ge=0, le=3)
    overall_quality: int = Field(ge=0, le=3)
    judge_rationale: str

    def total(self) -> int:
        return (
            self.component_match
            + self.similar_issues_relevance
            + self.resolution_estimate_reasonableness
            + self.priority_alignment
            + self.next_steps_actionability
            + self.overall_quality
        )

    def normalized(self) -> float:
        return self.total() / MAX_TOTAL


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def majority_component_baseline(
    issue: pd.Series, train_df: pd.DataFrame
) -> dict:
    """Always predict the most common component in training data."""
    top = train_df["component"].value_counts().idxmax()
    return {"predicted_component": top, "triage_summary": f"Majority baseline: {top}"}


def tfidf_only_baseline(issue: pd.Series, classifier) -> dict:
    """Predict only from TF-IDF classifier, no LLM enrichment."""
    import pandas as pd
    title = str(issue.get("title", ""))
    body = str(issue.get("body_clean", ""))
    text = f"{title}. {body}"
    pred = classifier.predict(pd.Series([text]))[0]
    proba = classifier.predict_proba(pd.Series([text]))[0]
    classes = classifier.classes_()
    conf = float(max(proba))
    return {
        "predicted_component": pred,
        "component_confidence": conf,
        "triage_summary": f"TF-IDF only: predicted {pred} with confidence {conf:.2f}",
    }


def retrieval_only_baseline(issue: pd.Series, detector, k: int = 5) -> dict:
    """Predict component from majority vote of top-k similar issues."""
    title = str(issue.get("title", ""))
    body = str(issue.get("body_clean", ""))
    text = f"{title}. {body}"
    num = int(issue.get("number", -1))
    hits = detector.retrieve(text, k=k, exclude_number=num if num > 0 else None)
    return {
        "similar_issues": hits,
        "triage_summary": f"Retrieval only: top-{k} similar issues retrieved",
    }


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class TriageJudge:
    """LLM-as-judge evaluator.

    Uses a stronger model (llama-3.1-70b-versatile or similar) to score
    each triage plan against the gold standard rubric.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise EnvironmentError("GROQ_API_KEY not set.")
        self._groq_key = key

    def score(
        self,
        issue_title: str,
        issue_body: str,
        triage_plan_json: str,
        gold: dict,
    ) -> JudgeScore:
        """Score a single triage plan.

        Args:
            issue_title: Original issue title.
            issue_body: Cleaned body (will be truncated).
            triage_plan_json: JSON string of the TriagePlan.
            gold: Gold standard dict with keys:
                component, priority, actual_resolution_days.
        """
        prompt = self._build_judge_prompt(issue_title, issue_body, triage_plan_json, gold)
        messages = [
            {"role": "system", "content": RUBRIC_DESCRIPTION},
            {"role": "user", "content": prompt},
        ]
        raw = self._groq_completion(messages)
        return self._parse_score(raw)

    def score_batch(
        self,
        records: list[dict],
        delay: float = 1.0,
    ) -> list[tuple[int, Optional[JudgeScore], Optional[str]]]:
        """Score a batch of triage plans.

        Each record dict: issue_number, issue_title, issue_body,
        triage_plan_json, gold (component, priority, actual_resolution_days).
        """
        results = []
        for i, rec in enumerate(records):
            if i > 0:
                time.sleep(delay)
            try:
                score = self.score(
                    rec["issue_title"],
                    rec["issue_body"],
                    rec["triage_plan_json"],
                    rec["gold"],
                )
                results.append((rec["issue_number"], score, None))
            except Exception as exc:
                logger.warning("Judge failed for #%s: %s", rec.get("issue_number"), exc)
                results.append((rec.get("issue_number", -1), None, str(exc)))
        return results

    def reliability_check(
        self,
        records: list[dict],
        sample_size: int = 10,
        delay: float = 1.5,
    ) -> dict:
        """Double-run judge on a sample; report score consistency with Cohen's kappa."""
        sample = records[:sample_size]
        run1 = [r for (_, r, _) in self.score_batch(sample, delay=delay) if r is not None]
        time.sleep(3.0)
        run2 = [r for (_, r, _) in self.score_batch(sample, delay=delay) if r is not None]

        if not run1 or not run2:
            return {"error": "No scores returned"}

        n = min(len(run1), len(run2))
        run1, run2 = run1[:n], run2[:n]

        dims = list(DIMENSION_MAX.keys())
        diffs = {d: [] for d in dims}
        pct_agree = {}
        kappa = {}

        for d in dims:
            v1 = [getattr(s, d) for s in run1]
            v2 = [getattr(s, d) for s in run2]
            diffs[d] = [abs(a - b) for a, b in zip(v1, v2)]
            pct_agree[d] = float(np.mean([a == b for a, b in zip(v1, v2)]))
            kappa[d] = float(_cohens_kappa(v1, v2, max_val=DIMENSION_MAX[d]))

        return {
            "sample_size": n,
            "mean_abs_diff_per_dim": {d: float(np.mean(v)) for d, v in diffs.items()},
            "pct_agreement_per_dim": pct_agree,
            "cohens_kappa_per_dim": kappa,
            "exact_agreement_rate": float(
                np.mean([
                    int(all(getattr(s1, d) == getattr(s2, d) for d in dims))
                    for s1, s2 in zip(run1, run2)
                ])
            ),
            "low_reliability_dims": [d for d, k in kappa.items() if k < 0.4],
        }


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_judge_prompt(
        self, title: str, body: str, plan_json: str, gold: dict
    ) -> str:
        body_preview = body[:600].strip()
        gold_component = gold.get("component", "unknown")
        gold_priority = gold.get("priority", "unknown")
        gold_days = gold.get("actual_resolution_days", "unknown")

        return f"""\
ISSUE:
Title: {title}
Body: {body_preview}

GOLD STANDARD:
  component: {gold_component}
  priority: {gold_priority}
  actual_resolution_days: {gold_days}

TRIAGE PLAN TO EVALUATE:
{plan_json}

Score this triage plan using the rubric. Return ONLY valid JSON.
"""

    def _groq_completion(self, messages: list[dict]) -> str:
        try:
            from groq import Groq
            from groq import RateLimitError, APIStatusError
        except ImportError as e:
            raise ImportError("pip install groq") from e

        client = Groq(api_key=self._groq_key)
        backoff = 6.0
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=512,
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError as e:
                # Fail fast on daily token quota (TPD) — retrying won't help within hours.
                err_str = str(e)
                if "tokens per day" in err_str or '"type": "tokens"' in err_str or "TPD" in err_str:
                    raise
                if attempt == 5:
                    raise
                jitter = backoff * (0.5 + 0.5 * (attempt / 5))
                logger.warning("Judge rate limit — sleeping %.1fs (attempt %d/6)", jitter, attempt + 1)
                time.sleep(jitter)
                backoff = min(backoff * 2, 90.0)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < 5:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 90.0)
                else:
                    raise
        raise RuntimeError("Judge completion failed after 6 attempts")

    @staticmethod
    def _parse_score(raw: str) -> JudgeScore:
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON in judge response: {raw[:200]}")
        data = json.loads(match.group(0))
        return JudgeScore.model_validate(data)


def _cohens_kappa(y1: list[int], y2: list[int], max_val: int) -> float:
    """Cohen's kappa for two ordinal rating sequences."""
    n = len(y1)
    if n == 0:
        return 0.0
    po = sum(a == b for a, b in zip(y1, y2)) / n
    cats = list(range(max_val + 1))
    p1 = [y1.count(c) / n for c in cats]
    p2 = [y2.count(c) / n for c in cats]
    pe = sum(p1[i] * p2[i] for i in range(len(cats)))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def aggregate_scores(scores: list[JudgeScore]) -> dict:
    """Compute mean, std, and per-dimension breakdown across a set of scores."""
    dims = list(DIMENSION_MAX.keys())
    result = {}
    for d in dims:
        vals = [getattr(s, d) for s in scores]
        result[d] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "max_possible": DIMENSION_MAX[d],
            "mean_pct": float(np.mean(vals) / DIMENSION_MAX[d]),
        }
    totals = [s.total() for s in scores]
    result["total"] = {
        "mean": float(np.mean(totals)),
        "std": float(np.std(totals)),
        "max_possible": MAX_TOTAL,
        "mean_pct": float(np.mean(totals) / MAX_TOTAL),
    }
    return result


def compare_baselines(
    judge_scores: list[JudgeScore],
    majority_scores: list[JudgeScore],
    tfidf_scores: list[JudgeScore],
) -> pd.DataFrame:
    """Return a comparison DataFrame of three systems."""
    rows = []
    for name, scores in [
        ("LLM Triage Assistant", judge_scores),
        ("TF-IDF Only", tfidf_scores),
        ("Majority Component", majority_scores),
    ]:
        agg = aggregate_scores(scores)
        row = {"system": name, "total_mean": agg["total"]["mean"], "total_pct": agg["total"]["mean_pct"]}
        for d in DIMENSION_MAX:
            row[d] = agg[d]["mean"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("system")
