"""System 4 — LLM Triage Assistant.

Integrates Systems 1–3 to produce a structured TriagePlan for each incoming
GitHub issue. Uses Groq (llama-3.1-8b-instant) with 2-shot examples and
Pydantic-validated JSON output.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from triage_iq.models.grounding import verify_plan_grounding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------


class SimilarIssue(BaseModel):
    number: int
    similarity: float = Field(ge=0.0, le=1.0)
    relevance_note: str


class ConformalIntervalResult(BaseModel):
    """CQR-adjusted prediction interval with empirical coverage metadata.

    lower_days / upper_days are per-request predictions (raw interval ± Q scalar).
    empirical_coverage and CI bounds are fixed per repo from the calibration run.
    This is marginal (not conditional) coverage; temporal data may violate exchangeability.
    See ADR-0010.
    """

    lower_days: float = Field(ge=0.0)
    upper_days: float = Field(ge=0.0)
    target_coverage: float = Field(ge=0.0, le=1.0)
    empirical_coverage: float = Field(ge=0.0, le=1.0)
    coverage_ci95_lower: float = Field(ge=0.0, le=1.0)
    coverage_ci95_upper: float = Field(ge=0.0, le=1.0)


class GroundingAttribution(BaseModel):
    """Reconstruction of what the LLM's TriagePlan already claimed as its sources.

    Not new attribution elicited from a prompt change — the prompt is unchanged this
    iteration. See ADR-0015.
    """

    component_source: str
    similar_issue_refs: list[int]


class GroundingStatus(BaseModel):
    """Deterministic verification of the plan's claims against upstream pipeline outputs.

    See src/triage_iq/models/grounding.py:verify_plan_grounding and ADR-0015. Grounding here
    means traceable to this pipeline's own classifier_top3 / retrieval outputs for this
    request — not verification against world/ground truth.
    """

    component_grounded: bool
    component_reason: str
    similar_issue_refs: list[int]
    ungrounded_refs: list[int]
    all_grounded: bool


class DeclaredAttribution(BaseModel):
    """LLM-emitted source attribution (elicited by the prompt — contrast GroundingAttribution,
    a post-hoc reconstruction of the same plan; ADR-0015/ADR-0020)."""

    component_source: Literal["classifier_top3", "model_override"]
    component_override_reason: str = ""
    summary_cited_issues: list[int] = Field(default_factory=list)
    next_steps_cited_issues: list[int] = Field(default_factory=list)


class StageAbstention(BaseModel):
    """Per-stage selective-prediction gate result (ADR-0021).

    Deterministic threshold on an already-computed signal (component_confidence /
    grounding_status / CQR interval width) — not a new model. `reason` is empty when
    not abstained, else a short machine-readable code ("low_confidence", "ungrounded",
    "wide_interval").
    """

    abstained: bool
    reason: str = ""


class AbstentionStatus(BaseModel):
    """Selective-prediction gate over existing pipeline signals (ADR-0021).

    Priority stage is intentionally absent: priority_guess has no calibrated confidence
    signal anywhere in the pipeline to threshold, unlike component_confidence (ADR-0004)
    or the CQR interval (ADR-0010). See ADR-0021 for why that gap is flagged, not gated.
    """

    component: StageAbstention
    resolution: StageAbstention


class TriagePlan(BaseModel):
    """Structured triage plan produced by the LLM assistant.

    W4 Phase 2 note: expected_resolution_lower_days / expected_resolution_upper_days
    are now produced by the de-leaked model trained on a correct created_at split
    (CI coverage 77% vs previous 0%). resolution_bucket is an additional field
    computed directly from the bucket classifier — it supplements but does not
    replace the float fields. The LLM prompt still receives float signals because
    empirical eval showed bucket-only prompting regresses judge scores (−0.53 on
    resolution_estimate_reasonableness). See ADR-0009 T2.7.
    """

    predicted_component: str
    component_confidence: float = Field(ge=0.0, le=1.0)
    similar_issues: list[SimilarIssue] = Field(default_factory=list)
    expected_resolution_summary: str
    expected_resolution_lower_days: float = Field(ge=0.0)
    expected_resolution_upper_days: float = Field(ge=0.0)
    resolution_bucket: str = Field(
        default="days",
        description="Coarse bucket from ordinal classifier: hours/days/weeks/months/long. "
                    "Supplemental to the float fields; k8s passes 60% obo threshold, "
                    "vscode uses naive prior (low confidence). See ADR-0009.",
    )
    resolution_confidence_pct: float = Field(
        default=33.0, ge=0.0, le=100.0,
        description="Bucket classifier confidence (0–100%). Below 40% = low signal.",
    )
    resolution_interval_conformal: ConformalIntervalResult | None = Field(
        default=None,
        description=(
            "CQR-adjusted interval. Empirical marginal coverage under temporal drift: "
            "k8s 76.6% [74.0%, 79.1%], vscode 74.1% [69.4%, 78.3%]. "
            "None when conformal adjustments are unavailable. See ADR-0010."
        ),
    )
    priority_guess: Literal["low", "medium", "high"]
    priority_rationale: str
    suggested_assignee_class: str
    suggested_next_steps: list[str] = Field(min_length=1)
    triage_summary: str
    grounding: GroundingAttribution | None = Field(default=None)
    grounding_status: GroundingStatus | None = Field(default=None)
    declared_attribution: DeclaredAttribution | None = Field(
        default=None,
        description="LLM-declared source attribution (ADR-0020). None when the model omitted "
                    "or malformed the block — counted as a compliance failure, never a request "
                    "failure.",
    )
    abstention_status: AbstentionStatus | None = Field(
        default=None,
        description="Selective-prediction gate (ADR-0021). None when conformal adjustments "
                    "are unavailable for this repo (same fail-open policy as "
                    "resolution_interval_conformal) — never blocks the response.",
    )

    @field_validator("component_confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, float(v)))

    @field_validator("declared_attribution", mode="before")
    @classmethod
    def tolerant_attribution(cls, v):
        if v is None or isinstance(v, DeclaredAttribution):
            return v
        try:
            return DeclaredAttribution.model_validate(v)
        except Exception:
            return None  # malformed attribution -> compliance failure, not a request failure

    @model_validator(mode="after")
    def upper_ge_lower(self):
        if self.expected_resolution_upper_days < self.expected_resolution_lower_days:
            self.expected_resolution_upper_days = self.expected_resolution_lower_days
        return self

    @field_validator("resolution_bucket", mode="before")
    @classmethod
    def validate_bucket(cls, v):
        from triage_iq.models.resolution import BUCKET_LABELS
        if str(v) not in BUCKET_LABELS:
            return "days"
        return str(v)


# ---------------------------------------------------------------------------
# Main assistant class
# ---------------------------------------------------------------------------


class TriageAssistant:
    """Orchestrates Systems 1–3 and calls an LLM to produce a TriagePlan.

    Usage:
        assistant = TriageAssistant(
            repo="microsoft/vscode",
            classifier=tfidf_clf,
            detector=dup_detector,
            predictor=resolution_predictor,
            train_df=train_df,
        )
        plan = assistant.triage(issue_row)
    """

    def __init__(
        self,
        repo: str,
        classifier,
        detector,
        predictor,
        train_df: pd.DataFrame,
        groq_api_key: str | None = None,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
        cache=None,
    ) -> None:
        self.repo = repo
        self.classifier = classifier
        self.detector = detector
        self.predictor = predictor
        self.train_df = train_df
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self._cache = cache  # LLMCache | None

        key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise OSError(
                "GROQ_API_KEY not set. Export it or pass groq_api_key= to TriageAssistant."
            )
        self._groq_key = key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def triage(self, issue: pd.Series) -> TriagePlan:
        """Produce a TriagePlan for a single issue row."""
        t0 = time.perf_counter()
        signals = self._collect_signals(issue)
        plan, raw = self._call_llm(signals)
        elapsed = time.perf_counter() - t0
        logger.info("[%s] Triaged #%s in %.2fs", self.repo, issue.get("number", "?"), elapsed)
        return plan

    def triage_with_metadata(self, issue: pd.Series) -> tuple[TriagePlan, dict]:
        """Like triage() but also returns per-system timing and token usage."""
        t0 = time.perf_counter()
        signals = self._collect_signals(issue)
        plan, raw, usage, llm_status, cache_hit = self._call_llm_verbose(signals)
        # Write back bucket-classifier output — the LLM prompt excludes these by default
        # (TRIAGE_PROMPT_INCLUDE_BUCKET off), so without this the Pydantic field default
        # (33.0) is returned for every request regardless of repo. See ADR-0009.
        plan.resolution_confidence_pct = signals["resolution_conf_pct"]
        plan.resolution_bucket = signals["resolution_bucket"]

        # Grounding: reconstruction of what the LLM already emitted (component + similar-issue
        # refs), not new attribution elicited from a prompt change — the synthesis prompt is
        # unchanged this iteration. See ADR-0015.
        plan.grounding = GroundingAttribution(
            component_source=plan.predicted_component,
            similar_issue_refs=[s.number for s in plan.similar_issues],
        )
        retrieved_numbers = {s["number"] for s in signals["similar_raw"]}
        report = verify_plan_grounding(plan, signals["classifier_top3"], retrieved_numbers)
        plan.grounding_status = GroundingStatus(
            component_grounded=report.component_grounded,
            component_reason=report.component_reason,
            similar_issue_refs=report.similar_issue_refs,
            ungrounded_refs=report.ungrounded_refs,
            all_grounded=report.all_grounded,
        )
        elapsed = time.perf_counter() - t0

        t_llm = max(
            0.0,
            elapsed - signals["_t_classify"] - signals["_t_retrieve"] - signals["_t_predict"],
        )
        mid = (plan.expected_resolution_lower_days + plan.expected_resolution_upper_days) / 2.0

        metadata = {
            "system1_latency_ms": round(signals["_t_classify"] * 1000, 1),
            "system2_latency_ms": round(signals["_t_retrieve"] * 1000, 1),
            "system3_latency_ms": round(signals["_t_predict"] * 1000, 1),
            "system4_latency_ms": round(t_llm * 1000, 1),
            "total_latency_ms": round(elapsed * 1000, 1),
            "groq_tokens_prompt": usage.get("prompt_tokens", 0),
            "groq_tokens_completion": usage.get("completion_tokens", 0),
            "estimated_cost_usd": round(
                (usage.get("prompt_tokens", 0) * 0.27 + usage.get("completion_tokens", 0) * 0.27)
                / 1_000_000,
                8,
            ),
            "duplicate_count": len(plan.similar_issues),
            "predicted_resolution_days_p50": round(mid, 1),
            "resolution_bucket": plan.resolution_bucket,
            "resolution_confidence_pct": plan.resolution_confidence_pct,
            "llm_status": llm_status,
            "llm_cache_hit": cache_hit,
            "classifier_top3": signals["classifier_top3"],
        }
        logger.info(
            "[%s] Triaged #%s in %.2fs (groq %d+%d tok)",
            self.repo,
            issue.get("number", "?"),
            elapsed,
            metadata["groq_tokens_prompt"],
            metadata["groq_tokens_completion"],
        )
        return plan, metadata

    def triage_batch(
        self, df: pd.DataFrame, delay: float = 0.5
    ) -> list[tuple[int, TriagePlan | None, str | None]]:
        """Triage a batch of issues.

        Returns list of (issue_number, plan_or_None, error_or_None).
        """
        results: list[tuple[int, TriagePlan | None, str | None]] = []
        for i, (_, row) in enumerate(df.iterrows()):
            if i > 0:
                time.sleep(delay)
            try:
                plan = self.triage(row)
                results.append((int(row["number"]), plan, None))
            except Exception as exc:
                logger.warning("Failed to triage #%s: %s", row.get("number", "?"), exc)
                results.append((int(row.get("number", -1)), None, str(exc)))
        return results

    # ------------------------------------------------------------------
    # Signal collection
    # ------------------------------------------------------------------

    def _collect_signals(self, issue: pd.Series) -> dict:
        from triage_iq.prompts.triage_prompt import build_triage_prompt

        title = str(issue.get("title", ""))
        body = str(issue.get("body_clean", issue.get("body", "")))
        text = f"{title}. {body}"

        # System 1: TF-IDF top-3 (calibrated probabilities when calibrator is present)
        t1 = time.perf_counter()
        try:
            proba = self.classifier.predict_proba_calibrated(pd.Series([text]))
            classes = self.classifier.classes_()
            top_idx = np.argsort(proba[0])[::-1][:3]
            classifier_top3 = [
                {"label": classes[i], "confidence": float(proba[0][i])} for i in top_idx
            ]
        except Exception as e:
            logger.warning("Classifier failed: %s", e)
            classifier_top3 = [{"label": "unknown", "confidence": 0.0}]
        t_classify = time.perf_counter() - t1

        # System 2: BGE top-5 similar
        t2 = time.perf_counter()
        try:
            num = int(issue.get("number", -1))
            similar_raw = self.detector.retrieve(text, k=5, exclude_number=num if num > 0 else None)
        except Exception as e:
            logger.warning("Retrieval failed: %s", e)
            similar_raw = []
        t_retrieve = time.perf_counter() - t2

        # System 3: resolution prediction (float + bucket)
        t3 = time.perf_counter()
        try:
            from triage_iq.models.resolution import engineer_features

            issue_df = pd.DataFrame([issue])
            # Gold parquet uses "gold_component" — remap for engineer_features
            if "gold_component" in issue_df.columns and "component" not in issue_df.columns:
                issue_df = issue_df.rename(columns={"gold_component": "component", "gold_priority": "priority"})
            feats, _ = engineer_features(issue_df, train_df=self.train_df)
            # Align columns to what the model expects
            for col in self.predictor.feature_names:
                if col not in feats.columns:
                    feats[col] = 0.0
            feats = feats[self.predictor.feature_names]

            # Float output → drives LLM prompt (empirically better than bucket-only)
            pred_hrs = self.predictor.predict(feats)[0]
            lo_hrs, hi_hrs = self.predictor.predict_intervals(feats)
            pred_days = pred_hrs / 24.0
            lo_days = float(lo_hrs[0]) / 24.0
            hi_days = float(hi_hrs[0]) / 24.0

            # Bucket output → supplemental API field (not in LLM prompt)
            buckets, confs = self.predictor.predict_bucket(feats)
            resolution_bucket   = buckets[0]
            resolution_conf_pct = round(float(confs[0]) * 100, 1)
        except Exception as e:
            logger.warning("Resolution predictor failed: %s", e)
            pred_days, lo_days, hi_days = 7.0, 1.0, 30.0
            resolution_bucket, resolution_conf_pct = "days", 33.0
        t_predict = time.perf_counter() - t3

        # Config C: include bucket in prompt when TRIAGE_PROMPT_INCLUDE_BUCKET=1
        _include_bucket = os.environ.get("TRIAGE_PROMPT_INCLUDE_BUCKET") == "1"
        prompt = build_triage_prompt(
            issue_title=title,
            issue_body=body,
            classifier_top3=classifier_top3,
            similar_issues=similar_raw,
            resolution_point_days=pred_days,
            resolution_lower_days=lo_days,
            resolution_upper_days=hi_days,
            repo=self.repo,
            resolution_bucket=resolution_bucket if _include_bucket else None,
            resolution_confidence_pct=resolution_conf_pct if _include_bucket else None,
        )
        return {
            "prompt": prompt,
            "classifier_top3": classifier_top3,
            "similar_raw": similar_raw,
            "pred_days": pred_days,
            "lo_days": lo_days,
            "hi_days": hi_days,
            "resolution_bucket": resolution_bucket,
            "resolution_conf_pct": resolution_conf_pct,
            "_t_classify": t_classify,
            "_t_retrieve": t_retrieve,
            "_t_predict": t_predict,
        }

    # ------------------------------------------------------------------
    # LLM call + parsing
    # ------------------------------------------------------------------

    def _call_llm(self, signals: dict) -> tuple[TriagePlan, str]:
        plan, raw, _, _, _ = self._call_llm_verbose(signals)
        return plan, raw

    @staticmethod
    def _build_retry_messages(messages: list[dict], raw: str) -> list[dict]:
        return [
            *messages,
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Respond with ONLY valid JSON. No preamble, no markdown fences, no trailing commas."
                ),
            },
        ]

    def _cache_response(self, content: str, usage: dict) -> dict:
        """Build the dict stored by cache.set() for a live Groq call, with provenance.

        2026-08-27 (Part D, multi-account diagnostic session): cassette entries previously
        carried NO record of which model, account, or when they were recorded -- `model`
        was passed into cache.set() but silently dropped by both CassettePlayer and
        LLMCache, and there was no timestamp or account identifier at all. That made it
        structurally impossible to tell, after the fact, whether a recording mixed
        entries from different Groq accounts/tiers/quota states (see PLAN.md/ADR for the
        investigation this closes). GROQ_ACCOUNT_LABEL is optional and self-reported --
        there is no verified Groq API to ask "which account is this" -- so it defaults to
        "unknown" rather than silently implying a label that was never actually set.
        """
        return {
            "content": content,
            "usage": usage,
            "_provenance": {
                "model": self.model,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "account_label": os.environ.get("GROQ_ACCOUNT_LABEL", "unknown"),
            },
        }

    def _call_llm_verbose(self, signals: dict) -> tuple[TriagePlan, str, dict, str, bool]:
        """Return (plan, raw, usage, llm_status, cache_hit)."""
        from triage_iq.prompts.triage_prompt import (
            SYSTEM_PROMPT,
            SYSTEM_PROMPT_LEGACY,
            build_few_shot_examples,
            build_few_shot_examples_legacy,
        )

        # ADR-0020: attribution prompt is opt-in via TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1, off by
        # default so eval/cassettes/eval_cassette.json (recorded pre-attribution) and
        # reports/eval_baseline.json stay valid without re-baselining. See ADR-0020 "Baseline
        # decision". Same env-var-gated pattern as TRIAGE_PROMPT_INCLUDE_BUCKET above.
        _include_attribution = os.environ.get("TRIAGE_PROMPT_INCLUDE_ATTRIBUTION") == "1"
        system_prompt = SYSTEM_PROMPT if _include_attribution else SYSTEM_PROMPT_LEGACY
        few_shots = build_few_shot_examples() if _include_attribution else build_few_shot_examples_legacy()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(few_shots)
        messages.append({"role": "user", "content": signals["prompt"]})

        cache = getattr(self, "_cache", None)
        cache_key: str | None = None
        if cache is not None:
            cache_key = cache.compute_key(
                "groq", self.model, messages, self.temperature, self.max_tokens
            )
            cached = cache.get(cache_key)
            if cached is not None:
                raw = cached["content"]
                usage = cached.get("usage", {})
                try:
                    return self._parse_plan(raw), raw, usage, "ok", True
                except (json.JSONDecodeError, ValueError):
                    # Corrupted primary entry — a valid retry may already be cached
                    # from a prior recovery of this exact malformed response. Check
                    # before falling through to a live call (which a replay-mode
                    # cache with no real credentials cannot make).
                    retry_messages = self._build_retry_messages(messages, raw)
                    retry_key = cache.compute_key(
                        "groq", self.model, retry_messages, self.temperature, self.max_tokens,
                    )
                    cached_retry = cache.get(retry_key)
                    if cached_retry is not None:
                        raw2 = cached_retry["content"]
                        usage2 = cached_retry.get("usage", {})
                        try:
                            return (
                                self._parse_plan(raw2), raw2, usage2, "parse_retry_succeeded", True,
                            )
                        except (json.JSONDecodeError, ValueError):
                            pass  # retry entry also corrupted — fall through to live call

        raw, usage = self._groq_completion(messages)
        if cache is not None and cache_key is not None:
            cache.set(cache_key, "groq", self.model, messages, self._cache_response(raw, usage))
        llm_status = "ok"

        try:
            plan = self._parse_plan(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "LLM JSON parse failed (attempt 1): %s — retrying with strict prompt. Raw: %.300s",
                exc, raw,
            )
            retry_messages = self._build_retry_messages(messages, raw)
            # Check cache for retry call too
            parse_retry_key: str | None = None
            if cache is not None:
                parse_retry_key = cache.compute_key(
                    "groq", self.model, retry_messages, self.temperature, self.max_tokens,
                )
                cached2 = cache.get(parse_retry_key)
                if cached2 is not None:
                    raw2 = cached2["content"]
                    usage = cached2.get("usage", {})
                else:
                    raw2, usage = self._groq_completion(retry_messages)
                    cache.set(
                        parse_retry_key, "groq", self.model, retry_messages,
                        self._cache_response(raw2, usage),
                    )
            else:
                raw2, usage = self._groq_completion(retry_messages)
            try:
                plan = self._parse_plan(raw2)
                llm_status = "parse_retry_succeeded"
                raw = raw2
                logger.info("LLM JSON parse retry succeeded.")
            except (json.JSONDecodeError, ValueError) as exc2:
                logger.error(
                    "LLM JSON parse failed after retry: %s — using fallback plan. Raw: %.300s",
                    exc2, raw2,
                )
                plan = self._make_fallback_plan(signals)
                llm_status = "parse_failure"

        return plan, raw, usage, llm_status, False

    def _make_fallback_plan(self, signals: dict) -> TriagePlan:
        """Structured fallback when LLM JSON cannot be parsed after retry."""
        top = (signals.get("classifier_top3") or [{}])[0]
        return TriagePlan(
            predicted_component=str(top.get("label", "unknown")),
            component_confidence=float(top.get("confidence", 0.0)),
            similar_issues=[],
            expected_resolution_summary="LLM response unparseable; estimate from predictor only.",
            expected_resolution_lower_days=float(signals.get("lo_days", 1.0)),
            expected_resolution_upper_days=float(signals.get("hi_days", 30.0)),
            resolution_bucket=signals.get("resolution_bucket", "days"),
            resolution_confidence_pct=float(signals.get("resolution_conf_pct", 33.0)),
            priority_guess="medium",
            priority_rationale="LLM parse failure — priority defaulting to medium.",
            suggested_assignee_class="unknown",
            suggested_next_steps=["Manual triage required — LLM response parsing failed."],
            triage_summary=(
                "Automated triage degraded: LLM JSON parse failed after retry. "
                "Component from TF-IDF only; manual review recommended."
            ),
        )

    def _groq_completion(self, messages: list[dict]) -> tuple[str, dict]:
        try:
            from groq import APIStatusError, Groq, RateLimitError
        except ImportError as e:
            raise ImportError("pip install groq") from e

        client = Groq(api_key=self._groq_key)
        backoff = 5.0
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                )
                content = (resp.choices[0].message.content or "").strip()
                usage = {}
                if resp.usage:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                    }
                return content, usage
            except RateLimitError:
                if attempt == 5:
                    raise
                jitter = backoff * (0.5 + 0.5 * (attempt / 5))
                logger.warning("Rate limit hit — sleeping %.1fs (attempt %d/6)", jitter, attempt + 1)
                time.sleep(jitter)
                backoff = min(backoff * 2, 60.0)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < 5:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                else:
                    raise
        raise RuntimeError("Groq completion failed after 6 attempts")

    @staticmethod
    def _parse_plan(raw: str) -> TriagePlan:
        """Extract JSON from raw LLM output and validate as TriagePlan."""
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)

        # Find first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in LLM response: {raw[:200]}")
        data = json.loads(match.group(0))
        return TriagePlan.model_validate(data)
