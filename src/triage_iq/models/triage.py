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
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from triage_iq.model_config import TRIAGE_MODEL
from triage_iq.models.grounding import compute_grounding_status

logger = logging.getLogger(__name__)


class TruncatedCompletionError(RuntimeError):
    """Raised when Groq's finish_reason == "length" -- the completion was cut off by
    max_tokens mid-generation, not a malformed-JSON parse failure.

    2026-08-28: this is the distinct failure mode that hid the actual defect behind this
    entire engagement -- a truncated completion used to fail silently as a generic JSON
    parse error, indistinguishable from the model genuinely emitting malformed JSON, so it
    was never possible to tell "the model is bad at JSON" apart from "max_tokens is too
    small" without manually inspecting raw content. Raised inside _groq_completion, before
    the caller ever gets a (content, usage) tuple back -- a truncated completion can
    therefore never reach cache.set() and can never enter a committed cassette.
    """

    def __init__(self, completion_tokens: int, max_tokens: int, content_preview: str) -> None:
        self.completion_tokens = completion_tokens
        self.max_tokens = max_tokens
        self.content_preview = content_preview
        super().__init__(
            f"Completion truncated: finish_reason='length' at completion_tokens="
            f"{completion_tokens} (max_tokens={max_tokens}). Raise max_tokens, not a "
            f"retry -- retrying at the same cap reproduces the same truncation. "
            f"Content tail: ...{content_preview[-80:]!r}"
        )


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------


class SimilarIssue(BaseModel):
    # extra="forbid" on every nested model here (2026-08-28): required for Groq's native
    # `strict: true` JSON-schema-constrained output, which rejects a schema unless
    # additionalProperties:false is set on every object -- Pydantic's model_json_schema()
    # doesn't set this by default on nested $defs.
    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    component_source: str
    similar_issue_refs: list[int]


class GroundingStatus(BaseModel):
    """Deterministic verification of the plan's claims against upstream pipeline outputs.

    See src/triage_iq/models/grounding.py:verify_plan_grounding and ADR-0015. Grounding here
    means traceable to this pipeline's own classifier_top3 / retrieval outputs for this
    request — not verification against world/ground truth.
    """

    model_config = ConfigDict(extra="forbid")

    component_grounded: bool
    component_reason: str
    similar_issue_refs: list[int]
    ungrounded_refs: list[int]
    all_grounded: bool


class DeclaredAttribution(BaseModel):
    """LLM-emitted source attribution (elicited by the prompt — contrast GroundingAttribution,
    a post-hoc reconstruction of the same plan; ADR-0015/ADR-0020)."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    abstained: bool
    reason: str = ""


class AbstentionStatus(BaseModel):
    """Selective-prediction gate over existing pipeline signals (ADR-0021).

    Priority stage is intentionally absent: priority_guess has no calibrated confidence
    signal anywhere in the pipeline to threshold, unlike component_confidence (ADR-0004)
    or the CQR interval (ADR-0010). See ADR-0021 for why that gap is flagged, not gated.
    """

    model_config = ConfigDict(extra="forbid")

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
# Groq native structured output (2026-08-28)
# ---------------------------------------------------------------------------


def _force_strict_schema_requirements(node: object) -> None:
    """Recursively satisfy Groq's `strict: true` JSON-schema requirements.

    Two independent requirements, both confirmed by trial (Groq's 400 response names
    exactly one violating $defs path at a time, so partial patching just surfaces the
    next one -- this walks the whole tree once for both instead of two passes):
    1. additionalProperties:false on every object.
    2. `required` must list every key in `properties` -- strict mode has no notion of an
       "optional" property; a Pydantic field with a default (e.g. component_override_reason
       str = "") is absent from Pydantic's own `required` list but Groq still needs it
       there. This does NOT change what the model can omit at the value level -- fields
       with a default still validate fine as their default if the model emits, say, "" or
       null for them; it only changes what the wire schema declares as present.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node.setdefault("additionalProperties", False)
            if "properties" in node:
                node["required"] = list(node["properties"].keys())
        for v in node.values():
            _force_strict_schema_requirements(v)
    elif isinstance(node, list):
        for v in node:
            _force_strict_schema_requirements(v)


def _inline_nullable_object_refs(schema: dict) -> None:
    """Rewrite `anyOf: [{$ref}, {type: null}]` properties into Groq's `type: [X, "null"]`
    form (2026-08-28, Part B).

    Pydantic's `model_json_schema()` renders `X | None = Field(default=None)` as an
    `anyOf` with a `$ref` branch and a `{"type": "null"}` branch. Live-tested against
    Groq's `strict: true` decoding: this `anyOf`+`$ref` shape was NOT reliably enforced --
    `grounding`, `grounding_status`, `declared_attribution`, and `abstention_status` (all
    four of TriagePlan's `X | None` fields) were silently omitted by the model in 2/7
    calls despite being listed in the schema's own `required` array, producing a 400
    ("missing properties") rather than a present-with-null value. Groq's own structured-
    outputs docs demonstrate the `anyOf` form for *array*-typed optional fields but present
    the `type` array as the primary pattern for optional values generally -- this inlines
    the referenced object's schema directly and sets `"type": [<object-type>, "null"]`,
    matching that primary documented form instead of the anyOf/$ref one that failed.
    Nested $refs *inside* a resolved object (e.g. AbstentionStatus -> StageAbstention) are
    untouched -- only the top-level optional-field anyOf/$ref/null pattern is rewritten.
    """
    defs = schema.get("$defs", {})
    if not defs:
        return

    def resolve(ref: str) -> dict:
        return defs[ref[len("#/$defs/"):]]

    def rewrite(node: object) -> None:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if isinstance(value, dict) and isinstance(value.get("anyOf"), list):
                    branches = value["anyOf"]
                    ref_branch = next((b for b in branches if "$ref" in b), None)
                    null_branch = next((b for b in branches if b.get("type") == "null"), None)
                    if ref_branch is not None and null_branch is not None and len(branches) == 2:
                        target = resolve(ref_branch["$ref"])
                        merged = dict(target)
                        merged["type"] = [merged.get("type", "object"), "null"]
                        for extra_key in ("default", "description"):
                            if extra_key in value:
                                merged[extra_key] = value[extra_key]
                        node[key] = merged
                        continue
                rewrite(value)
        elif isinstance(node, list):
            for item in node:
                rewrite(item)

    rewrite(schema.get("properties", {}))
    _prune_unreferenced_defs(schema)


def _prune_unreferenced_defs(schema: dict) -> None:
    """Drop `$defs` entries no longer reachable from `properties` after inlining.

    Avoids paying prompt tokens twice for the same object schema (once inlined into the
    optional field, once still sitting in `$defs` unused) -- see rule 15b/quota
    accounting in ADR-0052-adjacent Part A work. Reachability is transitive: a kept def
    may itself `$ref` another def (e.g. AbstentionStatus -> StageAbstention).
    """
    defs = schema.get("$defs", {})
    if not defs:
        return
    reachable: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref[len("#/$defs/"):]
                if name not in reachable:
                    reachable.add(name)
                    visit(defs.get(name, {}))
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(schema.get("properties", {}))
    for name in list(defs.keys()):
        if name not in reachable:
            del defs[name]


def _build_triage_plan_response_format() -> dict:
    """Groq response_format payload for native strict schema-constrained decoding.

    TriagePlan itself intentionally does NOT have extra="forbid" (app.py attaches
    _request_id/_llm_status/etc. to the response after synthesis), but the JSON SCHEMA
    sent to Groq for constrained decoding still needs additionalProperties:false
    everywhere per Groq's requirement -- that's a property of the wire schema, not of the
    Python class's own validation behavior, so patching the schema dict here doesn't
    conflict with leaving the class itself permissive.
    """
    schema = TriagePlan.model_json_schema()
    _inline_nullable_object_refs(schema)
    _force_strict_schema_requirements(schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": "TriagePlan", "schema": schema, "strict": True},
    }


_TRIAGE_PLAN_RESPONSE_FORMAT = _build_triage_plan_response_format()


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
        model: str = TRIAGE_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int = 42,
        cache=None,
        use_structured_output: bool = True,
        enable_validated_override_rescue: bool = False,
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
        # 2026-08-28: Groq native strict JSON-schema output as the primary mechanism,
        # regex-extract retained as fallback only (see _groq_completion). Constructor flag
        # (not a module constant) so tests/eval scripts can force the legacy path.
        self.use_structured_output = use_structured_output
        # 2026-08-28 (Part E2): disabled by default -- see grounding.py's
        # verify_override_reason_grounded for why the prior self-certifying version
        # (never merged) was unsound. A caller opts in deliberately, per request.
        self.enable_validated_override_rescue = enable_validated_override_rescue

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
        resolved = compute_grounding_status(
            plan,
            signals["classifier_top3"],
            retrieved_numbers,
            enable_validated_override_rescue=self.enable_validated_override_rescue,
            issue_title=str(issue.get("title", "")),
            issue_body=str(issue.get("body_clean", "")),
        )
        plan.grounding_status = GroundingStatus(
            component_grounded=resolved.component_grounded,
            component_reason=resolved.component_reason,
            similar_issue_refs=resolved.similar_issue_refs,
            ungrounded_refs=resolved.ungrounded_refs,
            all_grounded=resolved.all_grounded,
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

    def _call_llm_verbose(self, signals: dict) -> tuple[TriagePlan, str, dict, str, bool]:
        """Return (plan, raw, usage, llm_status, cache_hit)."""
        from triage_iq.prompts.triage_prompt import (
            SYSTEM_PROMPT,
            SYSTEM_PROMPT_LEGACY,
            SYSTEM_PROMPT_PROSE,
            build_few_shot_examples,
            build_few_shot_examples_legacy,
        )

        # ADR-0020: attribution prompt is opt-in via TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1, off by
        # default so eval/cassettes/eval_cassette.json (recorded pre-attribution) and
        # reports/eval_baseline.json stay valid without re-baselining. See ADR-0020 "Baseline
        # decision". Same env-var-gated pattern as TRIAGE_PROMPT_INCLUDE_BUCKET above.
        _include_attribution = os.environ.get("TRIAGE_PROMPT_INCLUDE_ATTRIBUTION") == "1"
        if _include_attribution:
            # 2026-08-28 (Part A): the JSON schema description is redundant prompt text when
            # native structured output is active -- Groq's response_format enforces it
            # structurally. Omit it from the prompt in that case; _groq_completion re-adds it
            # if structured output gets disabled mid-call (schema rejection fallback), since
            # the regex-extract path has no structural enforcement of its own.
            system_prompt = SYSTEM_PROMPT_PROSE if self.use_structured_output else SYSTEM_PROMPT
        else:
            system_prompt = SYSTEM_PROMPT_LEGACY
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
            cache.set(cache_key, "groq", self.model, messages, {"content": raw, "usage": usage})
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
                    cache.set(parse_retry_key, "groq", self.model, retry_messages, {"content": raw2, "usage": usage})
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
            kwargs: dict = {}
            if self.use_structured_output:
                kwargs["response_format"] = _TRIAGE_PLAN_RESPONSE_FORMAT
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                    **kwargs,
                )
                content = (resp.choices[0].message.content or "").strip()
                finish_reason = resp.choices[0].finish_reason
                usage = {}
                if resp.usage:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                    }
                usage["finish_reason"] = finish_reason
                usage["structured_output"] = self.use_structured_output
                if finish_reason == "length":
                    raise TruncatedCompletionError(
                        completion_tokens=usage.get("completion_tokens", -1),
                        max_tokens=self.max_tokens,
                        content_preview=content,
                    )
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
                elif (
                    e.status_code == 400
                    and self.use_structured_output
                    and "response_format" in str(e).lower()
                ):
                    # Native structured output rejected (e.g. this model/account
                    # combination doesn't actually support it, or a schema issue slipped
                    # past _build_triage_plan_response_format). Fall back to the classic
                    # unconstrained call + regex-extract for the rest of this assistant's
                    # lifetime -- retrying the same broken response_format every attempt
                    # would just fail identically each time.
                    logger.warning(
                        "Groq rejected structured output (%s) — falling back to "
                        "regex-extract for the remainder of this session.", e,
                    )
                    self.use_structured_output = False
                    # This call's `messages` may have been built with the schema
                    # description omitted (SYSTEM_PROMPT_PROSE, Part A) on the assumption
                    # that response_format would enforce it structurally. That's no longer
                    # true for the retry below or any further call on this assistant --
                    # make sure the schema is actually present before falling back to
                    # unconstrained decoding, which has no structural enforcement of its own.
                    from triage_iq.prompts.triage_prompt import _SCHEMA_BLOCK
                    if (
                        messages
                        and messages[0].get("role") == "system"
                        and "Schema:" not in messages[0]["content"]
                    ):
                        messages[0]["content"] += _SCHEMA_BLOCK
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
