from __future__ import annotations
"""Part D bake-off screen harness (v3, 2026-08-30). 2-arm, day-checkpointed, gated by a
mandatory zero-quota dry run that cannot be skipped.

Arms: A = gpt-oss-20b few-shot, C = gpt-oss-120b few-shot. Arm B (no-few-shot) eliminated
2026-08-30 -- 3/8 parse failures, mathematically below the pre-registered 90% floor even
in the best case. Arm D (qwen3.6-27b) eliminated 2026-08-29 -- tokenizer footprint
overflows the free-tier TPM ceiling even on the shortest eval-set issue.

v2->v3: v1 and v2 of this harness never persisted plan content, issue context, or gold
data -- only diagnostic metadata (tokens, latency, headers, clamp/grounding flags). 33 of
39 successful v1/v2 calls are unrecoverable as a result (confirmed: no cache, no log, no
server-side history holds the content -- see the companion report's A3). v2's own
mid-session fix (adding plan/issue_title/issue_body/gold to the row) only covered the
last 6 calls. Per 2026-08-30 decision: v1/v2 result files are NOT reused as measurements
-- this version starts a clean output file and re-runs both surviving arms from scratch
on all 20 issues. The old files stay on disk as debugging record only.

MANDATORY GATE: main() always runs dry_run_self_check() first, against a fully mocked
Groq client (zero quota, zero network), asserting the output artifact contains every
field every pre-registered metric needs. This cannot be bypassed by calling a different
entrypoint -- there is only one entrypoint (main()), and it always self-checks first.
"""
import json
import os
import sys
import time
import types
from pathlib import Path

# 2026-08-30: a model-generated error payload containing a non-ASCII character (a Unicode
# non-breaking hyphen, U+2011, inside "test-infra") crashed the whole process on Windows'
# default cp1252 console encoding -- the row was already durably written to OUT_PATH
# before the print that crashed, so no data was lost, but the crash still killed 23
# remaining calls' progress for the session. Reconfigure stdout/stderr to UTF-8 with
# replacement so any future model-generated content (which is untrusted, arbitrary text)
# can never crash the harness on a print statement again.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("TRIAGE_PROMPT_INCLUDE_ATTRIBUTION", "1")

sys.path.insert(0, "src")
sys.path.insert(0, "eval")

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from frozen_retriever import build_frozen_retrievers  # noqa: E402
from triage_iq.models import triage as triage_mod  # noqa: E402
from triage_iq.models.component_classifier import load_classifier  # noqa: E402
from triage_iq.models.grounding import compute_grounding_status  # noqa: E402
from triage_iq.models.resolution import ResolutionTimePredictor  # noqa: E402
from triage_iq.models.triage import (  # noqa: E402
    GroundingAttribution,
    GroundingStatus,
    TriageAssistant,
    TruncatedCompletionError,
)
from triage_iq.prompts import triage_prompt  # noqa: E402

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "part_d_v3_results.jsonl"
ONLY_ISSUES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
ONLY_ARMS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["A", "C"]

ARMS = {
    "A": {"model": "openai/gpt-oss-20b", "attribution": "1", "no_few_shot": False},
    "C": {"model": "openai/gpt-oss-120b", "attribution": "1", "no_few_shot": False},
}

REPO_SLUG = {"kubernetes/kubernetes": "kubernetes_kubernetes", "microsoft/vscode": "microsoft_vscode"}

SAMPLE = [
    ("kubernetes/kubernetes", 14054), ("kubernetes/kubernetes", 12277), ("kubernetes/kubernetes", 14723),
    ("kubernetes/kubernetes", 14557), ("kubernetes/kubernetes", 12287), ("microsoft/vscode", 4993),
    ("kubernetes/kubernetes", 14835), ("kubernetes/kubernetes", 12122), ("kubernetes/kubernetes", 14135),
    ("kubernetes/kubernetes", 12254), ("microsoft/vscode", 4996), ("microsoft/vscode", 311284),
    ("kubernetes/kubernetes", 14363), ("kubernetes/kubernetes", 12665), ("kubernetes/kubernetes", 13057),
    ("microsoft/vscode", 278113), ("microsoft/vscode", 312423), ("kubernetes/kubernetes", 14762),
    ("kubernetes/kubernetes", 12784), ("kubernetes/kubernetes", 13435),
]
if ONLY_ISSUES is not None:
    SAMPLE = [(r, n) for r, n in SAMPLE if n in ONLY_ISSUES]

issues = [json.loads(line) for line in open("eval/eval_set.jsonl", encoding="utf-8").read().splitlines() if line.strip()]
by_repo_num = {(iss["repo"], iss["number"]): iss for iss in issues}

_frozen_retrievers = None
_model_cache: dict[str, dict] = {}


def _get_frozen_retrievers():
    global _frozen_retrievers
    if _frozen_retrievers is None:
        _frozen_retrievers = build_frozen_retrievers("eval/eval_set.jsonl")
    return _frozen_retrievers


def _load_repo_models(repo: str) -> dict:
    if repo in _model_cache:
        return _model_cache[repo]
    slug = REPO_SLUG[repo]
    classifier = load_classifier("data/models", slug)
    predictor = ResolutionTimePredictor.load(f"data/models/resolution_predictor_{slug}.pkl")
    train_df = pd.read_parquet(f"data/processed/{slug}_temporal_train.parquet")
    _model_cache[repo] = {"classifier": classifier, "predictor": predictor, "train_df": train_df}
    return _model_cache[repo]


def _make_assistant(repo: str, arm: str) -> TriageAssistant:
    m = _load_repo_models(repo)
    cfg = ARMS[arm]
    return TriageAssistant(
        repo=repo,
        classifier=m["classifier"],
        detector=_get_frozen_retrievers()[repo],
        predictor=m["predictor"],
        train_df=m["train_df"],
        groq_api_key=os.environ.get("GROQ_API_KEY", "dry-run-no-key-needed"),
        model=cfg["model"],
        max_tokens=2048,
        cache=None,
        use_structured_output=True,
    )


def _wrapped_groq_completion(self, messages, max_tokens=None):
    """Real (live) implementation -- captures headers/latency. Replaced by a mock during
    the dry run (see dry_run_self_check)."""
    from groq import Groq
    effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
    client = Groq(api_key=self._groq_key)
    kwargs: dict = {}
    if self.use_structured_output:
        from triage_iq.models.triage import _TRIAGE_PLAN_RESPONSE_FORMAT
        kwargs["response_format"] = _TRIAGE_PLAN_RESPONSE_FORMAT
    t0 = time.perf_counter()
    raw_resp = client.chat.completions.with_raw_response.create(
        model=self.model, messages=messages, temperature=self.temperature,
        max_tokens=effective_max_tokens, seed=self.seed, **kwargs,
    )
    latency_s = time.perf_counter() - t0
    headers = {k: v for k, v in raw_resp.headers.items() if "ratelimit" in k.lower()}
    resp = raw_resp.parse()
    content = (resp.choices[0].message.content or "").strip()
    finish_reason = resp.choices[0].finish_reason
    completion_tokens = resp.usage.completion_tokens if resp.usage else -1
    prompt_tokens = resp.usage.prompt_tokens if resp.usage else -1
    usage: dict = {
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "finish_reason": finish_reason, "structured_output": self.use_structured_output,
        "latency_s": round(latency_s, 3), "headers": headers, "effective_max_tokens_sent": effective_max_tokens,
    }
    if finish_reason == "length":
        raise TruncatedCompletionError(
            completion_tokens=completion_tokens, max_tokens=effective_max_tokens,
            content_preview=content, prompt_tokens=prompt_tokens,
        )
    return content, usage


_assistants: dict[tuple[str, str], TriageAssistant] = {}


def _already_done(out_path: str) -> set[tuple[str, int, str]]:
    done: set[tuple[str, int, str]] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "error" in row:
                continue
            done.add((row["repo"], row["number"], row["arm"]))
    return done


def run_one(repo: str, number: int, arm: str) -> dict:
    cfg = ARMS[arm]
    os.environ["TRIAGE_PROMPT_INCLUDE_ATTRIBUTION"] = cfg["attribution"]
    key = (repo, arm)
    if key not in _assistants:
        a = _make_assistant(repo, arm)
        a._groq_completion = types.MethodType(_wrapped_groq_completion, a)
        _assistants[key] = a
    assistant = _assistants[key]

    iss = by_repo_num[(repo, number)]
    row = pd.Series({
        "title": iss["title"], "body_clean": iss["body"], "number": iss["number"],
        "created_at": pd.Timestamp(iss["created_at"]) if iss.get("created_at") else pd.Timestamp("now", tz="UTC"),
    })
    signals = assistant._collect_signals(row)

    predicted_calls: list[int] = []
    orig_est = triage_mod._estimate_prompt_tokens

    def _capture(messages):
        v = orig_est(messages)
        predicted_calls.append(v)
        return v

    triage_mod._estimate_prompt_tokens = _capture

    orig_few_shot = triage_prompt.build_few_shot_examples
    if cfg["no_few_shot"]:
        triage_prompt.build_few_shot_examples = lambda: []

    result: dict = {
        "repo": repo, "number": number, "arm": arm, "model": cfg["model"],
        "attribution": cfg["attribution"], "no_few_shot": cfg["no_few_shot"],
    }
    try:
        for attempt in range(4):
            try:
                plan, raw, usage, llm_status, cache_hit = assistant._call_llm_verbose(signals)
                break
            except Exception as exc:
                if "RateLimit" in str(type(exc)):
                    wait = 15 * (attempt + 1)
                    print(f"  rate-limited, sleeping {wait}s (attempt {attempt+1}/4)", flush=True)
                    time.sleep(wait)
                    continue
                raise
        else:
            raise RuntimeError("exhausted retries")
    finally:
        triage_mod._estimate_prompt_tokens = orig_est
        triage_prompt.build_few_shot_examples = orig_few_shot

    plan.grounding = GroundingAttribution(
        component_source=plan.predicted_component,
        similar_issue_refs=[s.number for s in plan.similar_issues],
    )
    retrieved_numbers = {s["number"] for s in signals["similar_raw"]}
    resolved = compute_grounding_status(
        plan, signals["classifier_top3"], retrieved_numbers,
        enable_validated_override_rescue=assistant.enable_validated_override_rescue,
        issue_title=str(iss.get("title", "")), issue_body=str(iss.get("body", "")),
    )
    plan.grounding_status = GroundingStatus(
        component_grounded=resolved.component_grounded,
        component_reason=resolved.component_reason,
        similar_issue_refs=resolved.similar_issue_refs,
        ungrounded_refs=resolved.ungrounded_refs,
        all_grounded=resolved.all_grounded,
    )

    result["llm_status"] = llm_status
    result["predicted_prompt_tokens_calls"] = predicted_calls
    result["chars_cut"] = len(predicted_calls) > 1
    result["clamp_fired"] = (
        usage.get("effective_max_tokens_sent", 2048) < 2048
        if "effective_max_tokens_sent" in usage else None
    )
    result["truncation_fired"] = llm_status == "degraded_truncated"
    result["usage"] = usage
    result["all_grounded"] = plan.grounding_status.all_grounded
    result["ungrounded_refs"] = plan.grounding_status.ungrounded_refs
    result["predicted_component"] = plan.predicted_component
    result["plan"] = plan.model_dump(mode="json")
    result["issue_title"] = iss["title"]
    result["issue_body"] = iss["body"]
    result["gold"] = {
        "component": iss.get("gold_component"),
        "priority": iss.get("gold_priority"),
        "actual_resolution_days": iss.get("actual_resolution_days"),
    }
    return result


# ---------------------------------------------------------------------------
# Mandatory dry-run gate (A1/A2) -- zero quota, mocked Groq client
# ---------------------------------------------------------------------------

# Every field every pre-registered metric needs, and which metric needs it -- this list
# IS the gate. Add a metric to the pre-registration -> add its required field(s) here, or
# the dry run cannot catch a future version of the same bug class.
_REQUIRED_FIELDS: dict[str, str] = {
    "llm_status": "parse-success rate (elimination criterion)",
    "plan": "judge mean (primary decision metric) -- full generated content",
    "issue_title": "judge input",
    "issue_body": "judge input",
    "gold": "judge input (gold component/priority/actual_resolution_days)",
    "all_grounded": "grounding rate / fabrication rate (elimination + primary metrics)",
    "ungrounded_refs": "fabrication-rate diagnosis (which refs, not just the rate)",
    "usage": "latency, tokens, headers, clamp/truncation source data",
    "clamp_fired": "clamp-status split (quality confound guard, prereg S5)",
    "truncation_fired": "truncation-rate diagnostic",
    "predicted_component": "spot-check / debugging",
}
_REQUIRED_USAGE_FIELDS: dict[str, str] = {
    "prompt_tokens": "completion-token-distribution / quota accounting",
    "completion_tokens": "completion-token-distribution metric (C1), quality-per-token (C2)",
    "finish_reason": "truncation detection",
    "latency_s": "latency metric (D5)",
    "headers": "x-ratelimit-remaining-* capture (D5)",
    "effective_max_tokens_sent": "clamp-status derivation",
}
_REQUIRED_PLAN_FIELDS = [
    "predicted_component", "component_confidence", "similar_issues",
    "expected_resolution_summary", "expected_resolution_lower_days",
    "expected_resolution_upper_days", "priority_guess", "priority_rationale",
    "suggested_assignee_class", "suggested_next_steps", "triage_summary",
]


def _assert_row_has_every_metric_field(row: dict, label: str) -> list[str]:
    """Returns a list of violations (empty = clean). Never raises -- caller decides."""
    violations = []
    for field, why in _REQUIRED_FIELDS.items():
        if field not in row:
            violations.append(f"[{label}] missing top-level field '{field}' -- needed for: {why}")
    if "usage" in row:
        for field, why in _REQUIRED_USAGE_FIELDS.items():
            if field not in row["usage"]:
                violations.append(f"[{label}] missing usage.'{field}' -- needed for: {why}")
        if not row["usage"].get("headers"):
            violations.append(f"[{label}] usage.headers is empty -- x-ratelimit capture (D5) would be uncomputable")
    if "plan" in row and row["plan"] is not None:
        for field in _REQUIRED_PLAN_FIELDS:
            if field not in row["plan"]:
                violations.append(f"[{label}] plan missing '{field}' -- judge scoring would fail on this field")
    if "gold" in row:
        for field in ("component", "priority", "actual_resolution_days"):
            if row["gold"].get(field) is None:
                violations.append(f"[{label}] gold.'{field}' is None -- judge input incomplete")
    return violations


class _MockGroqResponse:
    def __init__(self, content: str, finish_reason: str, prompt_tokens: int, completion_tokens: int):
        self._content = content
        self._finish_reason = finish_reason
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    def parse(self):
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content=self._content),
            finish_reason=self._finish_reason,
        )
        usage = types.SimpleNamespace(
            prompt_tokens=self._prompt_tokens, completion_tokens=self._completion_tokens,
        )
        return types.SimpleNamespace(choices=[choice], usage=usage)

    @property
    def headers(self):
        return {
            "x-ratelimit-limit-requests": "1000", "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-requests": "999", "x-ratelimit-remaining-tokens": "1234",
            "x-ratelimit-reset-requests": "1m0s", "x-ratelimit-reset-tokens": "30s",
        }


def _mock_groq_completion_factory(scenario: str):
    """scenario: 'ok' (complete, valid plan), 'truncated' (finish_reason=length),
    'incomplete_schema' (stops mid-schema -- the real bug class found 2026-08-30)."""

    def _mock(self, messages, max_tokens=None):
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if scenario == "ok":
            content = json.dumps({
                "predicted_component": "api", "component_confidence": 0.71,
                "similar_issues": [{"number": 123, "similarity": 0.8, "relevance_note": "dry-run mock"}],
                "expected_resolution_summary": "mock summary", "expected_resolution_lower_days": 1.0,
                "expected_resolution_upper_days": 5.0, "resolution_bucket": "days",
                "resolution_confidence_pct": 33.0, "resolution_interval_conformal": None,
                "priority_guess": "medium", "priority_rationale": "mock rationale",
                "suggested_assignee_class": "mock team", "suggested_next_steps": ["mock step"],
                "triage_summary": "mock triage summary", "grounding": None, "grounding_status": None,
                "declared_attribution": None, "abstention_status": None,
            })
            return content, {
                "prompt_tokens": 5000, "completion_tokens": 300, "finish_reason": "stop",
                "structured_output": True, "latency_s": 0.01,
                "headers": _MockGroqResponse("", "", 0, 0).headers,
                "effective_max_tokens_sent": effective_max_tokens,
            }
        if scenario == "truncated":
            raise TruncatedCompletionError(
                completion_tokens=effective_max_tokens, max_tokens=effective_max_tokens,
                content_preview='{"predicted_component": "api", ...', prompt_tokens=5000,
            )
        raise RuntimeError(f"unknown mock scenario {scenario}")

    return _mock


def dry_run_self_check() -> None:
    """Zero-quota, zero-network end-to-end check: run the REAL run_one() pipeline (real
    signals, real grounding computation, real prompt-budget guard) against a MOCKED Groq
    client, through to a temp output file, then assert every pre-registered metric's
    required field is present and non-empty in the written artifact. Raises SystemExit(1)
    with every violation listed if anything would be uncomputable -- this is what makes
    the gate loud, not silent.
    """
    print("=== DRY RUN (zero quota, mocked Groq client) ===", flush=True)
    tmp_path = "part_d_dry_run_scratch.jsonl"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    repo, number = SAMPLE[0]
    arm = ONLY_ARMS[0] if ONLY_ARMS else "A"
    cfg = ARMS[arm]

    a = _make_assistant(repo, arm)
    a._groq_completion = types.MethodType(_mock_groq_completion_factory("ok"), a)
    _assistants[(repo, arm)] = a

    row = run_one(repo, number, arm)
    all_violations = _assert_row_has_every_metric_field(row, "scenario=ok")

    # Truncation path: must degrade cleanly (not crash) and still be analyzable.
    a2 = _make_assistant(repo, arm)
    a2._groq_completion = types.MethodType(_mock_groq_completion_factory("truncated"), a2)
    _assistants[(repo, arm)] = a2
    try:
        trunc_row = run_one(repo, number, arm)
        if not trunc_row.get("truncation_fired"):
            all_violations.append(
                "[scenario=truncated] truncation_fired is False/missing -- the harness "
                "did not correctly flag a TruncatedCompletionError as a truncation"
            )
        if trunc_row.get("llm_status") != "degraded_truncated":
            all_violations.append(
                f"[scenario=truncated] llm_status={trunc_row.get('llm_status')!r}, "
                "expected 'degraded_truncated'"
            )
    except Exception as exc:
        all_violations.append(
            f"[scenario=truncated] run_one() raised instead of degrading cleanly: "
            f"{type(exc).__name__}: {exc}"
        )

    # Re-fetch a clean assistant for the next real run (avoid leaking mocked methods).
    del _assistants[(repo, arm)]

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    if all_violations:
        print("DRY RUN FAILED -- the following would be uncomputable from the real artifact:", flush=True)
        for v in all_violations:
            print(f"  - {v}", flush=True)
        print(
            "\nRefusing to start the paced-quota run. Fix the harness and re-run -- "
            "this check cannot be skipped (main() always calls it first).",
            flush=True,
        )
        sys.exit(1)

    print(f"DRY RUN PASSED -- all {len(_REQUIRED_FIELDS)} top-level fields, "
          f"{len(_REQUIRED_USAGE_FIELDS)} usage fields, {len(_REQUIRED_PLAN_FIELDS)} plan "
          f"fields verified present; truncation path degrades cleanly. Zero quota spent.",
          flush=True)


def main() -> None:
    dry_run_self_check()  # MANDATORY, always first, cannot be bypassed from this entrypoint.

    done = _already_done(OUT_PATH)
    if done:
        print(f"Resuming: {len(done)} (repo, number, arm) calls already recorded in {OUT_PATH}, skipping them.", flush=True)
    out_f = open(OUT_PATH, "a", encoding="utf-8")
    total = len(SAMPLE) * len(ONLY_ARMS)
    done_count = 0
    for repo, number in SAMPLE:
        for arm in ONLY_ARMS:
            done_count += 1
            if (repo, number, arm) in done:
                print(f"[{done_count}/{total}] {repo} #{number} arm={arm} -- already done, skip", flush=True)
                continue
            print(f"[{done_count}/{total}] {repo} #{number} arm={arm}...", flush=True)
            try:
                res = run_one(repo, number, arm)
                out_f.write(json.dumps(res) + "\n")
                out_f.flush()
                u = res["usage"]
                print(f"  -> llm_status={res['llm_status']} prompt={u['prompt_tokens']} "
                      f"completion={u['completion_tokens']} finish={u['finish_reason']} "
                      f"clamp={res['clamp_fired']} latency={u['latency_s']}s", flush=True)
            except Exception as exc:
                err = {"repo": repo, "number": number, "arm": arm, "error": f"{type(exc).__name__}: {exc}"}
                out_f.write(json.dumps(err) + "\n")
                out_f.flush()
                print(f"  -> ERROR {err['error']}", flush=True)
    out_f.close()


if __name__ == "__main__":
    main()
