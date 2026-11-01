"""Zero-token preflight for a recording set: assemble every synthesis prompt, call no LLM.

Builds the assistants exactly as eval/record_cassettes.py does (recording_config.build_assistant, the set's
RECORDING_CONFIG.json, frozen retrievers, the conformal store) and runs each issue up to the point where the
request would be sent. The cache's compute_key raises a sentinel carrying the final messages, so
no Groq client is ever constructed; any socket connect is blocked on top of that.

Reports prompt tokens per issue and in total (the assistant's own tiktoken-proxy estimate, the number its
budget guard uses), prompts that were degraded (body truncated by the budget guard, or no viable
completion budget), whether eval/prose_interval_check._PROMPT_RE matches every prompt and whether the
interval carries "(coverage-calibrated)", how many prompts carry the resolution-evidence block, and the
point-estimate distribution and source per repo.

Usage: python scripts/recording_preflight.py --set-dir eval/sets/quality_push_2026-10 [--json-out FILE]
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import prose_interval_check as pic  # noqa: E402
import run_eval  # noqa: E402
from frozen_retriever import build_frozen_retrievers  # noqa: E402
from recording_config import build_assistant, issue_to_row, load_recording_config  # noqa: E402

from triage_iq.api.loader import _load_conformal_adjustments  # noqa: E402
from triage_iq.models import triage as triage_mod  # noqa: E402
from triage_iq.models.resolution_evidence import EVIDENCE_HEADER  # noqa: E402


class PromptCaptured(Exception):  # noqa: N818  (control-flow sentinel, not an error)
    """Raised by the capturing cache once the request messages are final."""

    def __init__(self, messages: list[dict], max_tokens: int) -> None:
        super().__init__("prompt captured")
        self.messages, self.max_tokens = messages, max_tokens


class CapturingCache:
    """Stands in for the cassette: compute_key hands the final request out and stops the call."""

    def compute_key(self, provider: str, model: str, messages: list[dict], *args: Any) -> str:
        raise PromptCaptured(messages, int(args[-1]))

    def get(self, key: str) -> None:  # pragma: no cover - unreachable, compute_key always raises
        raise AssertionError("preflight must never reach the cache lookup")


def inspect_prompt(messages: list[dict]) -> dict[str, Any]:
    """What the replay-side checks will see in the final user turn of `messages`."""
    user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    m = pic._PROMPT_RE.search(user)
    return {
        "regex_match": m is not None,
        "calibrated": bool(m and m["cal"]),
        "has_evidence_block": EVIDENCE_HEADER in user,
        "point_days": float(m["pt"]) if m else None,
        "interval": [float(m["lo"]), float(m["hi"])] if m else None,
        "est_prompt_tokens": triage_mod._estimate_prompt_tokens(messages),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals and per-repo point-estimate stats over per-issue preflight rows."""
    out: dict[str, Any] = {
        "n": len(rows),
        "total_est_prompt_tokens": sum(r.get("est_prompt_tokens", 0) for r in rows),
        "degraded": [r["id"] for r in rows if r["degraded"]],
        "regex_misses": [r["id"] for r in rows if not r.get("regex_match")],
        "uncalibrated": [r["id"] for r in rows if r.get("regex_match") and not r["calibrated"]],
        "with_evidence_block": sum(bool(r.get("has_evidence_block")) for r in rows),
        "per_repo": {},
    }
    for repo in sorted({r["repo"] for r in rows}):
        pts = [
            r["point_days"] for r in rows if r["repo"] == repo and r.get("point_days") is not None
        ]
        sources: dict[str, int] = {}
        for r in rows:
            if r["repo"] == repo:
                sources[str(r["point_source"])] = sources.get(str(r["point_source"]), 0) + 1
        out["per_repo"][repo] = {
            "n": sum(r["repo"] == repo for r in rows),
            "point_days_min_median_max": [min(pts), median(pts), max(pts)] if pts else None,
            "point_sources": sources,
        }
    return out


class _TruncationLog(logging.Handler):
    """Records which issue numbers the budget guard shortened (a degraded prompt)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.numbers: set[str] = set()

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("Prompt too large for #"):
            self.numbers.add(str(record.args[0]))  # type: ignore[index]


def _forbid_network() -> None:
    def _blocked(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("preflight: network access attempted")

    socket.socket.connect = _blocked  # type: ignore[method-assign,assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]


def run(set_dir: Path) -> dict[str, Any]:
    """Assemble every prompt of the set; no LLM call is possible on this path."""
    _forbid_network()
    eval_set = set_dir / "eval_set.jsonl"
    issues = run_eval._load_eval_set(eval_set)
    config = load_recording_config(ROOT, set_dir / "cassettes")
    frozen = build_frozen_retrievers(eval_set)
    models_dir = ROOT / "data" / "models"
    assistants = {}
    for repo, slug in run_eval.REPO_MAP.items():
        if repo not in frozen:
            continue
        assistants[repo] = build_assistant(
            repo,
            slug,
            models_dir=models_dir,
            processed_dir=ROOT / "data" / "processed",
            detector=frozen[repo],
            config=config,
            groq_api_key="preflight-no-llm",
            cache=CapturingCache(),
            conformal_adjustment=_load_conformal_adjustments(models_dir).get(repo),
        )["assistant"]
    trunc = _TruncationLog()
    logging.getLogger(triage_mod.__name__).addHandler(trunc)
    rows: list[dict[str, Any]] = []
    for issue in issues:
        a = assistants[issue["repo"]]
        info: dict[str, Any] = {"id": issue["id"], "repo": issue["repo"], "point_source": None}
        orig = a._call_llm_verbose

        def spy(signals: dict, _orig: Any = orig, _info: dict = info) -> Any:
            _info["point_source"] = signals.get("resolution_point_source", "model")
            return _orig(signals)

        a._call_llm_verbose = spy  # type: ignore[method-assign]
        try:
            _plan, meta = a.triage_with_metadata(issue_to_row(issue))
            # Returned without reaching compute_key: the budget guard degraded it, no prompt exists.
            info.update(degraded=True, degraded_reason=meta.get("llm_status"), regex_match=False)
        except PromptCaptured as cap:
            info.update(inspect_prompt(cap.messages), request_max_tokens=cap.max_tokens)
            info["degraded"] = str(issue["number"]) in trunc.numbers
            info["degraded_reason"] = (
                "issue body truncated by budget guard" if info["degraded"] else None
            )
        finally:
            del a._call_llm_verbose
        rows.append(info)
    return {
        "set_dir": str(set_dir),
        "config": config.__dict__,
        "assistant_max_tokens": {r: x.max_tokens for r, x in assistants.items()},
        "run_eval_TRIAGE_MAX_TOKENS": run_eval.TRIAGE_MAX_TOKENS,
        "summary": summarize(rows),
        "issues": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set-dir", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    res = run(args.set_dir.resolve())
    if args.json_out:
        args.json_out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    for r in res["issues"]:
        print(
            f"{r['id']:<14} tok={r.get('est_prompt_tokens', '-'):>5} max_tok={r.get('request_max_tokens', '-'):>5} "
            f"cal={r.get('calibrated')} pt={r.get('point_days')} src={r['point_source']} "
            f"degraded={r['degraded']}"
        )
    print(json.dumps({k: v for k, v in res.items() if k != "issues"}, indent=1))
    s = res["summary"]
    return 1 if s["degraded"] or s["regex_misses"] or s["uncalibrated"] else 0


if __name__ == "__main__":
    sys.exit(main())
