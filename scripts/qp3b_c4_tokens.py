"""C4 (docs/QP3B_DESIGN_2026-10.md): prompt-token budget of prompt v3 vs v2 on the 150-issue set.

Zero LLM tokens, zero network (sockets blocked). For every row of
eval/sets/quality_push_2026-10/eval_set.jsonl the REAL TriageAssistant._collect_signals +
_call_llm_verbose message assembly runs against a capturing cache, so the counted messages are
exactly what would be sent (system + few-shot + user prompt, including the token-budget shrink
path). k8s rows: served classifier/predictor/conformal from --models-dir, retrieval from the
CANDIDATE index with neighbours restricted to issues created before the query (as
qp3b_c3_coverage.py), k=5 as in serving. vscode rows: the set's frozen similar_issues (neither
flag applies to vscode, so v3 == v2 there). v2 = both flags off; v3 = priority model (P4 loaded
in memory from its source dir) + resolution evidence.

Tokenizer: the project's own estimator (triage._estimate_prompt_tokens = tiktoken cl100k x the
calibrated 1.4896 ratio, the one the Groq budget guard uses). Raw cl100k and o200k_harmony (the
gpt-oss family encoding, message contents only, no chat-template overhead) are reported as
cross-checks. Threshold (pre-registered, not changeable): v3 mean <= 1.15 x v2 mean.

Usage: python scripts/qp3b_c4_tokens.py --models-dir DIR [--out reports/qp3b_c4_token_budget_2026-10.json]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

# The query encoder must come from the local HF cache: no hub HEAD requests (sockets are blocked).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from qp3_eval_set_lib import filter_neighbours  # noqa: E402
from qp3b_c3_coverage import EVAL_SET, EVIDENCE, INDEX, ISSUES, K_RETRIEVE  # noqa: E402

K8S, VSCODE = "kubernetes/kubernetes", "microsoft/vscode"
P4_DIR = Path(r"D:\triageiq-data\models\kubernetes__kubernetes\priority\p4")
THRESHOLD = 1.15
PLANNING_TOKENS_PER_ISSUE = 6000  # task/plan figure, prompt + completion
PLAN_TOTAL_TOKENS = 908_000  # design C4: ~908K at the 160K/day ledger cap
PRIOR_STUB = {"low": 0.25, "medium": 0.5, "high": 0.25}  # only if the P4 source dir is absent


class _Captured(Exception):  # noqa: N818 - control-flow sentinel
    def __init__(self, messages: list[dict]) -> None:
        self.messages = messages


class _CapturingCache:
    """compute_key runs after the budget guard, so the captured messages are the final ones."""

    def compute_key(self, provider, model, messages, temperature, max_tokens):  # noqa: ANN001, ANN201
        raise _Captured(messages)


class _CandidateRetriever:
    """Candidate index restricted to neighbours created before the current query (set per row)."""

    def __init__(self, det: Any, created: dict[int, pd.Timestamp]) -> None:
        self.det, self.created, self.row = det, created, None
        self._memo: dict[int, tuple[list[dict], Any]] = {}

    def retrieve_with_embedding(self, text: str, k: int = 5, exclude_number: int | None = None):  # noqa: ANN201
        r = self.row
        if r["number"] not in self._memo:
            hits, emb = self.det.retrieve_with_embedding(
                text, k=K_RETRIEVE, exclude_number=r["number"]
            )
            kept, _ = filter_neighbours(
                hits, r["number"], pd.Timestamp(r["created_at"]), self.created, k=k
            )
            self._memo[r["number"]] = (kept, emb)
        return self._memo[r["number"]]

    def retrieve(self, text: str, k: int = 5, exclude_number: int | None = None):  # noqa: ANN201
        return self.retrieve_with_embedding(text, k, exclude_number)[0]


def block_sockets() -> None:
    """Any network attempt raises (the o200k_harmony vocab must already be cached locally)."""

    def _deny(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("network blocked: C4 must be offline")

    socket.socket.connect = _deny  # type: ignore[method-assign]
    socket.create_connection = _deny  # type: ignore[assignment]


def load_priority_model() -> tuple[Any, str]:
    """(model, provenance): P4 packaged in memory from its source dir, else a fixed stub."""
    from triage_iq.models.priority import PriorityModel, PriorityPrediction

    if (P4_DIR / "model.joblib").exists():
        from qp3b_package_priority_model import package

        d = package(P4_DIR / "featurizer.joblib", P4_DIR / "model.joblib")
        return PriorityModel([(b["vectorizer"], b["field"]) for b in d["blocks"]], d["lr"]), (
            f"P4 from {P4_DIR} (in memory)"
        )

    class _Stub:
        def predict(self, title: str, body: str) -> Any:
            return PriorityPrediction("medium", dict(PRIOR_STUB))

    return _Stub(), "fixed stub probabilities (P4 source dir absent)"


def capture(asst: Any, issue: pd.Series) -> list[dict] | None:
    """Final messages the assistant would send; None if it degrades before any call."""
    asst._cache = _CapturingCache()
    sig = asst._collect_signals(issue)
    try:
        asst._call_llm_verbose(sig)
    except _Captured as c:
        return c.messages
    return None


def count_tokens(messages: list[dict], enc_cl: Any, enc_h: Any) -> dict[str, int]:
    """Project estimator (primary) plus raw cl100k and o200k_harmony contents-only counts."""
    from triage_iq.models.triage import _estimate_prompt_tokens

    texts = [m["content"] for m in messages]
    return {
        "estimator": _estimate_prompt_tokens(messages),
        "cl100k_raw": sum(len(enc_cl.encode(t)) for t in texts),
        "o200k_harmony": sum(len(enc_h.encode(t)) for t in texts),
    }


def summarize(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs, dtype=float)
    return {
        "n": len(a),
        "mean": round(float(a.mean()), 1),
        "median": round(float(np.median(a)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
    }


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "qp3b_c4_token_budget_2026-10.json")  # fmt: skip
    a = ap.parse_args(argv)

    import tiktoken

    enc_cl, enc_h = tiktoken.get_encoding("cl100k_base"), tiktoken.get_encoding("o200k_harmony")
    block_sockets()

    import run_eval
    from frozen_retriever import build_frozen_retrievers

    from triage_iq.models.resolution_evidence import EVIDENCE_BLOCK_MAX_CHARS, ResolutionEvidence
    from triage_iq.models.similar_issues import SimilarIssueRetriever

    run_eval.MODELS_DIR = a.models_dir
    rows = [json.loads(ln) for ln in EVAL_SET.read_text("utf-8").splitlines() if ln.strip()]
    iss = pd.read_parquet(ISSUES, columns=["number", "created_at"])
    created = dict(zip(iss["number"].astype(int), pd.to_datetime(iss["created_at"], utc=True), strict=True))  # fmt: skip
    cand = _CandidateRetriever(SimilarIssueRetriever.load(str(INDEX)), created)
    frozen = build_frozen_retrievers(EVAL_SET)
    detectors = {K8S: cand, VSCODE: frozen[VSCODE]}
    base = {
        repo: run_eval._load_models(repo, slug, _CapturingCache(), detectors)["assistant"]
        for repo, slug in run_eval.REPO_MAP.items()
    }
    pm, pm_prov = load_priority_model()
    v3_k8s = copy.copy(base[K8S])
    v3_k8s.priority_model, v3_k8s.resolution_evidence = pm, ResolutionEvidence.load(EVIDENCE)

    from checkpoint_validation import issue_to_row

    per_row: list[dict[str, Any]] = []
    for r in rows:
        row = issue_to_row(r)
        if r["repo"] == K8S:
            cand.row = r
            m2, m3 = capture(base[K8S], row), capture(v3_k8s, row)
            sig = v3_k8s._collect_signals(row)
            ev_chars = len(sig.get("_prompt_extra", {}).get("evidence_block", ""))
            ev_n = len(sig.get("resolution_evidence_neighbours", []))
        else:
            m2 = capture(base[r["repo"]], row)
            m3, ev_chars, ev_n = m2, 0, 0
        if m2 is None or m3 is None:
            per_row.append({"id": r["id"], "repo": r["repo"], "degraded": True})
            continue
        t2, t3 = count_tokens(m2, enc_cl, enc_h), count_tokens(m3, enc_cl, enc_h)
        assert ev_chars <= EVIDENCE_BLOCK_MAX_CHARS
        per_row.append(
            {
                "id": r["id"],
                "repo": r["repo"],
                "v2": t2,
                "v3": t3,
                "evidence_chars": ev_chars,
                "evidence_neighbours": ev_n,
            }  # fmt: skip
        )
    ok = [p for p in per_row if not p.get("degraded")]

    def stats(sel: list[dict], key: str) -> dict[str, dict[str, float]]:
        return {v: summarize([p[v][key] for p in sel]) for v in ("v2", "v3")}

    out: dict[str, Any] = {
        "n_rows": len(rows),
        "n_counted": len(ok),
        "degraded": len(rows) - len(ok),
    }
    for scope, sel in (
        ("pooled_150", ok),
        ("k8s_75", [p for p in ok if p["repo"] == K8S]),
        ("vscode_75", [p for p in ok if p["repo"] == VSCODE]),
    ):
        out[scope] = {}
        for key in ("estimator", "cl100k_raw", "o200k_harmony"):
            s = stats(sel, key)
            s["ratio_v3_over_v2_mean"] = round(s["v3"]["mean"] / s["v2"]["mean"], 4)
            out[scope][key] = s
    est = out["pooled_150"]["estimator"]
    ratio_pooled = est["ratio_v3_over_v2_mean"]
    ratio_k8s = out["k8s_75"]["estimator"]["ratio_v3_over_v2_mean"]
    k8s_ok = [p for p in ok if p["repo"] == K8S]
    # Projection: the ~6K/issue planning figure is Groq-reported prompt + completion for the v2
    # recordings; v3 adds only prompt text, so add the measured per-issue delta. o200k_harmony is
    # the model's own encoding (template overhead is identical in v2 and v3, so it cancels); the
    # estimator delta is shown too but is inflated by the 1.4896 calibration ratio.
    h = out["pooled_150"]["o200k_harmony"]
    d_h = h["v3"]["mean"] - h["v2"]["mean"]
    d_e = est["v3"]["mean"] - est["v2"]["mean"]
    n = len(ok)
    out["c4"] = {
        "threshold_ratio": THRESHOLD,
        "ratio_pooled_150_estimator": ratio_pooled,
        "ratio_k8s_only_estimator": ratio_k8s,
        "passed_pooled": ratio_pooled <= THRESHOLD,
        "passed_k8s_only": ratio_k8s <= THRESHOLD,
        "added_tokens_per_issue_pooled_o200k_harmony": round(d_h, 1),
        "added_tokens_per_issue_pooled_estimator": round(d_e, 1),
        "planning_tokens_per_issue": PLANNING_TOKENS_PER_ISSUE,
        "projected_rerecord_total_tokens_v3": round(n * (PLANNING_TOKENS_PER_ISSUE + d_h)),
        "projected_rerecord_total_tokens_v3_estimator_delta": round(
            n * (PLANNING_TOKENS_PER_ISSUE + d_e)
        ),
        "plan_total_tokens": PLAN_TOTAL_TOKENS,
        "fits_plan": n * (PLANNING_TOKENS_PER_ISSUE + d_h) <= PLAN_TOTAL_TOKENS,
        "orchestrator_decision": "2026-10-23: a projected total slightly above the ~908K planning "
        "figure (same 6 ledger days at 160K/day) is accepted; the pre-registered ratio "
        "threshold is the gate and it passes.",
    }
    out["evidence"] = {
        "k8s_rows_with_block": sum(p["evidence_chars"] > 0 for p in k8s_ok),
        "k8s_rows": len(k8s_ok),
        "mean_block_chars_when_present": round(
            float(np.mean([p["evidence_chars"] for p in k8s_ok if p["evidence_chars"]] or [0])), 1
        ),  # fmt: skip
        "max_block_chars": max((p["evidence_chars"] for p in k8s_ok), default=0),
    }
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    out["provenance"] = {
        "script": "scripts/qp3b_c4_tokens.py", "git_head_at_run": head,
        "tokenizer_primary": "triage_iq.models.triage._estimate_prompt_tokens (tiktoken cl100k x 1.4896)",  # fmt: skip
        "priority_model": pm_prov,
        "eval_set_sha256": sha256_file(EVAL_SET),
        "candidate_index_faiss_sha256": sha256_file(INDEX / "index.faiss"),
        "evidence_sha256": sha256_file(EVIDENCE),
        "models_dir": str(a.models_dir),
        "caveat": "candidate artifacts (nothing served); retrieval = candidate index filtered to "
        "neighbours created before the query; vscode rows use the set's frozen neighbours "
        "(v3 == v2 there); token counts are estimates of Groq's prompt_tokens, not Groq-reported",
    }  # fmt: skip
    out["rows"] = per_row
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_counted", "degraded", "pooled_150", "k8s_75", "c4", "evidence")}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
