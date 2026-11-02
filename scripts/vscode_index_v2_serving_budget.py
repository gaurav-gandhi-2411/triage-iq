"""Item 3c serving measurements (criteria 4-memory and 5, plus reported load time and sizes).

Pre-registration: docs/QP_VSCODE_INDEX_V2_PREREG_2026-11.md. One process per --config:

  served : triage_iq.api.loader.ModelStore.load_all() exactly as production does (prod flags below).
  v2     : the same, but the vscode detector is the dated v2 candidate. There is no vscode index flag
           in the loader yet, so loader._load_detector is wrapped to return the candidate for vscode
           (it replaces the served index at load time, so the peak never holds both).

Flags set before any import: HF_HUB_OFFLINE=1 (as the prod image), ENVIRONMENT=prod TRIAGE_CLEAN_REQUEST_BODY=1 TRIAGE_VSCODE_CLASSIFIER=family_v2
TRIAGE_K8S_INDEX=v2 TRIAGE_RESOLUTION_EVIDENCE=1 GROQ_API_KEY=dummy. Nothing leaves the machine: sockets are
blocked (connect/connect_ex/create_connection/getaddrinfo raise and are counted) and the Groq completion is
replaced by a raise of groq.APIConnectionError, which takes the production provider-outage degrade path.

Peak = Windows PeakWorkingSetSize (psutil is not installed). Latency: torch and faiss threads = 2 (Cloud Run
--cpu 2); 10 warm-up then 100 timed calls of retrieve(text, k=5, exclude_number, created_before) on the vscode
detector, timing the whole call (encode + search + filter).

`--index-load-only DIR` is the separate cold-load measurement of one index dir (fresh process).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

PROD_FLAGS = {
    "ENVIRONMENT": "prod",
    "TRIAGE_CLEAN_REQUEST_BODY": "1",
    "TRIAGE_VSCODE_CLASSIFIER": "family_v2",
    "TRIAGE_K8S_INDEX": "v2",
    "TRIAGE_RESOLUTION_EVIDENCE": "1",
    "GROQ_API_KEY": "dummy",
    "HF_HUB_OFFLINE": "1",  # the prod image bakes the embedder and runs offline (ci.yml L134)
    "CUDA_VISIBLE_DEVICES": "-1",
}
os.environ.update(PROD_FLAGS)

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "eval", ROOT / "scripts"):
    sys.path.insert(0, str(_p))

REPO = "microsoft/vscode"
VSCODE_SLUG = "microsoft_vscode"
GIB = 1024**3
BLOCKED_ATTEMPTS: list[str] = []


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def mem() -> dict[str, float]:
    """This process's current and peak working set in GiB (Windows psapi)."""
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return {"rss_gib": pmc.WorkingSetSize / GIB, "peak_rss_gib": pmc.PeakWorkingSetSize / GIB}


def block_network() -> None:
    """Make every outbound socket attempt raise (and be counted)."""
    import socket

    def blocked(name: str) -> Any:
        def _f(*a: Any, **k: Any) -> None:
            BLOCKED_ATTEMPTS.append(f"{name}{a[1:] if name.startswith('socket') else a}")
            raise RuntimeError("serving-budget harness: network access attempted")
        return _f

    socket.socket.connect = blocked("socket.connect")  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = blocked("socket.connect_ex")  # type: ignore[method-assign,assignment]
    socket.create_connection = blocked("create_connection")  # type: ignore[assignment]
    socket.getaddrinfo = blocked("getaddrinfo")  # type: ignore[assignment]


def pct(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy default)."""
    import numpy as np

    return float(np.percentile(np.asarray(xs), q))


def index_load_only(d: Path) -> dict[str, Any]:
    """Cold load of one index dir in this fresh process: meta + faiss, then the full retriever."""
    import faiss
    import joblib

    from triage_iq.models.similar_issues import SimilarIssueRetriever

    sizes = {p.name: p.stat().st_size for p in d.iterdir() if p.is_file()}
    t0 = time.perf_counter()
    meta = joblib.load(str(d / "meta.pkl"))
    t_meta = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx = faiss.read_index(str(d / "index.faiss"))
    t_faiss = time.perf_counter() - t0
    n = int(idx.ntotal)
    del meta, idx
    t0 = time.perf_counter()
    SimilarIssueRetriever.load(str(d))
    t_full = time.perf_counter() - t0
    return {"dir": str(d), "n_vectors": n, "file_bytes": sizes, "meta_pkl_load_s": t_meta,
            "faiss_read_s": t_faiss, "retriever_load_incl_bge_model_s": t_full, **mem()}


def run_config(a: argparse.Namespace) -> dict[str, Any]:
    """Load like production, run 20 degraded triage calls, then time vscode retrieval."""
    import faiss
    import groq
    import httpx
    import pandas as pd
    import torch

    torch.set_num_threads(2)
    faiss.omp_set_num_threads(2)
    block_network()
    from recording_config import issue_to_row

    from triage_iq.api import loader
    from triage_iq.models.similar_issues import SimilarIssueRetriever, _build_text
    from triage_iq.models.triage import TriageAssistant

    def no_llm(self: Any, *args: Any, **kw: Any) -> Any:
        raise groq.APIConnectionError(request=httpx.Request("POST", "http://llm-stub.invalid"))

    TriageAssistant._groq_completion = no_llm  # type: ignore[method-assign]
    if a.config == "v2":
        orig = loader._load_detector

        def swapped(models_dir: Path, slug: str, v2: bool = False) -> Any:
            if slug == VSCODE_SLUG:
                return SimilarIssueRetriever.load(str(a.candidate_dir))
            return orig(models_dir, slug, v2=v2)

        loader._load_detector = swapped  # type: ignore[assignment]

    out: dict[str, Any] = {"config": a.config, "env_flags": {k: v for k, v in PROD_FLAGS.items() if k != "GROQ_API_KEY"},
                           "torch_threads": torch.get_num_threads(), "mem_before_load": mem()}
    t0 = time.perf_counter()
    store = loader.ModelStore.load_all(data_dir=ROOT / "data", groq_api_key="dummy")
    out["load_all_s"] = time.perf_counter() - t0
    out["repos_loaded"] = store.repos
    assert len(store.repos) == 2, store.repos
    out["vscode_detector_n_vectors"] = int(len(store.get(REPO).detector.issue_numbers))
    out["mem_after_load"] = mem()

    rows = [json.loads(ln) for ln in a.eval_set.read_text(encoding="utf-8").split("\n") if ln.strip()]
    picked = [r for r in rows if r["repo"] == REPO][:10] + [r for r in rows if r["repo"] != REPO][:10]
    statuses: dict[str, int] = {}
    now = pd.Timestamp.now(tz="UTC")  # api/app.py defaults created_at to now
    for issue in picked:
        row = issue_to_row(issue)
        row["created_at"] = now
        _, meta = store.get(issue["repo"]).assistant.triage_with_metadata(row)
        statuses[meta["llm_status"]] = statuses.get(meta["llm_status"], 0) + 1
    out["triage_calls"] = {"n": len(picked), "llm_status_counts": statuses}
    out["mem_after_20_triage"] = mem()  # criterion 4: peak_rss_gib here

    det = store.get(REPO).detector
    pairs = pd.read_parquet(Path(r"D:\triageiq-data") / "splits" / "microsoft__vscode" / "dup_task_pairs.parquet")
    srcs = sorted({int(s) for s in pairs[pairs["split_of_source"] == "test"]["source"]})[:110]
    import qk_kaggle
    from check_prompt_neighbour_leakage import load_issue_dates, raw_created_at

    texts = qk_kaggle.build_texts(REPO, "issues", Path(r"D:\triageiq-data")).set_index("number")
    dates = load_issue_dates(Path(r"D:\triageiq-data") / "derived_v2", REPO)

    def time_detector(d: Any) -> dict[str, Any]:
        lat: list[float] = []
        for i, n in enumerate(srcs):
            row = texts.loc[n]
            text = _build_text(pd.Series([row["title"]]), pd.Series([row["body_clean"]]), d.model.tokenizer)[0]
            cb = dates.get(n) or raw_created_at(ROOT / "data" / "raw", REPO, n)
            t = time.perf_counter()
            d.retrieve(text, k=5, exclude_number=n, created_before=cb)
            if i >= 10:  # first 10 are warm-up
                lat.append((time.perf_counter() - t) * 1000.0)
        return {"n_timed": len(lat), "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "mean_ms": sum(lat) / len(lat),
                "max_ms": max(lat), "dated_index": getattr(d, "created_at", None) is not None}

    out["latency_serving_detector"] = time_detector(det)
    if a.config == "served":  # prod served index is undated, so its filter is a no-op; time the dated copy too
        out["latency_served_dated_copy"] = time_detector(SimilarIssueRetriever.load(str(a.served_dated_dir)))
    out["mem_final"] = mem()
    out["blocked_network_attempts"] = len(BLOCKED_ATTEMPTS)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", choices=["served", "v2"])
    ap.add_argument("--candidate-dir", type=Path)
    ap.add_argument("--served-dated-dir", type=Path)
    ap.add_argument("--eval-set", type=Path)
    ap.add_argument("--index-load-only", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.index_load_only:
        res = index_load_only(a.index_load_only)
    else:
        if not (a.config and a.candidate_dir and a.served_dated_dir and a.eval_set):
            ap.error("--config, --candidate-dir, --served-dated-dir and --eval-set are required")
        res = run_config(a)
    a.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
