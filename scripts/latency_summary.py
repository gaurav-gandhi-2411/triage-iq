"""Machine-readable summary of the committed production latency replays and the contention test.

The replay text files (reports/latency_replay_*.txt) are the raw output of scripts/replay_triage.py joined with the
Cloud Run access log, one table row per call. This script re-reads those rows and the contention JSON and writes
reports/latency_summary.json: per-run per-call values, medians recomputed from the rows (not copied from the files'
own summary lines), the cold-start lifespan line, and the contention-test outcome counts. The README latency numbers
are checked against this file by scripts/readme_numbers_check.py.

Usage: python scripts/latency_summary.py [--check]   (--check fails if the committed JSON differs from a re-run)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = REPORTS / "latency_summary.json"

ROW = re.compile(
    r"^\d\s+(?:microsoft/vscode|kubernetes/kubernetes)\s+\d+\s+"
    r"(?P<client>[\d.]+)\s+(?P<srv>[\d.]+)\s+(?P<s1>[\d.]+)\s+(?P<s2>[\d.]+)\s+(?P<s3>[\d.]+)\s+(?P<s4>[\d.]+)\s+(?P<degr>True|False)\b",
    re.M,
)


def _rows(text: str) -> list[dict[str, float | bool]]:
    return [
        {
            "client_ms": float(m["client"]),
            "server_total_ms": float(m["srv"]),
            "stage1_ms": float(m["s1"]),
            "stage2_ms": float(m["s2"]),
            "stage3_ms": float(m["s3"]),
            "stage4_ms": float(m["s4"]),
            "degraded": m["degr"] == "True",
        }
        for m in ROW.finditer(text)
    ]


CALL = re.compile(r"^call \d: HTTP \d+ (\d+) ms", re.M)


def _logged(text: str) -> list[int]:
    return [int(x) for x in CALL.findall(text)]


def _run(rows: list[dict], logged: list[int]) -> dict:
    cols = ["client_ms", "server_total_ms", "stage1_ms", "stage2_ms", "stage3_ms", "stage4_ms"]
    assert len(logged) == len(rows)
    out: dict = {"n": len(rows), "client_ms_as_logged": logged, "calls": rows}
    out["max_client_ms_as_logged"] = max(logged)
    out["median_client_ms_as_logged"] = statistics.median(logged)
    out["median"] = {c: statistics.median(r[c] for r in rows) for c in cols}
    out["max_client_ms"] = max(r["client_ms"] for r in rows)
    answered = [r for r in rows if not r["degraded"]]
    degraded = [r for r in rows if r["degraded"]]
    out["llm_answered_n"] = len(answered)
    out["degraded_n"] = len(degraded)
    if degraded:
        out["degraded_client_ms_min"] = min(r["client_ms"] for r in degraded)
        out["degraded_client_ms_max"] = max(r["client_ms"] for r in degraded)
    return out


def build() -> dict:
    t07 = (REPORTS / "latency_replay_2026-10-07.txt").read_text(encoding="utf-8")
    r07 = _rows(t07)
    assert len(r07) == 10, len(r07)
    t12 = (REPORTS / "latency_replay_2026-10-12.txt").read_text(encoding="utf-8")
    ta1 = (REPORTS / "latency_replay_2026-10-12_post_a1.txt").read_text(encoding="utf-8")
    runs = {
        "pre_137": _run(r07[:5], _logged(t07)[:5]),
        "post_137": _run(r07[5:], _logged(t07)[5:]),
        "post_161": _run(_rows(t12), _logged(t12)),
        "post_a1": _run(_rows(ta1), _logged(ta1)),
    }
    cold = re.search(r"^(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\S+\s+\S+\s*$", t07, re.M)
    assert cold, "cold-start line not found"
    cold_start = {
        "log_lines": int(cold[1]),
        "cold_starts": int(cold[2]),
        "median_s": float(cold[3]),
        "min_s": float(cold[4]),
        "max_s": float(cold[5]),
    }
    cont = json.loads((REPORTS / "contention_test_2026-10-16.json").read_text(encoding="utf-8"))
    log = (REPORTS / "contention_test_2026-10-16_cloudrun_log.txt").read_text(encoding="utf-8")
    sleeps = [float(x) for x in re.findall(r"sleeping ([\d.]+)s", log)]
    contention = {
        "n": len(cont),
        "calls": [
            {k: c[k] for k in ("i", "repo", "offset_s", "http", "client_ms", "degraded", "reason")}
            for c in cont
        ],
        "llm_answered_n": sum(not c["degraded"] for c in cont),
        "degraded_n": sum(c["degraded"] for c in cont),
        "http_5xx_n": sum(c["http"] >= 500 for c in cont),
        "rate_limit_sleeps_s": sleeps,
        "retries_n": len(sleeps),
        "window_s": cont[-1]["offset_s"] - cont[0]["offset_s"],
        "span_to_last_response_s": cont[-1]["offset_s"] + cont[-1]["client_ms"] / 1000,
        "budget_left_at_degrade_s": float(re.search(r"([\d.]+)s left", log)[1]),
    }
    return {
        "provenance": "scripts/latency_summary.py over reports/latency_replay_*.txt and reports/contention_test_2026-10-16*",
        "runs": runs,
        "cold_start": cold_start,
        "contention_2026_10_16": contention,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    if a.check:
        ok = OUT.read_text(encoding="utf-8") == text
        print(
            "OK"
            if ok
            else "reports/latency_summary.json is stale: re-run scripts/latency_summary.py"
        )
        return 0 if ok else 1
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
