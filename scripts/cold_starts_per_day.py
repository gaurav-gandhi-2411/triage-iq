"""Count Cloud Run cold starts per day from the service's own logs and write reports/cold_starts_per_day.json.

A cold start is one "Application startup complete" log line from the triageiq-api service (every instance start,
including scale-out and deploys). Cloud Logging keeps 30 days by default, so run this while the window of interest is
still inside it; the committed JSON is the artifact the README / docs numbers are checked against.

The two comparison windows are fixed by the keep-warm monitor's first ping (2026-10-05 12:39:50 UTC, docs/DECISION_LOG_2026-10.md D10):
the six full days before it (2026-09-29 .. 2026-10-04) and the days after.

Usage: python scripts/cold_starts_per_day.py --account gaurav.gandhi1129@gmail.com [--project triageiq-prod-260812]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "cold_starts_per_day.json"
FILTER = (
    'resource.type="cloud_run_revision" AND resource.labels.service_name="triageiq-api" '
    'AND textPayload:"Application startup complete"'
)
PRE_MONITOR_DAYS = [f"2026-09-{d}" for d in (29, 30)] + [f"2026-10-0{d}" for d in (1, 2, 3, 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--project", default="triageiq-prod-260812")
    a = ap.parse_args()
    out = subprocess.run(  # noqa: S603
        [
            "gcloud",
            "logging",
            "read",
            FILTER,
            f"--project={a.project}",
            f"--account={a.account}",
            "--freshness=30d",
            "--limit=5000",
            "--format=value(timestamp)",
        ],
        capture_output=True,
        text=True,
        check=True,
        shell=sys.platform == "win32",
    ).stdout.split()
    stamps = sorted(datetime.fromisoformat(s.replace("Z", "+00:00")) for s in out)
    per_day = Counter(s.strftime("%Y-%m-%d") for s in stamps)
    pre = [per_day.get(d, 0) for d in PRE_MONITOR_DAYS]
    oct7_to_1221 = sum(
        1
        for s in stamps
        if s.strftime("%Y-%m-%d") == "2026-10-07" and (s.hour, s.minute) < (12, 21)
    )
    report = {
        "provenance": "scripts/cold_starts_per_day.py: count of 'Application startup complete' log lines, triageiq-api, Cloud Logging (30-day retention)",
        "first_log": stamps[0].isoformat(),
        "last_log": stamps[-1].isoformat(),
        "per_day": dict(sorted(per_day.items())),
        "pre_monitor_days": PRE_MONITOR_DAYS,
        "pre_monitor_min_per_day": min(pre),
        "pre_monitor_max_per_day": max(pre),
        "after_monitor_2026_10_06": per_day.get("2026-10-06", 0),
        "after_monitor_2026_10_07_until_12_21_utc": oct7_to_1221,
    }
    OUT.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote {OUT.relative_to(ROOT)}: pre-monitor {min(pre)}-{max(pre)}/day, 10-06: {report['after_monitor_2026_10_06']}, 10-07 to 12:21: {oct7_to_1221}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
