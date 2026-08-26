from __future__ import annotations

"""Estimate when the Groq TPD (tokens-per-day) quota will have enough headroom for the
next record-cassette.yml recording, instead of just checking "not at the wall this
instant" (what the workflow's pre-flight step does -- see its comment for why it can't
do better on its own: Groq only reveals the 'Used' figure inside a 429 error body once
you're already at the wall, never on a healthy response).

Uses reports/groq_tpd_probe_log.json: a log of real (timestamp, Used) readings captured
whenever a 429 happens, plus known usage bursts with their approximate token cost and
timing. Projects forward using the one fact we DO know for certain -- it's a ~24h rolling
window, so tokens spent at time X stop counting against the cap at roughly X+24h -- to
estimate when enough of today's usage will have rolled off for a ~215-220K recording.

This is a coarse projection, not an exact simulation -- see the log's own
"reconciliation_note" for a measured ~1-6% gap between summed burst tokens and actual
Groq-reported usage. Prefer a live --probe reading over the projection whenever you can
get one; use the projection to decide WHEN it's worth spending a live probe at all.

Usage:
    python scripts/groq_tpd_estimate.py                 # report using the existing log
    python scripts/groq_tpd_estimate.py --probe          # also make one live 1-token
                                                          # Groq call; if it 429s, append
                                                          # the fresh reading and re-report
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG_PATH = ROOT / "reports" / "groq_tpd_probe_log.json"

# Recording needs ~215K tokens for 64 issues (measured average ~3361-4505 tok/call across
# the recordings so far); require a margin above that so a fresh attempt doesn't die 90%
# through from token-cost variance between issues.
TARGET_HEADROOM = 220_000


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_log() -> dict:
    return json.loads(LOG_PATH.read_text(encoding="utf-8"))


def save_log(log: dict) -> None:
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def latest_reading(log: dict) -> dict:
    return max(log["readings"], key=lambda r: _parse_ts(r["timestamp_utc"]))


def rolloff_milestones(log: dict, anchor_time: datetime) -> list[tuple[datetime, int, str]]:
    """Bursts whose 24h-later rolloff time is after the anchor, sorted earliest-first.

    Each entry: (rolloff_time, tokens_freed, source). Only bursts ending after
    (anchor_time - 24h) can still be "not yet rolled off" at the anchor -- older bursts
    are already reflected in the anchor reading's Used figure, so re-subtracting them
    would double-count.
    """
    out = []
    for b in log.get("known_usage_bursts", []):
        end = _parse_ts(b["end_utc"])
        rolloff = end + timedelta(hours=24)
        if rolloff > anchor_time:
            out.append((rolloff, b["exact_tokens"], b["source"]))
    out.sort(key=lambda x: x[0])
    return out


def project_headroom(log: dict) -> None:
    limit = log["limit"]
    anchor = latest_reading(log)
    anchor_time = _parse_ts(anchor["timestamp_utc"])
    anchor_used = anchor["used"]
    now = datetime.now(timezone.utc)

    print(f"Anchor reading: {anchor_used}/{limit} used at {anchor['timestamp_utc']} ({anchor['source']})")
    print(f"Now: {now.isoformat(timespec='seconds')}  (age of anchor: {(now - anchor_time).total_seconds()/3600:.1f}h)")
    print(f"Target: >= {TARGET_HEADROOM} headroom for a comfortable full recording\n")

    milestones = rolloff_milestones(log, anchor_time)
    if not milestones:
        print("No known usage bursts roll off after the anchor reading -- nothing to project "
              "forward from. Run with --probe to get a fresh reading, or check "
              "known_usage_bursts in the log for staleness.")
        return

    running_used = anchor_used
    found = False
    print(f"{'rolloff time (UTC)':22} {'tokens freed':>12} {'projected used':>15} {'projected headroom':>19}  from")
    for rolloff_time, tokens, source in milestones:
        running_used -= tokens
        # Clamped at 0 for display -- the known_usage_bursts sum exceeds the anchor reading
        # by the amount in the log's reconciliation_note, so late milestones can otherwise
        # show a nonsensical negative "used". The clamp doesn't change WHEN the target is
        # first reached (that's determined before any milestone goes negative in practice).
        headroom = min(limit, limit - max(0, running_used))
        marker = " <-- target reached" if not found and headroom >= TARGET_HEADROOM else ""
        if marker:
            found = True
        print(f"{rolloff_time.isoformat(timespec='minutes'):22} {tokens:>12,} {max(0, running_used):>15,} {headroom:>19,}{marker}")

    print()
    if found:
        running = anchor_used
        for rolloff_time, tokens, source in milestones:
            running -= tokens
            if limit - running >= TARGET_HEADROOM:
                hours_out = (rolloff_time - now).total_seconds() / 3600
                if hours_out <= 0:
                    print(f"Projection says target headroom should already be available "
                          f"(crossed at {rolloff_time.isoformat(timespec='minutes')}) -- "
                          f"run --probe to confirm with a live reading before spending a full recording.")
                else:
                    print(f"Projected target headroom reached around {rolloff_time.isoformat(timespec='minutes')} "
                          f"(~{hours_out:.1f}h from now).")
                break
    else:
        print("None of the known bursts rolling off get us to the target headroom on their own -- "
              "there's more usage in the window than this log currently accounts for (see the "
              "log's reconciliation_note), or a probe is needed to find the true current Used.")

    print("\nCaveat: this is a coarse projection (see reconciliation_note in the log file), "
          "not a certified figure. A live --probe reading always overrides it.")


def live_probe() -> None:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY not set -- skipping live probe (fetch it with gcloud secrets "
              "versions access if you want a fresh reading).")
        return

    # Shells out to curl rather than urllib -- urllib's default User-Agent gets a Cloudflare
    # 403 (error 1010) in front of Groq's API in this environment; curl (used throughout this
    # session's manual pre-flight checks) doesn't hit that.
    import re
    import subprocess

    result = subprocess.run(
        [
            "curl", "-sS", "-w", "\n%{http_code}",
            "-X", "POST", "https://api.groq.com/openai/v1/chat/completions",
            "-H", f"Authorization: Bearer {key}",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "model": "openai/gpt-oss-20b",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }),
        ],
        capture_output=True, text=True, timeout=30,
    )
    *body_lines, http_code = result.stdout.rsplit("\n", 1)
    body = "\n".join(body_lines) if body_lines else result.stdout
    http_code = http_code.strip()

    if http_code == "200":
        print("Live probe: HTTP 200 -- not at the daily wall. Groq doesn't expose remaining "
              "headroom on a healthy response, so no new reading to log; the projection above "
              "is still the best available estimate.")
    elif http_code == "429" and "tokens per day" in body.lower():
        used_m = re.search(r"Used (\d+)", body)
        req_m = re.search(r"Requested (\d+)", body)
        if used_m:
            used = int(used_m.group(1))
            requested = int(req_m.group(1)) if req_m else None
            log = load_log()
            log["readings"].append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "used": used,
                "requested": requested,
                "source": "live probe via scripts/groq_tpd_estimate.py --probe",
            })
            save_log(log)
            print(f"Live probe: still at the wall, Used={used}/{log['limit']}. Logged this reading.")
        else:
            print(f"Live probe: got a 429 but couldn't parse Used from the body: {body[:300]}")
    else:
        print(f"Live probe: unexpected HTTP {http_code}: {body[:300]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="Also make one live 1-token Groq call")
    args = parser.parse_args()

    if args.probe:
        live_probe()
        print()

    project_headroom(load_log())


if __name__ == "__main__":
    main()
