"""Zero-token future-leakage check for a recording set (GG decision D95.2).

For every issue of the set, enumerates each neighbour that the recorder's prompt would carry and
checks that its created_at is STRICTLY before the query issue's created_at:

  * "similar"        - the SYSTEM 2 similar issues (up to 5; from the frozen retriever).
  * "evidence_nbr"   - the neighbours (top-3 of the similar issues) that got a RESOLUTION EVIDENCE
                       line; their created_at is checked like any neighbour.
  * "evidence_item"  - the dated items inside those lines (fix-PR merged_at, closing-comment time).
  * self-match       - neighbour number == the issue's own number (a violation in either kind).

Neighbours come from the real prompt-assembly path: the assistants are built exactly as
scripts/recording_preflight.py / eval/record_cassettes.py build them (eval/recording_config.py,
frozen retrievers, CapturingCache that stops the call before any LLM client exists, sockets blocked).
The numbers are read from the assistant's signals AND re-parsed out of the final user message, and
the two must agree (a mismatch is reported, never ignored).

Dates are NOT in the retriever indexes (meta.pkl holds only issue_numbers + texts), so created_at
comes from the scraped issue parquet (<derived-dir>/<owner>__<repo>/issues.parquet), falling back to
the raw issue JSON the index was built from (<raw-dir>/<owner>_<repo>/<n>.json; the v2 scrape starts
in 2023, older vscode neighbours exist only there). Where both exist they are compared and any
disagreement is reported. Evidence item dates (fix-PR merged_at, closing-comment time) come from
the evidence artifact the prompt itself uses, the same values the as-of filter compares.
A neighbour with no date is "unknown" and counts as a failure (fail closed).

Usage: python scripts/check_prompt_neighbour_leakage.py --set-dir eval/sets/quality_push_2026-10
           --derived-dir D:/triageiq-data/derived_v2 [--raw-dir data/raw] [--json-out FILE]
Exit code 0 only when there is no violation and no unknown date.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "eval", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BEFORE, EQUAL, AFTER, UNKNOWN = "before", "equal", "after", "unknown"
_SIMILAR_RE = re.compile(r"^\s+#(\d+) \(similarity:", re.MULTILINE)
_EVIDENCE_RE = re.compile(r"^#(\d+): ", re.MULTILINE)


def _utc(value: Any) -> pd.Timestamp:
    """tz-aware UTC timestamp (naive input is taken as UTC)."""
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def compare(query_ts: Any, neighbour_ts: Any) -> str:
    """BEFORE only when the neighbour is strictly earlier; a missing/unparseable date is UNKNOWN."""
    if neighbour_ts is None or pd.isna(neighbour_ts) or str(neighbour_ts) == "":
        return UNKNOWN
    try:
        n, q = _utc(neighbour_ts), _utc(query_ts)
    except (ValueError, TypeError):
        return UNKNOWN
    if pd.isna(n) or pd.isna(q):
        return UNKNOWN
    return BEFORE if n < q else EQUAL if n == q else AFTER


def prompt_neighbours(user_text: str) -> tuple[list[int], list[int]]:
    """(similar-issue numbers, evidence-line numbers) read out of a final user message."""
    head, _, tail = user_text.partition("--- SYSTEM 2: SIMILAR ISSUES")
    sys2, _, rest = tail.partition("--- SYSTEM 3")
    evidence = ""
    if "--- RESOLUTION EVIDENCE" in sys2:
        sys2, _, evidence = sys2.partition("--- RESOLUTION EVIDENCE")
    return (
        [int(x) for x in _SIMILAR_RE.findall(sys2)],
        [int(x) for x in _EVIDENCE_RE.findall(evidence)],
    )


def check_issue(
    issue: dict[str, Any],
    similar: list[int],
    evidence_nbrs: list[int],
    evidence_items: list[dict[str, Any]],
    dates: dict[int, Any],
) -> list[dict[str, Any]]:
    """One row per (neighbour or evidence item) of one issue, with status and self_match.

    `evidence_items` are {"neighbour", "kind", "number", "at"} dicts (the dated items rendered)."""
    q = issue["created_at"]
    rows: list[dict[str, Any]] = []

    def row(kind: str, n: int, ts: Any, **extra: Any) -> None:
        rows.append(
            {
                "id": issue["id"],
                "repo": issue["repo"],
                "issue_number": issue["number"],
                "issue_created_at": str(q),
                "type": kind,
                "neighbour": n,
                "neighbour_created_at": None if ts is None or ts == "" else str(ts),
                "status": compare(q, ts),
                "self_match": n == issue["number"],
                **extra,
            }
        )

    for n in similar:
        row("similar", n, dates.get(n))
    for n in evidence_nbrs:
        row("evidence_nbr", n, dates.get(n))
    for it in evidence_items:
        row("evidence_item", it["neighbour"], it["at"], item=f"{it['kind']} {it['number']}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per repo and type, plus the violation list (anything not strictly before)."""
    counts: dict[str, dict[str, Counter]] = {}
    for r in rows:
        c = counts.setdefault(r["repo"], {}).setdefault(r["type"], Counter())
        c["total"] += 1
        c[r["status"]] += 1
        c["self_match"] += bool(r["self_match"])
    per = {
        repo: {
            t: {k: c.get(k, 0) for k in ("total", BEFORE, EQUAL, AFTER, UNKNOWN, "self_match")}
            for t, c in types.items()
        }
        for repo, types in counts.items()
    }
    bad = [r for r in rows if r["status"] != BEFORE or r["self_match"]]
    return {
        "per_repo": per,
        "n_violations": sum(r["status"] in (EQUAL, AFTER) or r["self_match"] for r in bad),
        "n_unknown": sum(r["status"] == UNKNOWN for r in bad),
        "violations": bad,
    }


def load_issue_dates(derived_dir: Path, repo: str) -> dict[int, Any]:
    """number -> created_at from the scraped issue parquet. A number listed twice with different
    dates is ambiguous and is dropped (so it reports as unknown)."""
    df = pd.read_parquet(
        derived_dir / repo.replace("/", "__") / "issues.parquet", columns=["number", "created_at"]
    )
    out: dict[int, Any] = {}
    ambiguous: set[int] = set()
    for n, ts in zip(df["number"].tolist(), df["created_at"].tolist(), strict=True):
        n = int(n)
        if n in out and compare(out[n], ts) != EQUAL:
            ambiguous.add(n)
        out[n] = ts
    return {n: ts for n, ts in out.items() if n not in ambiguous}


def raw_created_at(raw_dir: Path, repo: str, number: int) -> str | None:
    """created_at from the raw GitHub API issue JSON the retriever index was built from
    (<raw-dir>/<owner>_<repo>/<number>.json); None when the file or the field is absent."""
    path = raw_dir / repo.replace("/", "_") / f"{number}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("created_at")
    except (OSError, ValueError):
        return None


class RepoDates:
    """created_at lookup for one repo: scraped issue parquet first, raw issue JSON as fallback.

    Records which source answered each number and any number on which the two sources disagree
    (the raw JSON is read for every looked-up number when a raw dir is given, as a cross-check)."""

    def __init__(self, repo: str, parquet: dict[int, Any], raw_dir: Path | None) -> None:
        self.repo, self.parquet, self.raw_dir = repo, parquet, raw_dir
        self.source: Counter = Counter()
        self.disagreements: list[dict[str, Any]] = []
        self._seen: dict[int, Any] = {}

    def get(self, number: int, default: Any = None) -> Any:
        """Date of `number` or `default`; memoised so each number is read and counted once."""
        if number in self._seen:
            return self._seen[number]
        pq = self.parquet.get(number)
        raw = raw_created_at(self.raw_dir, self.repo, number) if self.raw_dir else None
        if pq is not None and raw is not None and compare(pq, raw) != EQUAL:
            self.disagreements.append({"number": number, "parquet": str(pq), "raw": str(raw)})
        ts = pq if pq is not None else raw
        self.source["parquet" if pq is not None else "raw" if raw is not None else "none"] += 1
        self._seen[number] = ts if ts is not None else default
        return self._seen[number]


def _evidence_items(asst: Any, similar_raw: list[dict], as_of: Any) -> tuple[list, list]:
    """(items rendered, items the as-of filter dropped) for the top-3 neighbours of one issue."""
    from triage_iq.models.resolution_evidence import EVIDENCE_TOP_NEIGHBOURS

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    ev = asst.resolution_evidence
    if ev is None:
        return kept, dropped
    for s in similar_raw[:EVIDENCE_TOP_NEIGHBOURS]:
        n = int(s["number"])
        rec = ev.records.get(n)
        if rec is None:
            continue
        live = rec.before(as_of)
        # render_item shows only the first fix PR and the closing comment
        if live.fix_prs:
            kept.append(
                {
                    "neighbour": n,
                    "kind": "fix PR",
                    "number": live.fix_prs[0][0],
                    "at": live.fix_prs[0][3],
                }
            )
        if live.closing_comment:
            kept.append(
                {
                    "neighbour": n,
                    "kind": "closing comment",
                    "number": 0,
                    "at": live.closing_comment_at,
                }
            )
        dropped += [
            {"neighbour": n, "kind": "fix PR", "number": p[0], "at": p[3]}
            for p in rec.fix_prs
            if compare(as_of, p[3]) != BEFORE
        ]
        if rec.closing_comment and not live.closing_comment:
            dropped.append(
                {
                    "neighbour": n,
                    "kind": "closing comment",
                    "number": 0,
                    "at": rec.closing_comment_at,
                }
            )
    return kept, dropped


def collect(set_dir: Path, derived_dir: Path, raw_dir: Path | None) -> dict[str, Any]:
    """Assemble every prompt of the set (no LLM possible) and check its neighbours."""
    import recording_preflight as pf
    import run_eval
    from frozen_retriever import build_frozen_retrievers
    from recording_config import build_assistant, issue_to_row, load_recording_config

    from triage_iq.api.loader import _load_conformal_adjustments
    from triage_iq.models import triage as triage_mod

    pf._forbid_network()
    eval_set = set_dir / "eval_set.jsonl"
    issues = run_eval._load_eval_set(eval_set)
    config = load_recording_config(ROOT, set_dir / "cassettes")
    frozen = build_frozen_retrievers(eval_set)
    models_dir = ROOT / "data" / "models"
    assistants = {
        repo: build_assistant(
            repo,
            slug,
            models_dir=models_dir,
            processed_dir=ROOT / "data" / "processed",
            detector=frozen[repo],
            config=config,
            groq_api_key="preflight-no-llm",
            cache=pf.CapturingCache(),
            conformal_adjustment=_load_conformal_adjustments(models_dir).get(repo),
        )["assistant"]
        for repo, slug in run_eval.REPO_MAP.items()
        if repo in frozen
    }
    logging.getLogger(triage_mod.__name__).setLevel(logging.ERROR)
    dates = {
        repo: RepoDates(repo, load_issue_dates(derived_dir, repo), raw_dir) for repo in assistants
    }
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    removed_by_filter = 0
    no_prompt: list[str] = []
    for issue in issues:
        a = assistants[issue["repo"]]
        cap: dict[str, Any] = {}
        orig = a._call_llm_verbose

        def spy(signals: dict, _orig: Any = orig, _cap: dict = cap) -> Any:
            _cap["signals"] = signals
            return _orig(signals)

        a._call_llm_verbose = spy  # type: ignore[method-assign]
        try:
            a.triage_with_metadata(issue_to_row(issue))
            no_prompt.append(issue["id"])
            continue
        except pf.PromptCaptured as pc:
            msgs = pc.messages
        finally:
            del a._call_llm_verbose
        sig = cap["signals"]
        user = next(m["content"] for m in reversed(msgs) if m.get("role") == "user")
        p_sim, p_ev = prompt_neighbours(user)
        s_sim = [int(s["number"]) for s in sig["similar_raw"][:5]]
        s_ev = [int(n) for n in sig.get("resolution_evidence_neighbours", [])]
        if p_sim != s_sim or p_ev != s_ev:
            problems.append(f"{issue['id']}: prompt {p_sim}/{p_ev} != signals {s_sim}/{s_ev}")
        as_of = pd.Timestamp(issue["created_at"])
        items, removed = _evidence_items(a, sig["similar_raw"], as_of)
        removed_by_filter += len(removed)
        rows += check_issue(issue, p_sim, p_ev, items, dates[issue["repo"]])
    return {
        "set_dir": str(set_dir),
        "n_issues": len(issues),
        "evidence_items_removed_by_as_of_filter": removed_by_filter,
        "date_source_counts": {r: dict(d.source) for r, d in dates.items()},
        "date_source_disagreements": {r: d.disagreements for r, d in dates.items()},
        "prompt_vs_signals_mismatches": problems,
        "issues_without_prompt": no_prompt,
        **summarize(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set-dir", type=Path, required=True)
    ap.add_argument("--derived-dir", type=Path, required=True)
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="raw issue JSON root (data/raw): fallback + check",
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    res = collect(
        args.set_dir.resolve(), args.derived_dir.resolve(), args.raw_dir and args.raw_dir.resolve()
    )
    if args.json_out:
        args.json_out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "violations"}, indent=1))
    for v in res["violations"]:
        print("VIOLATION/UNKNOWN", json.dumps(v))
    clean = (
        not res["n_violations"]
        and not res["n_unknown"]
        and not res["prompt_vs_signals_mismatches"]
        and not res["issues_without_prompt"]
    )
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
