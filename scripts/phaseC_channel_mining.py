"""Phase C: local-mining channels for genuine issue->related-issue pairs (ADR-0030 input).

Analysis only -- no gold-set writes, no scraping. Mines what's ALREADY in the local corpus for
two channels that require no new API calls:

  A. Extended body patterns ("related to #N", "similar to #N", "ref(s) #N", issue URLs) --
     patterns the current miner (scripts/07_extract_related_pairs.py) does NOT use. Same
     candidate class as phase2a_corpus_feasibility.py's EXTENDED_PATTERNS, re-run against the
     CURRENT v2 corpus (k8s #1-30,000, vscode dup-scraped) and CURRENT gold_related_v2 (so the
     count reflects headroom beyond what's already mined), filtered to issue->issue only (PR
     classification from raw JSON ground truth, same method as phase2b_pr_pair_breakdown.py).
  B. Comments "related to / see also / similar to #N" -- vscode only (k8s has zero comments_data
     locally; see scripts/phaseC_live_probe.py for the k8s comments feasibility probe).
  E. Label-cluster weak-related -- issues sharing the same fine-grained `component` label within
     a +-14 day window. Weak signal by construction; sampled for precision, not assumed.

For each channel x repo: candidate count (issue->issue, not already gold), whether target falls
in the LIVE retrieval index, and a fixed-seed sample for manual precision judging.

Output: reports/phaseC_channel_mining.json
Reproduce: python scripts/phaseC_channel_mining.py
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")
GOLD_PATH = Path("data/gold_related_v2.parquet")
MODELS_DIR = Path("data/models")
OUTPUT_PATH = Path("reports/phaseC_channel_mining.json")

REPOS = ["kubernetes_kubernetes", "microsoft_vscode"]

# Same patterns already mined by scripts/07_extract_related_pairs.py -- used to exclude overlap.
CURRENT_PATTERNS = [
    r"[Dd]uplicate[sd]?(?: of)? #?(\d+)",
    r"[Dd]up(?:licate)? of #?(\d+)",
    r"[Ss]ame as #(\d+)",
    r"[Cc]losing as dup(?:licate)? of #?(\d+)",
    r"[Ss]ee(?: also)? #(\d+)",
    r"[Cc]loses? #(\d+)",
    r"[Ff]ixes? #(\d+)",
]
# The new channel: patterns the current miner does not use (candidate-grade).
EXTENDED_PATTERNS = [
    r"[Rr]elated(?: to)? #(\d+)",
    r"[Ss]imilar to #(\d+)",
    r"[Rr]efs? #(\d+)",
    r"github\.com/[\w.-]+/[\w.-]+/issues/(\d+)",
]
COMMENT_RELATED_PATTERN = re.compile(
    r"[Rr]elated(?: to)? #(\d+)|[Ss]ee(?: also)? #(\d+)|[Ss]imilar to #(\d+)"
)

LABEL_CLUSTER_WINDOW_DAYS = 14
LABEL_CLUSTER_MAX_PER_ISSUE = 3  # cap fan-out per issue to avoid combinatorial explosion
SAMPLE_N = 30
SEED = 42


def _findall_refs(text: str, patterns: list[str]) -> set[int]:
    refs: set[int] = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            refs.add(int(m.group(1)))
    return refs


_pr_cache: dict[tuple[str, int], bool | None] = {}


def is_pr(repo: str, number: int) -> bool | None:
    key = (repo, number)
    if key not in _pr_cache:
        f = RAW_DIR / repo / f"{number}.json"
        _pr_cache[key] = (
            "pull_request" in json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        )
    return _pr_cache[key]


def _live_index_numbers(repo: str) -> set[int] | None:
    """Numbers in the LIVE-serving index (dup_index_{repo}_bge, no _v2 suffix). None if missing."""
    p = MODELS_DIR / f"dup_index_{repo}_bge" / "meta.pkl"
    if not p.exists():
        return None
    meta = joblib.load(str(p))
    return {int(n) for n in meta["issue_numbers"]}


def channel_a_extended_body(repo: str, df: pd.DataFrame, gold_keys: set[frozenset]) -> dict:
    num_to_idx = {int(n): i for i, n in enumerate(df["number"])}
    created = dict(zip(df["number"].astype(int), df["created_at"], strict=True))
    body_len = dict(
        zip(df["number"].astype(int), df["body_clean"].fillna("").str.strip().str.len(), strict=True)
    )
    live_nums = _live_index_numbers(repo)

    candidates = []
    for _, row in df.iterrows():
        q = int(row["number"])
        combined = str(row.get("title", "")) + " " + str(row.get("body_clean", ""))
        current_refs = _findall_refs(combined, CURRENT_PATTERNS)
        extended_refs = _findall_refs(combined, EXTENDED_PATTERNS) - current_refs
        for t in extended_refs:
            if t == q or t <= 0 or t not in num_to_idx:
                continue
            if created[t] > created[q]:
                continue
            if body_len.get(q, 0) <= 10 or body_len.get(t, 0) <= 10:
                continue
            key = frozenset((q, t))
            if key in gold_keys:
                continue
            qp, tp = is_pr(repo, q), is_pr(repo, t)
            if qp is None or tp is None or qp or tp:
                continue  # product-task = issue->issue only
            candidates.append(
                {
                    "query_number": q,
                    "original_number": t,
                    "query_title": row["title"],
                    "original_title": df.iloc[num_to_idx[t]]["title"],
                    "query_body_excerpt": str(row.get("body_clean", ""))[:300],
                    "target_in_live_index": bool(live_nums and t in live_nums and q in live_nums),
                }
            )
    # dedupe (a pair could match via multiple query rows only if q repeats, which it doesn't)
    n_in_live = sum(1 for c in candidates if c["target_in_live_index"])
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(candidates), size=min(SAMPLE_N, len(candidates)), replace=False) if candidates else []
    sample = [candidates[i] for i in sorted(sample_idx)]
    return {
        "channel": "A_extended_body_patterns",
        "candidate_count_issue_to_issue": len(candidates),
        "in_live_index_range": n_in_live,
        "sample_for_precision_review": sample,
    }


def channel_b_comments_vscode(df: pd.DataFrame, gold_keys: set[frozenset]) -> dict:
    repo = "microsoft_vscode"
    num_to_idx = {int(n): i for i, n in enumerate(df["number"])}
    created = dict(zip(df["number"].astype(int), df["created_at"], strict=True))
    live_nums = _live_index_numbers(repo)

    candidates = []
    n_with_comments = 0
    n_total = 0
    for number in df["number"].astype(int):
        f = RAW_DIR / repo / f"{number}.json"
        if not f.exists():
            continue
        n_total += 1
        d = json.loads(f.read_text(encoding="utf-8"))
        comments = d.get("comments_data") or []
        if not comments:
            continue
        n_with_comments += 1
        for c in comments:
            body = str(c.get("body", ""))
            for m in COMMENT_RELATED_PATTERN.finditer(body):
                t = int(next(g for g in m.groups() if g))
                q = number
                if t == q or t <= 0 or t not in num_to_idx:
                    continue
                if created[t] > created[q]:
                    continue
                key = frozenset((q, t))
                if key in gold_keys:
                    continue
                qp, tp = is_pr(repo, q), is_pr(repo, t)
                if qp is None or tp is None or qp or tp:
                    continue
                candidates.append(
                    {
                        "query_number": q,
                        "original_number": t,
                        "comment_excerpt": body[:200],
                        "target_in_live_index": bool(live_nums and t in live_nums and q in live_nums),
                    }
                )
    # dedupe by (q,t)
    seen = set()
    deduped = []
    for c in candidates:
        key = (c["query_number"], c["original_number"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    n_in_live = sum(1 for c in deduped if c["target_in_live_index"])
    rng = np.random.default_rng(SEED)
    sample_idx = (
        rng.choice(len(deduped), size=min(SAMPLE_N, len(deduped)), replace=False)
        if deduped
        else []
    )
    sample = [deduped[i] for i in sorted(sample_idx)]
    return {
        "channel": "B_comments_related_vscode",
        "local_comment_coverage": f"{n_with_comments}/{n_total} issues have comments_data locally",
        "candidate_count_issue_to_issue": len(deduped),
        "in_live_index_range": n_in_live,
        "sample_for_precision_review": sample,
    }


def channel_e_label_cluster(repo: str, df: pd.DataFrame, gold_keys: set[frozenset]) -> dict:
    live_nums = _live_index_numbers(repo)
    d2 = df[df["component"].notna() & (df["component"] != "")].copy()
    d2["created_at"] = pd.to_datetime(d2["created_at"], utc=True)
    d2 = d2.sort_values("created_at")
    candidates = []
    for component, group in d2.groupby("component"):
        g = group.reset_index(drop=True)
        nums = g["number"].astype(int).values
        dates = g["created_at"].values
        titles = g["title"].values
        n = len(g)
        for i in range(n):
            fan_out = 0
            for j in range(i - 1, -1, -1):
                if fan_out >= LABEL_CLUSTER_MAX_PER_ISSUE:
                    break
                gap_days = (dates[i] - dates[j]) / np.timedelta64(1, "D")
                if gap_days > LABEL_CLUSTER_WINDOW_DAYS:
                    break
                q, t = int(nums[i]), int(nums[j])
                key = frozenset((q, t))
                if key in gold_keys:
                    continue
                qp, tp = is_pr(repo, q), is_pr(repo, t)
                if qp is None or tp is None or qp or tp:
                    continue
                candidates.append(
                    {
                        "query_number": q,
                        "original_number": t,
                        "component": component,
                        "gap_days": round(float(gap_days), 1),
                        "query_title": titles[i],
                        "original_title": titles[j],
                        "target_in_live_index": bool(live_nums and t in live_nums and q in live_nums),
                    }
                )
                fan_out += 1
    n_in_live = sum(1 for c in candidates if c["target_in_live_index"])
    rng = np.random.default_rng(SEED)
    sample_idx = (
        rng.choice(len(candidates), size=min(SAMPLE_N, len(candidates)), replace=False)
        if candidates
        else []
    )
    sample = [candidates[i] for i in sorted(sample_idx)]
    return {
        "channel": "E_label_cluster_weak_related",
        "window_days": LABEL_CLUSTER_WINDOW_DAYS,
        "candidate_count_issue_to_issue": len(candidates),
        "in_live_index_range": n_in_live,
        "sample_for_precision_review": sample,
    }


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    report: dict = {"generated_by": "scripts/phaseC_channel_mining.py", "repos": {}}

    for repo in REPOS:
        df = pd.read_parquet(PROCESSED_DIR / f"issues_{repo}.parquet")
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        g = gold[gold["repo"] == repo]
        gold_keys = {
            frozenset((int(q), int(o)))
            for q, o in zip(g["query_number"], g["original_number"], strict=True)
        }
        log.info("[%s] channel A (extended body patterns) ...", repo)
        chan_a = channel_a_extended_body(repo, df, gold_keys)
        log.info("[%s] A: %d candidates (%d in live index)", repo, chan_a["candidate_count_issue_to_issue"], chan_a["in_live_index_range"])

        log.info("[%s] channel E (label-cluster) ...", repo)
        chan_e = channel_e_label_cluster(repo, df, gold_keys)
        log.info("[%s] E: %d candidates (%d in live index)", repo, chan_e["candidate_count_issue_to_issue"], chan_e["in_live_index_range"])

        report["repos"][repo] = {"channel_A_extended_body": chan_a, "channel_E_label_cluster": chan_e}

    log.info("channel B (vscode comments) ...")
    vsc_df = pd.read_parquet(PROCESSED_DIR / "issues_microsoft_vscode.parquet")
    vsc_df["created_at"] = pd.to_datetime(vsc_df["created_at"], utc=True)
    vsc_gold = gold[gold["repo"] == "microsoft_vscode"]
    vsc_gold_keys = {
        frozenset((int(q), int(o)))
        for q, o in zip(vsc_gold["query_number"], vsc_gold["original_number"], strict=True)
    }
    chan_b = channel_b_comments_vscode(vsc_df, vsc_gold_keys)
    log.info("[vscode] B: %d candidates (%d in live index)", chan_b["candidate_count_issue_to_issue"], chan_b["in_live_index_range"])
    report["repos"]["microsoft_vscode"]["channel_B_comments"] = chan_b
    report["repos"]["kubernetes_kubernetes"]["channel_B_comments"] = {
        "channel": "B_comments_related_k8s",
        "note": "k8s has ZERO comments_data locally (neither the original #1-15,002 scrape nor "
        "the #15,003-30,000 forward-scrape fetched comments) -- not locally mineable. "
        "See reports/phaseC_live_probe.json for a small live-API feasibility probe.",
        "candidate_count_issue_to_issue": None,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
