"""Phase 2b: build the gold_related v2 CANDIDATE pairs from collected data. No retrain here.

Inputs:
  - existing corpus parquets + raw JSONs (k8s extended-pattern mine, ADR-0026: ~857 candidates)
  - data/raw/phase2b/vscode_dup_issues.jsonl + vscode_targets.jsonl (the Phase 2b scrape)

Guards applied (GG's go conditions):
  1. PR filter — a pair's TARGET must be an issue, not a PR (shared number space). Query-side
     PR composition is measured and reported, consistent with existing gold's ADR-0008 framing.
  2. ADR-0018 disjointness — any candidate pair touching a gold_triage_plans (judge-eval) issue
     is DROPPED (it would enter retrieval training). Overlaps with classifier/temporal train
     splits are reported for transparency (different models; not a drop criterion).
  3. Spot-check sample — 30 random dup pairs (seed 42) with matched snippets written to
     reports/phase2b_spotcheck_sample.json for human review BEFORE the yield is trusted.

Outputs:
  data/gold_related_v2_candidates.parquet   (candidates — gold_related.parquet is NOT touched)
  reports/phase2b_collection_report.json
  reports/phase2b_spotcheck_sample.json

Reproduce: python scripts/phase2b_build_candidates.py [--skip-vscode]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.data.preprocess import clean_text  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RAW_K8S = Path("data/raw/kubernetes_kubernetes")
CKPT_DIR = Path("data/raw/phase2b")
GOLD_PATH = Path("data/gold_related.parquet")
TRIAGE_GOLD_PATH = Path("data/gold_triage_plans.parquet")
OUT_PARQUET = Path("data/gold_related_v2_candidates.parquet")
OUT_REPORT = Path("reports/phase2b_collection_report.json")
OUT_SPOTCHECK = Path("reports/phase2b_spotcheck_sample.json")

MIN_BODY_CHARS = 10  # same validity rule as scripts/07_extract_related_pairs.py

# Extended patterns exactly as measured in ADR-0026 (scripts/phase2a_corpus_feasibility.py)
K8S_EXTENDED_PATTERNS = [
    r"[Rr]elated(?: to)? #(\d+)",
    r"[Ss]imilar to #(\d+)",
    r"[Rr]efs? #(\d+)",
    r"github\.com/[\w.-]+/[\w.-]+/issues/(\d+)",
]
K8S_CURRENT_PATTERNS = [
    r"[Dd]uplicate[sd]?(?: of)? #?(\d+)",
    r"[Dd]up(?:licate)? of #?(\d+)",
    r"[Ss]ame as #(\d+)",
    r"[Cc]losing as dup(?:licate)? of #?(\d+)",
    r"[Ss]ee(?: also)? #(\d+)",
    r"[Cc]loses? #(\d+)",
    r"[Ff]ixes? #(\d+)",
]


def _findall(text: str, patterns: list[str]) -> set[int]:
    refs: set[int] = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            refs.add(int(m.group(1)))
    return refs


def _is_pr_k8s(number: int, cache: dict[int, bool]) -> bool:
    if number not in cache:
        f = RAW_K8S / f"{number}.json"
        cache[number] = (
            "pull_request" in json.loads(f.read_text(encoding="utf-8")) if f.exists() else False
        )
    return cache[number]


def mine_k8s_extended(gold: pd.DataFrame) -> tuple[list[dict], dict]:
    """Emit the in-corpus extended-pattern candidate pairs ADR-0026 counted (+ current-pattern
    findall misses), with PR flags on both sides. Target-PR pairs are dropped (guard #1)."""
    repo = "kubernetes_kubernetes"
    df = pd.read_parquet(PROCESSED_DIR / f"issues_{repo}.parquet")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    by_num = df.set_index(df["number"].astype(int))
    g = gold[gold["repo"] == repo]
    gold_keys = {
        frozenset((int(q), int(o)))
        for q, o in zip(g["query_number"], g["original_number"], strict=True)
    }
    pr_cache: dict[int, bool] = {}
    pairs: list[dict] = []
    stats = {"target_is_pr_dropped": 0, "query_is_pr_kept": 0}
    seen: set[frozenset] = set()

    for _, row in df.iterrows():
        q = int(row["number"])
        combined = str(row.get("title", "")) + " " + str(row.get("body_clean", ""))
        cur = _findall(combined, K8S_CURRENT_PATTERNS)
        ext = _findall(combined, K8S_EXTENDED_PATTERNS) - cur
        for refs, source in ((cur, "body_related"), (ext, "body_related_ext")):
            for t in refs:
                key = frozenset((q, t))
                if t == q or t not in by_num.index or key in gold_keys or key in seen:
                    continue
                orig = by_num.loc[t]
                if orig["created_at"] > row["created_at"]:
                    continue
                if (
                    len(str(row["body_clean"]).strip()) <= MIN_BODY_CHARS
                    or len(str(orig["body_clean"]).strip()) <= MIN_BODY_CHARS
                ):
                    continue
                if _is_pr_k8s(t, pr_cache):
                    stats["target_is_pr_dropped"] += 1
                    continue
                if _is_pr_k8s(q, pr_cache):
                    stats["query_is_pr_kept"] += 1
                seen.add(key)
                pairs.append(
                    {
                        "repo": repo,
                        "query_number": q,
                        "original_number": t,
                        "query_title": row["title"],
                        "original_title": orig["title"],
                        "query_body": str(row["body_clean"]),
                        "original_body": str(orig["body_clean"]),
                        "source": source,
                        "confidence": "medium",
                        "query_is_pr": _is_pr_k8s(q, pr_cache),
                        "channel": "k8s_extended_mine",
                    }
                )
    log.info("[k8s] extended mine: %d pairs (%s)", len(pairs), stats)
    return pairs, stats


def measure_existing_gold_pr_rate(gold: pd.DataFrame) -> dict:
    """Honesty datum: how PR-heavy is the EXISTING k8s gold (built without a PR filter)?"""
    g = gold[gold["repo"] == "kubernetes_kubernetes"]
    pr_cache: dict[int, bool] = {}
    q_pr = sum(_is_pr_k8s(int(n), pr_cache) for n in g["query_number"])
    t_pr = sum(_is_pr_k8s(int(n), pr_cache) for n in g["original_number"])
    return {
        "n_pairs": int(len(g)),
        "query_is_pr": int(q_pr),
        "target_is_pr": int(t_pr),
        "note": "existing gold was built without PR awareness (ADR-0008 framing); reported for "
        "transparency — new candidates apply the target-must-be-issue filter.",
    }


def build_vscode_pairs(gold: pd.DataFrame) -> tuple[list[dict], dict]:
    dups_path = CKPT_DIR / "vscode_dup_issues.jsonl"
    targets_path = CKPT_DIR / "vscode_targets.jsonl"
    dups = [json.loads(x) for x in dups_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    targets = {
        r["number"]: r
        for r in (
            json.loads(x)
            for x in targets_path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        )
    }
    dup_by_num = {d["number"]: d for d in dups}
    g = gold[gold["repo"] == "microsoft_vscode"]
    gold_keys = {
        frozenset((int(q), int(o)))
        for q, o in zip(g["query_number"], g["original_number"], strict=True)
    }
    stats = {
        "dup_issues_scraped": len(dups),
        "dup_target_parsed": sum(1 for d in dups if d.get("dup_target")),
        "target_missing_404": 0,
        "target_is_pr_dropped": 0,
        "target_newer_dropped": 0,
        "body_too_short_dropped": 0,
        "already_in_gold": 0,
    }
    pairs: list[dict] = []
    seen: set[frozenset] = set()

    def resolve(t: int) -> dict | None:
        if t in dup_by_num:  # target itself is a scraped dup issue
            d = dup_by_num[t]
            return {
                "number": t,
                "is_pr": False,
                "title": d["title"],
                "body": d["body"],
                "created_at": d["created_at"],
            }
        return targets.get(t)

    def add_pair(d: dict, t: int, source: str, confidence: str, channel: str) -> None:
        key = frozenset((d["number"], t))
        if t == d["number"] or key in seen:
            return
        if key in gold_keys:
            stats["already_in_gold"] += 1
            return
        rec = resolve(t)
        if rec is None or rec.get("missing"):
            stats["target_missing_404"] += 1
            return
        if rec.get("is_pr"):
            stats["target_is_pr_dropped"] += 1
            return
        if str(rec.get("created_at") or "9999") > str(d["created_at"]):
            stats["target_newer_dropped"] += 1
            return
        q_body, _ = clean_text(str(d.get("body") or ""))
        t_body, _ = clean_text(str(rec.get("body") or ""))
        if len(q_body.strip()) <= MIN_BODY_CHARS or len(t_body.strip()) <= MIN_BODY_CHARS:
            stats["body_too_short_dropped"] += 1
            return
        seen.add(key)
        pairs.append(
            {
                "repo": "microsoft_vscode",
                "query_number": d["number"],
                "original_number": t,
                "query_title": d["title"],
                "original_title": rec["title"],
                "query_body": q_body,
                "original_body": t_body,
                "source": source,
                "confidence": confidence,
                "query_is_pr": False,
                "channel": channel,
                "dup_pattern": d.get("dup_pattern"),
                "dup_snippet": d.get("dup_snippet"),
            }
        )

    for d in dups:
        if d.get("dup_target"):
            add_pair(d, int(d["dup_target"]), "dup_comment", "high", "vscode_dup_scrape")
        for t in d.get("body_refs_current", []):
            add_pair(d, int(t), "body_related", "medium", "vscode_body_refs")
        for t in d.get("body_refs_extended", []):
            add_pair(d, int(t), "body_related_ext", "medium", "vscode_body_refs")

    log.info("[vscode] pairs built: %d (%s)", len(pairs), stats)
    return pairs, stats


def apply_disjointness(pairs: list[dict]) -> tuple[list[dict], dict]:
    """ADR-0018 line: drop any pair touching a judge-eval (gold_triage_plans) issue."""
    tg = pd.read_parquet(TRIAGE_GOLD_PATH)
    eval_nums = {
        ("kubernetes_kubernetes" if "kubernetes" in r else "microsoft_vscode", int(n))
        for r, n in zip(tg["repo"], tg["number"], strict=True)
    }
    kept, dropped = [], []
    for p in pairs:
        touches = {(p["repo"], p["query_number"]), (p["repo"], p["original_number"])}
        (dropped if touches & eval_nums else kept).append(p)
    detail = [
        {"repo": p["repo"], "query": p["query_number"], "target": p["original_number"]}
        for p in dropped
    ]
    assert not any(
        {(p["repo"], p["query_number"]), (p["repo"], p["original_number"])} & eval_nums
        for p in kept
    ), "disjointness violated after drop — bug"
    return kept, {"dropped_touching_judge_eval": len(dropped), "detail": detail}


def mine_k8s_forward(gold: pd.DataFrame, already: set[frozenset]) -> tuple[list[dict], dict]:
    """Mine pairs from the forward-scraped slice (#15,003-30,000, GraphQL, body channel).
    Queries are new-slice records; targets may be anywhere in #1-30,000. Same guards."""
    repo = "kubernetes_kubernetes"
    old = pd.read_parquet(PROCESSED_DIR / f"issues_{repo}.parquet")
    old["created_at"] = pd.to_datetime(old["created_at"], utc=True)
    lookup: dict[int, dict] = {
        int(r["number"]): {
            "title": r["title"],
            "body": str(r["body_clean"]),
            "created_at": r["created_at"].isoformat(),
            "is_pr": None,
        }
        for _, r in old.iterrows()
    }
    pr_cache: dict[int, bool] = {}
    new_slice: dict[int, dict] = {}
    for f in sorted(RAW_K8S.glob("*.json"), key=lambda p: int(p.stem)):
        n = int(f.stem)
        if n <= 15_002 or n > 30_000:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        body, _ = clean_text(str(d.get("body") or ""))
        rec = {
            "title": d.get("title"),
            "body": body,
            "created_at": str(d.get("created_at") or ""),
            "is_pr": "pull_request" in d,
        }
        new_slice[n] = rec
        lookup[n] = rec

    def target_is_pr(t: int) -> bool:
        if lookup[t]["is_pr"] is not None:
            return bool(lookup[t]["is_pr"])
        return _is_pr_k8s(t, pr_cache)  # old-slice records: flag lives in the raw JSON

    g = gold[gold["repo"] == repo]
    gold_keys = {
        frozenset((int(q), int(o)))
        for q, o in zip(g["query_number"], g["original_number"], strict=True)
    }
    pairs: list[dict] = []
    stats = {
        "new_slice_records": len(new_slice),
        "target_is_pr_dropped": 0,
        "query_is_pr_kept": 0,
        "cross_era_pairs": 0,
    }

    for q, qrec in new_slice.items():
        combined = str(qrec["title"] or "") + " " + qrec["body"]
        cur = _findall(combined, K8S_CURRENT_PATTERNS)
        ext = _findall(combined, K8S_EXTENDED_PATTERNS) - cur
        for refs, source in ((cur, "body_related"), (ext, "body_related_ext")):
            for t in refs:
                key = frozenset((q, t))
                if t == q or t not in lookup or key in gold_keys or key in already:
                    continue
                trec = lookup[t]
                if str(trec["created_at"]) > str(qrec["created_at"]):
                    continue
                if (
                    len(qrec["body"].strip()) <= MIN_BODY_CHARS
                    or len(trec["body"].strip()) <= MIN_BODY_CHARS
                ):
                    continue
                if target_is_pr(t):
                    stats["target_is_pr_dropped"] += 1
                    continue
                if qrec["is_pr"]:
                    stats["query_is_pr_kept"] += 1
                if t <= 15_002:
                    stats["cross_era_pairs"] += 1
                already.add(key)
                pairs.append(
                    {
                        "repo": repo,
                        "query_number": q,
                        "original_number": t,
                        "query_title": qrec["title"],
                        "original_title": trec["title"],
                        "query_body": qrec["body"],
                        "original_body": trec["body"],
                        "source": source,
                        "confidence": "medium",
                        "query_is_pr": bool(qrec["is_pr"]),
                        "channel": "k8s_forward_scrape",
                    }
                )
    log.info("[k8s] forward mine: %d pairs (%s)", len(pairs), stats)
    return pairs, stats


def report_split_overlap(pairs: list[dict]) -> dict:
    """Informational: overlap of candidate issues with classifier/temporal train splits."""
    out: dict = {}
    for repo in ("kubernetes_kubernetes", "microsoft_vscode"):
        nums = {p["query_number"] for p in pairs if p["repo"] == repo} | {
            p["original_number"] for p in pairs if p["repo"] == repo
        }
        row: dict = {"candidate_issues": len(nums)}
        for split in ("classifier_train", "temporal_train"):
            f = PROCESSED_DIR / f"{repo}_{split}.parquet"
            if f.exists():
                split_nums = set(pd.read_parquet(f, columns=["number"])["number"].astype(int))
                row[f"overlap_{split}"] = len(nums & split_nums)
        out[repo] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--skip-vscode", action="store_true", help="build k8s only (vscode scrape not finished)"
    )
    args = ap.parse_args()

    gold = pd.read_parquet(GOLD_PATH)
    k8s_pairs, k8s_stats = mine_k8s_extended(gold)
    existing_pr = measure_existing_gold_pr_rate(gold)

    seen_keys = {frozenset((p["query_number"], p["original_number"])) for p in k8s_pairs}
    k8s_fwd_pairs, k8s_fwd_stats = mine_k8s_forward(gold, seen_keys)

    vsc_pairs: list[dict] = []
    vsc_stats: dict = {"skipped": True}
    if not args.skip_vscode:
        vsc_pairs, vsc_stats = build_vscode_pairs(gold)

    all_pairs, disjoint = apply_disjointness(k8s_pairs + k8s_fwd_pairs + vsc_pairs)
    overlap = report_split_overlap(all_pairs)

    df = pd.DataFrame(all_pairs)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    # spot-check sample: 30 dup pairs, seed 42 (GG guard #4)
    dup_df = df[df["source"] == "dup_comment"]
    sample = dup_df.sample(n=min(30, len(dup_df)), random_state=42) if len(dup_df) else dup_df
    OUT_SPOTCHECK.write_text(
        json.dumps(
            [
                {
                    "query": int(r.query_number),
                    "target": int(r.original_number),
                    "query_title": r.query_title,
                    "target_title": r.original_title,
                    "pattern": r.dup_pattern,
                    "snippet": r.dup_snippet,
                }
                for r in sample.itertuples()
            ],
            indent=2,
            ensure_ascii=False,
        )
    )

    summary = {
        "generated_by": "scripts/phase2b_build_candidates.py",
        "existing_gold": gold.groupby(["repo", "source"])
        .size()
        .unstack(fill_value=0)
        .to_dict("index"),
        "existing_k8s_gold_pr_composition": existing_pr,
        "k8s_extended_mine": k8s_stats,
        "k8s_forward_mine": k8s_fwd_stats,
        "vscode_scrape": vsc_stats,
        "disjointness_adr0018": {k: v for k, v in disjoint.items() if k != "detail"},
        "disjointness_dropped_detail": disjoint["detail"],
        "split_overlap_informational": overlap,
        "candidates_by_repo_source": {}
        if df.empty
        else df.groupby(["repo", "source"]).size().unstack(fill_value=0).to_dict("index"),
        "totals": {
            "candidates": int(len(df)),
            "existing_gold": int(len(gold)),
            "combined_if_merged": int(len(df) + len(gold)),
        },
    }
    OUT_REPORT.write_text(json.dumps(summary, indent=2))
    log.info("Wrote %s (%d pairs), %s, %s", OUT_PARQUET, len(df), OUT_REPORT, OUT_SPOTCHECK)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("disjointness_dropped_detail",)},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
