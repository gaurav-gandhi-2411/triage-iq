"""Q3 part b: the stratified ~150-issue LLM end-to-end eval set (spec 4.5, 8, 10).

PRE-REGISTERED DESIGN (2026-10-23; committed BEFORE any sample is drawn; do not change after
seeing the sample). Builds the set only: no LLM call, no cassette, no recording.
 * Population per repo: TEST-split issues (D:/triageiq-data/splits/<repo>/test.parquet, created
   >= 2026-01-01) with classifier_eligible (leak_texthash and leak_dup_pair both False), that carry
   a component gold (k8s: >= 1 sig/* label at the end, ADR-0067; vscode: >= 1 non-excluded label,
   ADR-0070), whose cleaned body (preprocess.clean_text) is non-empty and <= BODY_CHARS (2000, the
   cap of build_vscode_expansion_eval_set.py; longer bodies are EXCLUDED, not truncated, so the
   LLM never sees a cut issue). Counts removed by each rule are reported. SELECTION EFFECT: the
   length rule drops long issues; estimates generalise to the short-body population only.
 * Gold. Component: k8s class = sig/* primary label with >= 30 train issues else `other`
   (ADR-0067); vscode class = parent area of the primary label if it is in the frozen 85-class
   family (ADR-0070, sha256 asserted), else `other` (strict headline; raw primary and parent are
   kept in meta). Priority: k8s label gold = qp3b_priority.gold3 on labels_final; otherwise (and for
   vscode) inferred from resolution days with infer_priority of the expansion builder (<1 d high,
   <7 d medium, else low); open issues without a label gold get null. Resolution: actual days
   (closed) or null with censored=true (open).
 * Strata (all binary except k8s target). k8s: has priority-label gold x target {dup_or_peer = source
   of a duplicate pair or member of a fix group; fix_pr = merged fix PR in fix_links only; none} x
   closed/open. vscode: primary in-family/out-of-family x has duplicate target x closed/open.
   Component class is spread INSIDE each stratum (below), not a stratum dimension (too sparse).
 * Allocation (75 per repo, proportional with minimums): every non-empty stratum gets
   min(N_h, MINIMUM) (k8s 3, vscode 5); the remainder is split in proportion to the spare capacity
   N_h - base_h by largest remainder (ties: stratum key). Within a stratum, items are drawn one
   at a time: pick the class with the fewest items drawn so far in this repo (ties: seeded random),
   then a uniform random member of that class. Strata are processed by ascending population (ties:
   key) with one numpy default_rng(42) per repo; members are sorted by issue number first.
   Deterministic. In/out-of-family ratio stays near the population's because the remainder is
   proportional (reported).
 * Reweighting: per-stratum x class population and sample counts are recorded; the weight of a
   row is N(h, c) / n(h, c). Population cells never drawn are listed (no weight can cover them).
 * similar_issues = frozen top-5 of the SERVED index (data/models/similar_issue_index_<repo>_bge),
   CPU float32, issue itself excluded: the same call eval/freeze_query_embeddings.py makes
   (retrieve_with_embedding), identical to eval/freeze_similar_issues.py output; k8s also gets
   frozen_query_embeddings.npz (RESOLUTION_EMBEDDINGS_REPOS). LIMITATION: that index holds issues
   numbered <= 30000 (k8s) / a 2014-2026 sample (vscode) and is not the Phase 2/3b retriever; it
   must be re-frozen with the final retriever before the single re-record. Meta records, per row,
   which duplicate / fix targets exist in that index.
 * Token estimate: spec 10 planning figure 6,000 tokens/issue x n vs the 160,000/day recorder cap,
   plus prompt-text size of this set vs the committed eval_set.jsonl (the 6K figure's origin).

v2 DESIGN (2026-10-23), supersedes v1 BEFORE ANY USE (nothing was recorded or scored on the v1
set, sha256 in V1_SET_SHA256 below). Every change, with its reason; everything not listed is unchanged.
 1. Length: long bodies are KEPT; only empty cleaned bodies are excluded. The prompt sends
    body[:800] (build_triage_prompt max_body_chars) and 120 chars per similar issue, so a long
    body costs no extra tokens; v1's cap removed 461 k8s / 2,070 vscode issues and biased the
    population. Rows keep the full cleaned body (<= 10,000, clean_text). Token estimate is
    recomputed from what the prompt sends (body[:800], similar text[:120]).
 2. vscode: simple random sampling inside each stratum (numpy default_rng(42) over members sorted
    by number, strata in the same ascending-population order); no class balancing; minimum 5 kept.
    Class coverage is an outcome. k8s keeps the class-spread draw. Weights: k8s N(h,c)/n(h,c),
    vscode N(h)/n(h); the report gives the range and Kish n_eff = (sum w)^2 / sum w^2 per repo.
 3. Temporal leakage: a neighbour created after the query cannot exist at serve time. Retrieval
    asks for K_RETRIEVE=50 and keeps the first 5 whose created_at < the query's. created_at comes
    from data/processed/<repo>_temporal_train.parquet and derived_v2 issues (the index meta has
    none). A neighbour with no known created_at falls back to its issue number being lower than
    the query's (GitHub numbers issues and PRs in one creation-ordered sequence) and is counted
    separately; one with neither is dropped and counted. Drop counts are in the report; the k8s
    query embeddings are unchanged (same query vector).
 4. similar_issues stays on the served index (production as-is). Relevance vs duplicate/fix links
    is unscorable on it (0 targets indexed): the Phase 2 / 3b retriever must be frozen into this
    set (re-run this script's freeze step with that retriever) before the single re-record. The
    report records `relevance_scoring` and, per repo, how many sampled issues have a dup/fix
    target present in derived_v2 and created before the query (indexable by a new retriever).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import qp0_ceiling_labels as cl  # noqa: E402
import qp2_vscode_taxonomy as tx  # noqa: E402
from build_vscode_expansion_eval_set import infer_priority  # noqa: E402
from qp0_make_splits import VAL_START, sha256_file  # noqa: E402
from qp3b_priority import gold3, gold4  # noqa: E402

from triage_iq.data.preprocess import clean_text  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
DATA = Path(r"D:\triageiq-data")
OUT_DIR = ROOT / "eval" / "sets" / "quality_push_2026-10"
REPORT = ROOT / "reports" / "qp3_eval_set_150_2026-10.json"
SEED, N_PER_REPO, K, K_RETRIEVE = 42, 75, 5, 50
PROMPT_BODY_CHARS, PROMPT_SIM_CHARS = 800, 120  # build_triage_prompt: body[:800], text[:120]
CASSETTE_PROMPT_CHARS = 16_577  # median synthesis prompt of the 64 groq entries, eval_cassette.json
V1_SET_SHA256 = "cfe8b8bddaae1b9743d4822435a6deda9651d54e6cc388adf335db6d0a3acd2d"
V1_NPZ_SHA256 = "7a2e06063d49d3feed96f946ccfcb2aa046d993d6efdcc9d30922bfe89ba9319"
MIN_ISSUES = 30  # ADR-0067 / ADR-0070 class floor (train issues)
TOKENS_PER_ISSUE, DAILY_CAP = 6000, 160_000  # spec 10; eval/tpd_budget_gate.RECORDER_CAP_TOKENS
VSCODE_CLASSES_SHA = "95129c95809c21574ed34d41a46358149322680a7ccb0e42e90f4b62e7b761ea"  # ADR-0070
REPOS = {
    "kubernetes/kubernetes": {"slug": "kubernetes__kubernetes", "id": "k8s", "minimum": 3,
                              "balance": True},
    "microsoft/vscode": {"slug": "microsoft__vscode", "id": "vscode", "minimum": 5,
                         "balance": False},
}  # fmt: skip


def allocate(pop: dict[str, int], total: int, minimum: int) -> dict[str, int]:
    """Proportional allocation with minimums (see module docstring); sums to min(total, sum)."""
    base = {k: min(v, minimum) for k, v in pop.items()}
    rest = total - sum(base.values())
    if rest < 0:
        raise ValueError(f"minimums need {sum(base.values())} > total {total}")
    cap = {k: pop[k] - base[k] for k in pop}
    tot_cap = sum(cap.values())
    rest = min(rest, tot_cap)
    exact = {k: (rest * c / tot_cap if tot_cap else 0.0) for k, c in cap.items()}
    out = {k: base[k] + math.floor(exact[k]) for k in pop}
    order = sorted(pop, key=lambda k: (-(exact[k] - math.floor(exact[k])), k))
    for k in order[: rest - sum(math.floor(e) for e in exact.values())]:
        out[k] += 1
    assert all(out[k] <= pop[k] for k in pop)
    return out


def draw(
    pop: pd.DataFrame, total: int, minimum: int, seed: int = SEED, balance: bool = True
) -> pd.DataFrame:
    """Draw the sample from `pop` (columns number, stratum, cls); returns the chosen rows.

    balance=True spreads draws over classes; False is simple random sampling in each stratum.
    """
    sizes = pop.groupby("stratum").size().to_dict()
    quota = allocate(sizes, total, minimum)
    rng = np.random.default_rng(seed)
    drawn: Counter[str] = Counter()
    chosen: list[int] = []
    for h in sorted(sizes, key=lambda k: (sizes[k], k)):
        cell = pop[pop["stratum"] == h].sort_values("number")
        if not balance:
            chosen += rng.choice(cell["number"].to_numpy(), size=quota[h], replace=False).tolist()
            continue
        by_cls = {c: g["number"].tolist() for c, g in cell.groupby("cls", sort=True)}
        for _ in range(quota[h]):
            avail = [c for c, v in by_cls.items() if v]
            c = min(avail, key=lambda c: (drawn[c], rng.random()))
            chosen.append(by_cls[c].pop(int(rng.integers(len(by_cls[c])))))
            drawn[c] += 1
    return pop[pop["number"].isin(chosen)].sort_values("number").reset_index(drop=True)


def strata_tables(pop: pd.DataFrame, sample: pd.DataFrame) -> dict[str, Any]:
    """Per-stratum and per-(stratum, class) population vs sample counts, and the weights."""
    ps, ss = pop.groupby("stratum").size(), sample.groupby("stratum").size()
    pc = pop.groupby(["stratum", "cls"]).size()
    sc = sample.groupby(["stratum", "cls"]).size()
    table = {h: {"population": int(ps[h]), "sample": int(ss.get(h, 0))} for h in ps.index}
    fine = {f"{h}|{c}": [int(n), int(sc.get((h, c), 0))] for (h, c), n in pc.items()}
    undrawn = [k for k, (_, s) in fine.items() if s == 0]
    return {"strata": table, "stratum_x_class": fine, "undrawn_cells": len(undrawn),
            "undrawn_population": int(sum(fine[k][0] for k in undrawn))}  # fmt: skip


def weight(fine: dict[str, list[int]], stratum: str, cls: str) -> float:
    n_pop, n_s = fine[f"{stratum}|{cls}"]
    return n_pop / n_s


def stratum_weight(strata: dict[str, dict[str, int]], stratum: str) -> float:
    return strata[stratum]["population"] / strata[stratum]["sample"]


def weight_stats(ws: list[float]) -> dict[str, float]:
    """Range of the weights and the Kish effective sample size (sum w)^2 / sum w^2."""
    w = np.asarray(ws, dtype=float)
    return {"min": float(w.min()), "max": float(w.max()), "n": len(w),
            "kish_n_eff": float(w.sum() ** 2 / (w**2).sum())}  # fmt: skip


def _read(slug: str, name: str, **kw: Any) -> pd.DataFrame:
    return pd.read_parquet(DATA / "derived_v2" / slug / f"{name}.parquet", **kw)


def load_population(repo: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Eligible test population of one repo with gold, strata and target columns."""
    slug = REPOS[repo]["slug"]
    test = pd.read_parquet(DATA / "splits" / slug / "test.parquet")
    test = test[test["classifier_eligible"]]
    nums = test["number"].tolist()
    cols = [
        "number", "created_at", "closed_at", "state", "title", "body", "labels_final",
        "labels_t72h", "resolution_hours", "censored", "scraped_at_max",
    ]  # fmt: skip
    iss = _read(slug, "issues", columns=cols, filters=[("number", "in", nums)])
    ev = _read(slug, "label_events", filters=[("number", "in", nums)])
    is_k8s = repo.startswith("kube")
    fam = (lambda x: x.startswith("sig/")) if is_k8s else tx.not_excluded
    parent = (lambda x: x) if is_k8s else tx.parent_of
    train = _read(slug, "issues", columns=["number", "created_at", "labels_final"],
                  filters=[("created_at", "<", VAL_START)]).pipe(
        lambda d: d[d["created_at"] >= pd.Timestamp("2023-01-01", tz="UTC")]
    )  # fmt: skip
    counts = Counter(parent(x) for ls in train["labels_final"] for x in {y for y in ls if fam(y)})
    classes = sorted(k for k, v in counts.items() if v >= MIN_ISSUES)
    if not is_k8s:
        assert hashlib.sha256("\n".join(classes).encode()).hexdigest() == VSCODE_CLASSES_SHA
    pi = cl.per_issue(iss.assign(text_hash=""), ev, fam)[["number", "primary"]]
    d = iss.merge(pi, on="number")
    removed = {"eligible_test": len(d), "no_component_gold": int(d["primary"].isna().sum())}
    d = d[d["primary"].notna()].copy()
    d["parent"] = d["primary"].map(parent)
    d["in_family"] = d["parent"].isin(classes)
    d["cls"] = d["parent"].where(d["in_family"], "other")
    d["body"] = d["body"].fillna("").map(lambda b: clean_text(b)[0])
    d["title"] = d["title"].fillna("").str.strip()
    removed["empty_body"] = int((d["body"].str.len() == 0).sum())
    removed["body_gt_prompt_cap_kept"] = int((d["body"].str.len() > PROMPT_BODY_CHARS).sum())
    d = d[(d["body"].str.len() > 0) & (d["title"] != "")]
    dp, fl, fg = (_read(slug, t) for t in ("duplicate_pairs", "fix_links", "fix_groups"))
    dup = dp.groupby("source")["canonical"].apply(lambda s: sorted(set(map(int, s)))).to_dict()
    fixpr = fl[fl["pr_merged"]].groupby("issue")["pr"].apply(lambda s: sorted(set(map(int, s))))
    peers: dict[int, list[int]] = {}
    for grp in fg["issues"]:
        for n in grp:
            peers[int(n)] = sorted(int(m) for m in grp if m != n)
    d["dup_targets"] = d["number"].map(lambda n: dup.get(n, []))
    d["fix_peers"] = d["number"].map(lambda n: peers.get(n, []))
    d["fix_prs"] = d["number"].map(lambda n: fixpr.get(n, []))
    d["closed"] = d["state"] == "CLOSED"
    d["prio_label"] = d["labels_final"].map(gold3) if is_k8s else None
    d["days"] = (d["resolution_hours"] / 24).where(d["closed"] & ~d["censored"])
    c, p = d["closed"].astype(int).astype(str), d["prio_label"].notna().astype(int).astype(str)
    if is_k8s:
        tgt = np.where(d["dup_targets"].str.len() + d["fix_peers"].str.len() > 0, "dup_or_peer",
                       np.where(d["fix_prs"].str.len() > 0, "fix_pr", "none"))  # fmt: skip
        d["stratum"] = "prio=" + p + "|target=" + tgt + "|closed=" + c
    else:
        dd = (d["dup_targets"].str.len() > 0).astype(int).astype(str)
        d["stratum"] = (
            "in=" + d["in_family"].astype(int).astype(str) + "|dup=" + dd + "|closed=" + c
        )
    removed["population"] = len(d)
    return d.reset_index(drop=True), {"removed": removed, "classes": classes}


def to_row(r: pd.Series, repo: str, w: float) -> dict[str, Any]:
    """One eval_set.jsonl row in the committed schema, plus a `meta` block."""
    days = None if pd.isna(r["days"]) else float(r["days"])
    if r["prio_label"] is not None and not pd.isna(r["prio_label"]):
        prio, src = r["prio_label"], "label"
    elif days is not None:
        prio, src = infer_priority(days), "resolution_inferred"
    else:
        prio, src = None, "none"
    age = (r["scraped_at_max"] - r["created_at"]).total_seconds() / 86400
    return {
        "id": f"{REPOS[repo]['id']}-{int(r['number'])}", "repo": repo, "number": int(r["number"]),
        "title": r["title"], "body": r["body"], "gold_component": r["cls"], "gold_priority": prio,
        "actual_resolution_days": days, "created_at": r["created_at"].isoformat(),
        "meta": {
            "stratum": r["stratum"], "weight": w, "primary_label": r["primary"],
            "parent_area": r["parent"], "in_family": bool(r["in_family"]), "state": r["state"],
            "censored": days is None, "age_days_at_scrape": None if days is not None else age,
            "gold_priority_source": src, "priority_label4": gold4(r["labels_final"]),
            "dup_targets": r["dup_targets"], "fix_peers": r["fix_peers"], "fix_prs": r["fix_prs"],
        },
    }  # fmt: skip


def filter_neighbours(
    hits: list[dict[str, Any]], q_num: int, q_created: pd.Timestamp,
    created: dict[int, pd.Timestamp], k: int = K,
) -> tuple[list[dict[str, Any]], dict[str, int]]:  # fmt: skip
    """First k hits that existed when the query was created (see v2 design, point 3)."""
    st = {"examined": 0, "dropped_newer": 0, "kept_by_number_fallback": 0, "dropped_unknown": 0}
    kept: list[dict[str, Any]] = []
    for h in hits:
        if len(kept) >= k:
            break
        st["examined"] += 1
        t = created.get(h["number"])
        if t is not None:
            ok = t < q_created
            st["dropped_newer"] += not ok
        else:
            ok = h["number"] < q_num  # unknown time: provably earlier only by the shared numbering
            st["kept_by_number_fallback"] += ok
            st["dropped_unknown"] += not ok
        if ok:
            kept.append(h)
    return kept, st


def created_lookup(repo: str) -> dict[int, pd.Timestamp]:
    """number -> created_at from the served train parquet and derived_v2 (index meta has none)."""
    out: dict[int, pd.Timestamp] = {}
    proc = ROOT / "data" / "processed" / f"{repo.replace('/', '_')}_temporal_train.parquet"
    v2 = _read(REPOS[repo]["slug"], "issues", columns=["number", "created_at"])
    for d in (pd.read_parquet(proc, columns=["number", "created_at"]), v2):
        out.update(zip(d["number"].astype(int), pd.to_datetime(d["created_at"], utc=True),
                       strict=True))  # fmt: skip
    return out


def freeze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach served-index top-K similar_issues (CPU float32); write the k8s query embeddings."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported: hardware-independent prompt
    from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: PLC0415
    from triage_iq.models.triage import RESOLUTION_EMBEDDINGS_REPOS  # noqa: PLC0415

    dets: dict[str, Any] = {}
    created: dict[str, dict[int, pd.Timestamp]] = {}
    v2: dict[str, dict[int, pd.Timestamp]] = {}
    emb: dict[str, list[tuple[int, np.ndarray]]] = {}
    info: dict[str, Any] = {"index_sha256": {}, "neighbour_filter": {}}
    for r in rows:
        repo, n = r["repo"], r["number"]
        if repo not in dets:
            idx = ROOT / "data" / "models" / f"similar_issue_index_{repo.replace('/', '_')}_bge"
            dets[repo] = SimilarIssueRetriever.load(str(idx))
            info["index_sha256"][repo] = sha256_file(idx / "index.faiss")
            created[repo] = created_lookup(repo)
            d2 = _read(REPOS[repo]["slug"], "issues", columns=["number", "created_at"])
            v2[repo] = dict(zip(d2["number"].astype(int), d2["created_at"], strict=True))
            info["neighbour_filter"][repo] = {"examined": 0, "dropped_newer": 0,
                "kept_by_number_fallback": 0, "dropped_unknown": 0, "rows_with_fewer_than_k": 0}  # fmt: skip
        hits, e = dets[repo].retrieve_with_embedding(f"{r['title']}. {r['body']}", k=K_RETRIEVE,
                                                     exclude_number=n)  # fmt: skip
        q_created = pd.Timestamp(r["created_at"])
        r["similar_issues"], st = filter_neighbours(hits, n, q_created, created[repo])
        acc = info["neighbour_filter"][repo]
        for key, v in st.items():
            acc[key] += v
        acc["rows_with_fewer_than_k"] += len(r["similar_issues"]) < K
        if repo in RESOLUTION_EMBEDDINGS_REPOS:
            emb.setdefault(repo, []).append((n, e.astype(np.float32)))
        have = set(map(int, dets[repo].issue_numbers))
        m = r["meta"]
        m["dup_targets_in_index"] = [t for t in m["dup_targets"] if t in have]
        m["fix_peers_in_index"] = [t for t in m["fix_peers"] if t in have]
        # targets a new retriever could index: present in derived_v2 and older than the query
        m["indexable_targets"] = [t for t in m["dup_targets"] + m["fix_peers"]
                                  if t in v2[repo] and v2[repo][t] < q_created]  # fmt: skip
    if emb:
        repos = [r for r, v in emb.items() for _ in v]
        nums = [n for v in emb.values() for n, _ in v]
        mat = np.stack([e for v in emb.values() for _, e in v])
        np.savez(OUT_DIR / "frozen_query_embeddings.npz", repos=np.array(repos),
                 numbers=np.array(nums, dtype=np.int64), embeddings=mat)  # fmt: skip
    return info


def dump_line(row: dict[str, Any]) -> str:
    """One JSON line that str.splitlines() cannot split: U+2028/2029/0085 stay escaped.

    Some eval scripts read the set with read_text().splitlines(), which breaks on those raw
    characters (U+2028 occurs in issue bodies); file iteration does not.
    """
    out = json.dumps(row, ensure_ascii=False)
    for code in (0x2028, 0x2029, 0x85):
        out = out.replace(chr(code), chr(92) + "u" + format(code, "04x"))
    return out


def text_chars(r: dict[str, Any]) -> int:
    """Variable prompt text actually sent: title + body[:800] + 120 chars per similar issue."""
    sims = sum(min(len(s["text"]), PROMPT_SIM_CHARS) for s in r["similar_issues"][:K])
    return len(r["title"]) + min(len(r["body"]), PROMPT_BODY_CHARS) + sims


def token_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Planning-figure budget, adjusted by the prompt text this set sends vs the committed set."""
    chars = np.array([text_chars(r) for r in rows])
    text = (ROOT / "eval" / "eval_set.jsonl").read_text("utf-8")
    ref_chars = np.array([text_chars(json.loads(x)) for x in text.split("\n") if x])
    delta = float((chars.mean() - ref_chars.mean()) / 4)  # chars/4 ~ tokens
    per_issue = TOKENS_PER_ISSUE + delta
    total = int(round(per_issue * len(rows)))
    return {"n": len(rows), "planning_tokens_per_issue": TOKENS_PER_ISSUE,
            "planning_total_tokens": TOKENS_PER_ISSUE * len(rows),
            "sent_text_tokens_chars_div4": {"mean": float(chars.mean() / 4),
                                            "p95": float(np.percentile(chars, 95) / 4),
                                            "max": float(chars.max() / 4)},
            "committed_set_sent_text_mean_tokens": float(ref_chars.mean() / 4),
            "delta_tokens_per_issue_vs_committed": delta,
            "synthesis_prompt_tokens_est": CASSETTE_PROMPT_CHARS / 4 + delta,
            "adjusted_tokens_per_issue": per_issue, "adjusted_total_tokens": total,
            "daily_cap": DAILY_CAP, "days_at_cap": math.ceil(total / DAILY_CAP)}  # fmt: skip


def git_head_of_script() -> str | None:
    """Last commit that touched THIS script (qp0_make_splits.script_commit names its own file)."""
    cmd = ["git", "log", "-1", "--format=%H", "--", "scripts/qp3_build_eval_set.py"]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)  # noqa: S603
    except (subprocess.CalledProcessError, OSError):
        return None
    return r.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    rows: list[dict[str, Any]] = []
    rep: dict[str, Any] = {"design": DESIGN, "design_version": 2, "seed": SEED,
                           "script_commit": git_head_of_script(),
                           "supersedes_v1": {"eval_set.jsonl": V1_SET_SHA256,
                                             "frozen_query_embeddings.npz": V1_NPZ_SHA256}}  # fmt: skip
    for repo, cfg in REPOS.items():
        pop, info = load_population(repo)
        sample = draw(pop, N_PER_REPO, cfg["minimum"], balance=cfg["balance"])
        tabs = strata_tables(pop, sample)
        mine = [
            to_row(r, repo, (weight(tabs["stratum_x_class"], r["stratum"], r["cls"])
                             if cfg["balance"] else stratum_weight(tabs["strata"], r["stratum"])))
            for _, r in sample.iterrows()
        ]  # fmt: skip
        rows += mine
        rep[repo] = {**info, **tabs, "minimum": cfg["minimum"], "balance": cfg["balance"],
                     "in_family_share_pop": pop["in_family"].mean(),
                     "in_family_share_sample": sample["in_family"].mean(), "n": len(sample),
                     "classes_in_sample": int(sample["cls"].nunique()),
                     "classes_in_population": int(pop["cls"].nunique()),
                     "weights": weight_stats([r["meta"]["weight"] for r in mine]),
                     "input_sha256": {"test_split": sha256_file(
                         DATA / "splits" / cfg["slug"] / "test.parquet")}}  # fmt: skip
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep["freeze"] = freeze(rows)
    path = OUT_DIR / "eval_set.jsonl"
    path.write_text("".join(dump_line(r) + "\n" for r in rows), "utf-8", newline="\n")
    rep["files"] = {p.name: sha256_file(p) for p in sorted(OUT_DIR.iterdir())}
    rep["tokens"] = token_report(rows)
    rep["relevance_scoring"] = {
        "status": "UNSCORABLE on the served index: 0 sampled duplicate/fix targets are indexed",
        "action": "freeze the Phase 2 / 3b retriever into this set (re-run freeze with it) "
                  "before the single re-record",
        "per_repo": {
            repo: {
                "with_dup_or_peer": sum(bool(r["meta"]["dup_targets"] or r["meta"]["fix_peers"])
                                        for r in rows if r["repo"] == repo),
                "target_in_served_index": sum(bool(r["meta"]["dup_targets_in_index"]
                                                   or r["meta"]["fix_peers_in_index"])
                                              for r in rows if r["repo"] == repo),
                "indexable_in_derived_v2_before_query": sum(
                    bool(r["meta"]["indexable_targets"]) for r in rows if r["repo"] == repo),
            }
            for repo in REPOS
        },
    }  # fmt: skip
    REPORT.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str), "utf-8")
    keys = ("tokens", "relevance_scoring", "files")
    print(
        json.dumps(
            {k: rep[k] for k in keys} | {"freeze": rep["freeze"]["neighbour_filter"]}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
