"""Reusable helpers of the Q3 eval-set builder (scripts/qp3_build_eval_set.py): stratified
allocation and drawing, weights with Kish n_eff, creation-time-aware neighbour filtering and the
splitlines-safe JSONL writer. Behaviour-preserving extraction; the design lives in the builder."""

from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qp0_derive_stats import audited_fix_links

SEED, K = 42, 5  # K: similar issues kept per row


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


def created_lookup(train_parquet: Path, v2_issues: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """number -> created_at from the served train parquet and derived_v2 (index meta has none)."""
    out: dict[int, pd.Timestamp] = {}
    for d in (pd.read_parquet(train_parquet, columns=["number", "created_at"]), v2_issues):
        out.update(
            zip(d["number"].astype(int), pd.to_datetime(d["created_at"], utc=True), strict=True)
        )
    return out


def dump_line(row: dict[str, Any]) -> str:
    """One JSON line that str.splitlines() cannot split: U+2028/2029/0085 stay escaped.

    Some eval scripts read the set with read_text().splitlines(), which breaks on those raw
    characters (U+2028 occurs in issue bodies); file iteration does not.
    """
    out = json.dumps(row, ensure_ascii=False)
    for code in (0x2028, 0x2029, 0x85):
        out = out.replace(chr(code), chr(92) + "u" + format(code, "04x"))
    return out


def target_maps(
    dp: pd.DataFrame, fl: pd.DataFrame, fg: pd.DataFrame
) -> tuple[dict[int, list[int]], dict[int, list[int]], pd.Series]:
    """(duplicate canonicals by source, co-closed fix-group peers, merged fix PRs by issue)."""
    dup = dp.groupby("source")["canonical"].apply(lambda s: sorted(set(map(int, s)))).to_dict()
    fixpr = audited_fix_links(fl).groupby("issue")["pr"].apply(lambda s: sorted(set(map(int, s))))
    peers: dict[int, list[int]] = {}
    for grp in fg["issues"]:
        for n in grp:
            peers[int(n)] = sorted(int(m) for m in grp if m != n)
    return dup, peers, fixpr


def git_last_commit(cwd: Path, rel_path: str) -> str | None:
    """Last commit that touched `rel_path` (None outside git)."""
    cmd = ["git", "log", "-1", "--format=%H", "--", rel_path]
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)  # noqa: S603
    except (subprocess.CalledProcessError, OSError):
        return None
    return r.stdout.strip() or None


def text_chars(r: dict[str, Any], caps: tuple[int, int]) -> int:
    """Variable prompt text actually sent: title + body[:caps[0]] + caps[1] chars per hit."""
    sims = sum(min(len(s["text"]), caps[1]) for s in r["similar_issues"][:K])
    return len(r["title"]) + min(len(r["body"]), caps[0]) + sims


def token_report(
    rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]], per_issue_plan: int,
    daily_cap: int, cassette_chars: int, caps: tuple[int, int],
) -> dict[str, Any]:  # fmt: skip
    """Planning-figure budget, adjusted by the prompt text this set sends vs the committed set."""
    chars = np.array([text_chars(r, caps) for r in rows])
    ref_chars = np.array([text_chars(r, caps) for r in ref_rows])
    delta = float((chars.mean() - ref_chars.mean()) / 4)  # chars/4 ~ tokens
    per_issue = per_issue_plan + delta
    total = int(round(per_issue * len(rows)))
    return {"n": len(rows), "planning_tokens_per_issue": per_issue_plan,
            "planning_total_tokens": per_issue_plan * len(rows),
            "sent_text_tokens_chars_div4": {"mean": float(chars.mean() / 4),
                                            "p95": float(np.percentile(chars, 95) / 4),
                                            "max": float(chars.max() / 4)},
            "committed_set_sent_text_mean_tokens": float(ref_chars.mean() / 4),
            "delta_tokens_per_issue_vs_committed": delta,
            "synthesis_prompt_tokens_est": cassette_chars / 4 + delta,
            "adjusted_tokens_per_issue": per_issue, "adjusted_total_tokens": total,
            "daily_cap": daily_cap, "days_at_cap": math.ceil(total / daily_cap)}  # fmt: skip
