from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp3_eval_set_lib as q  # noqa: E402


def make_pop(n: int = 400) -> pd.DataFrame:
    """Synthetic population: 4 strata of very different sizes, 6 classes with a rare one."""
    rng = np.random.default_rng(0)
    strata = rng.choice(["a", "b", "c", "d"], size=n, p=[0.6, 0.25, 0.1, 0.05])
    cls = rng.choice(["x", "y", "z", "w", "v", "rare"], size=n, p=[0.3, 0.3, 0.2, 0.1, 0.07, 0.03])
    return pd.DataFrame({"number": np.arange(1000, 1000 + n), "stratum": strata, "cls": cls})


def test_allocate_sums_and_respects_minimum_and_capacity() -> None:
    pop = {"a": 600, "b": 250, "c": 100, "d": 2}
    out = q.allocate(pop, 75, 5)
    assert sum(out.values()) == 75
    assert out["d"] == 2  # stratum smaller than the minimum is taken whole
    assert all(5 <= out[k] <= pop[k] for k in ("a", "b", "c"))
    assert out["a"] > out["b"] > out["c"]  # remainder is proportional


def test_allocate_is_proportional_when_minimum_is_inactive() -> None:
    out = q.allocate({"a": 800, "b": 200}, 50, 0)
    assert out == {"a": 40, "b": 10}


def test_allocate_rejects_impossible_minimums() -> None:
    with pytest.raises(ValueError):
        q.allocate({"a": 50, "b": 50, "c": 50}, 10, 5)


def test_draw_is_deterministic_and_seed_sensitive() -> None:
    pop = make_pop()
    a = q.draw(pop, 40, 3, seed=42)
    b = q.draw(pop.sample(frac=1, random_state=1), 40, 3, seed=42)  # row order must not matter
    assert a["number"].tolist() == b["number"].tolist()
    assert a["number"].tolist() != q.draw(pop, 40, 3, seed=7)["number"].tolist()


def test_draw_respects_strata_quotas_and_membership() -> None:
    pop = make_pop()
    s = q.draw(pop, 40, 3)
    quota = q.allocate(pop.groupby("stratum").size().to_dict(), 40, 3)
    assert s.groupby("stratum").size().to_dict() == quota
    assert s["number"].is_unique and set(s["number"]) <= set(pop["number"])
    assert len(s) == 40


def test_draw_spreads_over_classes_including_rare() -> None:
    pop = make_pop()
    s = q.draw(pop, 40, 3)
    assert set(s["cls"]) == set(pop["cls"])  # greedy least-drawn-class picks reach every class


def test_strata_tables_and_weights_recover_population() -> None:
    pop = make_pop()
    s = q.draw(pop, 40, 3)
    t = q.strata_tables(pop, s)
    assert sum(v["population"] for v in t["strata"].values()) == len(pop)
    assert sum(v["sample"] for v in t["strata"].values()) == 40
    covered = sum(n for n, k in t["stratum_x_class"].values() if k)
    assert covered + t["undrawn_population"] == len(pop)
    total = sum(q.weight(t["stratum_x_class"], r.stratum, r.cls) for r in s.itertuples())
    assert total == pytest.approx(covered)  # weights re-expand the sample to the covered cells


def test_dump_line_survives_splitlines_and_round_trips() -> None:
    row = {"body": "a" + chr(0x2028) + "b" + chr(0x2029) + "c" + chr(0x85) + "d"}
    line = q.dump_line(row)
    assert len(line.splitlines()) == 1 and json.loads(line) == row


def test_filter_neighbours_drops_newer_and_counts() -> None:
    t = pd.Timestamp
    created = {
        1: t("2020-01-01", tz="UTC"),
        2: t("2026-06-01", tz="UTC"),
        3: t("2021-01-01", tz="UTC"),
    }
    hits = [
        {"number": n} for n in (2, 1, 9, 50, 3)
    ]  # 9 unknown and above the query; 50 unknown below
    kept, st = q.filter_neighbours(hits, 40, t("2026-01-05", tz="UTC"), created, k=3)
    assert [h["number"] for h in kept] == [1, 9, 3]
    assert st["dropped_newer"] == 1 and st["dropped_unknown"] == 1
    assert st["kept_by_number_fallback"] == 1
    kept, st = q.filter_neighbours([{"number": 7}], 40, t("2026-01-05", tz="UTC"), {}, k=5)
    assert len(kept) == 1 and st["kept_by_number_fallback"] == 1


def test_vscode_style_srs_is_deterministic_and_order_invariant() -> None:
    pop = make_pop()
    a = q.draw(pop, 40, 3, seed=42, balance=False)
    b = q.draw(pop.sample(frac=1, random_state=3), 40, 3, seed=42, balance=False)
    assert a["number"].tolist() == b["number"].tolist() and len(a) == 40
    quota = q.allocate(pop.groupby("stratum").size().to_dict(), 40, 3)
    assert a.groupby("stratum").size().to_dict() == quota
    assert a["number"].tolist() != q.draw(pop, 40, 3, seed=7, balance=False)["number"].tolist()


def test_weight_stats_kish() -> None:
    s = q.weight_stats([1.0, 1.0, 1.0, 1.0])
    assert s["kish_n_eff"] == pytest.approx(4.0)
    s = q.weight_stats([1.0, 3.0])
    assert s["kish_n_eff"] == pytest.approx(16 / 10) and s["max"] == 3.0


def test_target_maps_and_created_lookup(tmp_path: Path) -> None:
    dp = pd.DataFrame({"source": [5, 5, 6], "canonical": [1, 2, 3]})
    fl = pd.DataFrame(
        {
            "issue": [5, 6, 5],
            "pr": [90, 91, 92],
            "pr_merged": [True, False, True],
            "method": ["closed_event_closer_pr"] * 2 + ["pr_closing_refs"],  # 92: unaudited
        }
    )
    fg = pd.DataFrame({"pr": [90], "issues": [[5, 7, 8]]})
    dup, peers, fixpr = q.target_maps(dp, fl, fg)
    assert dup == {5: [1, 2], 6: [3]} and peers[7] == [5, 8] and fixpr.to_dict() == {5: [90]}
    train = tmp_path / "t.parquet"
    pd.DataFrame({"number": [1], "created_at": [pd.Timestamp("2020-01-01", tz="UTC")]}).to_parquet(
        train
    )
    v2 = pd.DataFrame({"number": [2], "created_at": [pd.Timestamp("2024-01-01", tz="UTC")]})
    assert set(q.created_lookup(train, v2)) == {1, 2}


def test_token_report_adjusts_by_sent_text_delta() -> None:
    def row(body: int) -> dict:
        return {"title": "t", "body": "x" * body, "similar_issues": [{"text": "y" * 500}] * 5}

    caps = (800, 120)  # a longer body than the cap adds nothing; 120 chars per hit
    same = q.token_report([row(900)], [row(800)], 6000, 160_000, 16_000, caps)
    assert (
        same["delta_tokens_per_issue_vs_committed"] == 0 and same["adjusted_total_tokens"] == 6000
    )
    more = q.token_report([row(800)] * 40, [row(400)], 6000, 160_000, 16_000, caps)
    assert more["delta_tokens_per_issue_vs_committed"] == 100 and more["days_at_cap"] == 2
