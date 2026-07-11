"""T3: Connected-component temporal split for W3 fine-tuning.

Builds a pair graph on (query_number, original_number) within each repo, finds
connected components, and assigns entire components to train/val/test at ~70/15/15
proportion. Components are ordered chronologically by the earliest created_at of
any issue in the component so that test always contains newer pairs.

Cross-repo components (16 found in initial analysis): edges that link a vscode
issue number to a k8s issue number arise from coincidental number collisions and
represent data artefacts, not true relatedness. These edges are dropped; the
resulting intra-repo sub-components are treated as independent.

Outputs:
  data/w3_split.parquet
    columns: repo, query_number, original_number, component_id, split
             (split ∈ {"train", "val", "test"})
  data/w3_split_stats.json — split size report

Verification: asserts zero (repo, query_number) and zero (repo, original_number)
overlap across splits.
"""

from __future__ import annotations

import json
import logging

import networkx as nx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets the remainder


def load_created_at(repo: str) -> dict[int, pd.Timestamp]:
    """Return {issue_number: created_at} for a repo."""
    df = pd.read_parquet(
        f"data/processed/issues_{repo}.parquet",
        columns=["number", "created_at"],
    )
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    return dict(zip(df["number"].astype(int), df["created_at"], strict=True))


def component_min_date(
    comp: frozenset[int],
    created_at: dict[int, pd.Timestamp],
    fallback: pd.Timestamp,
) -> pd.Timestamp:
    dates = [created_at.get(n) for n in comp if created_at.get(n) is not None]
    return min(dates) if dates else fallback


def split_repo(
    repo: str,
    gold: pd.DataFrame,
) -> pd.DataFrame:
    repo_gold = gold[gold["repo"] == repo].copy()
    created_at = load_created_at(repo)
    fallback_date = pd.Timestamp("2099-01-01", tz="UTC")

    # Build intra-repo pair graph; pairs where both issue numbers belong to this
    # repo's namespace only (cross-repo edges already removed before calling here)
    G = nx.Graph()
    for _, row in repo_gold.iterrows():
        q, o = int(row["query_number"]), int(row["original_number"])
        G.add_edge(q, o)

    components = list(nx.connected_components(G))
    logger.info("[%s] %d components from %d pairs", repo, len(components), len(repo_gold))

    # Sort components chronologically by earliest issue created_at
    components.sort(key=lambda c: component_min_date(c, created_at, fallback_date))

    # Count total pairs per component (a component contributes all gold rows that
    # have any issue in it — query or original)
    comp_nodes: list[frozenset[int]] = [frozenset(c) for c in components]
    comp_id_map: dict[int, int] = {}
    for cid, nodes in enumerate(comp_nodes):
        for n in nodes:
            comp_id_map[n] = cid

    # Assign pairs to component ids
    repo_gold = repo_gold.copy()
    repo_gold["component_id"] = repo_gold["query_number"].astype(int).map(comp_id_map)

    # Count pairs per component (sorted chronologically by cid)
    pairs_per_comp = (
        repo_gold.groupby("component_id").size().reindex(range(len(comp_nodes)), fill_value=0)
    )
    total_pairs = pairs_per_comp.sum()
    train_target = int(TRAIN_FRAC * total_pairs)
    val_target = int(VAL_FRAC * total_pairs)

    # Greedy assignment using a state machine: assign component to current state,
    # then transition after cumulative crosses each threshold. This guarantees val
    # gets at least one component even when a large component overshoots 70%.
    split_by_comp: dict[int, str] = {}
    cumulative = 0
    state = "train"
    for cid in range(len(comp_nodes)):
        n = pairs_per_comp[cid]
        split_by_comp[cid] = state
        cumulative += n
        if state == "train" and cumulative >= train_target:
            state = "val"
        elif state == "val" and cumulative >= train_target + val_target:
            state = "test"

    repo_gold["split"] = repo_gold["component_id"].map(split_by_comp)

    counts = repo_gold["split"].value_counts().to_dict()
    logger.info(
        "[%s] Split: train=%d  val=%d  test=%d",
        repo,
        counts.get("train", 0),
        counts.get("val", 0),
        counts.get("test", 0),
    )
    return repo_gold


def verify_no_leakage(result: pd.DataFrame) -> None:
    """Assert that no issue appears in multiple splits (per repo)."""
    for repo in result["repo"].unique():
        sub = result[result["repo"] == repo]
        for col in ("query_number", "original_number"):
            issue_splits = sub.groupby(col.replace("_number", "_number"))["split"].apply(set)
            # Check using groupby col directly
            issue_splits = sub.groupby(col)["split"].apply(set)
            leaked = issue_splits[issue_splits.map(len) > 1]
            if len(leaked) > 0:
                logger.error(
                    "[%s] LEAKAGE in %s: %d issues appear in multiple splits!",
                    repo,
                    col,
                    len(leaked),
                )
                logger.error("Leaked issues: %s", leaked.index.tolist()[:10])
                raise RuntimeError(f"Split leakage detected in {repo}/{col}")
    logger.info("Leakage check PASSED — no issue appears in multiple splits")


def main() -> None:
    gold = pd.read_parquet("data/gold_related_v2.parquet")
    logger.info("Gold pairs loaded: %d", len(gold))

    # Detect and drop cross-repo edges (issue numbers that appear in both repos'
    # query/original columns indicate coincidental number collisions, not real links)
    k8s_numbers = set(
        gold.loc[gold["repo"] == "kubernetes_kubernetes", "query_number"].astype(int)
    ) | set(gold.loc[gold["repo"] == "kubernetes_kubernetes", "original_number"].astype(int))
    vsc_numbers = set(
        gold.loc[gold["repo"] == "microsoft_vscode", "query_number"].astype(int)
    ) | set(gold.loc[gold["repo"] == "microsoft_vscode", "original_number"].astype(int))
    cross_repo_numbers = k8s_numbers & vsc_numbers
    logger.info("Cross-repo number collisions: %d unique issue numbers", len(cross_repo_numbers))

    # Flag cross-repo rows (any row where query or original number appears in both repos)
    is_cross_repo = gold["query_number"].astype(int).isin(cross_repo_numbers) | gold[
        "original_number"
    ].astype(int).isin(cross_repo_numbers)
    n_cross = is_cross_repo.sum()
    logger.info(
        "Dropping %d cross-repo-contaminated pairs (%.1f%%)", n_cross, 100 * n_cross / len(gold)
    )
    gold_clean = gold[~is_cross_repo].copy()
    logger.info("Clean gold pairs after cross-repo removal: %d", len(gold_clean))

    dfs = [split_repo(repo, gold_clean) for repo in ["kubernetes_kubernetes", "microsoft_vscode"]]
    result = pd.concat(dfs, ignore_index=True)

    verify_no_leakage(result)

    out = "data/w3_split_v2.parquet"
    result.to_parquet(out, index=False)
    logger.info("Saved split → %s", out)

    # Stats
    stats: dict = {}
    for repo in result["repo"].unique():
        sub = result[result["repo"] == repo]
        counts = sub["split"].value_counts().to_dict()
        n_comps = sub["component_id"].nunique()
        stats[repo] = {
            "total_pairs": len(sub),
            "train": counts.get("train", 0),
            "val": counts.get("val", 0),
            "test": counts.get("test", 0),
            "n_components": n_comps,
        }
        logger.info(
            "[%s] total=%d  train=%d  val=%d  test=%d  n_components=%d",
            repo,
            len(sub),
            counts.get("train", 0),
            counts.get("val", 0),
            counts.get("test", 0),
            n_comps,
        )

    stats_path = "data/w3_split_v2_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats → %s", stats_path)


if __name__ == "__main__":
    main()
