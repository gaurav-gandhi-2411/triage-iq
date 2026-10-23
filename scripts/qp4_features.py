"""Phase 4 creation-time features for the kubernetes resolution-time task (see qp4_resolution.py).

Every feature of an issue created at t_q uses only information that existed at t_q: the issue's own
title / body / author_association, and OTHER issues that were created before t_q (an earlier issue
counts as resolved only if its closed_at <= t_q). Nothing about labels applied later, comments or
the closing event of the row itself is read. The planted-future tests in tests/test_qp4_resolution.py
pin this for author history, open-issue load and the kNN features.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp1_tfidf_lr as t1  # noqa: E402

SEED, K_NN, N_PCA, WEEK_H = 42, 20, 16, 168.0
ASSOC = {"NONE": 0, "CONTRIBUTOR": 1, "MEMBER": 2}
# Template type from the first matching body header of the kubernetes issue forms; 0 = no template.
TEMPLATES = [
    (1, "### What happened?"), (2, "### What would you like to be added"),
    (3, "### Which jobs are failing"), (4, "### Which tests are flaking"),
]  # fmt: skip
TRACE_RE = re.compile(r"Traceback|goroutine \d+|panic:|\.go:\d+|\bat [\w.$]+\(.*\)")
URL_RE = re.compile(r"https?://\S+")


def author_history(df: pd.DataFrame) -> pd.DataFrame:
    """Per row: earlier issues by the same author (n, n closed by t_q, median resolution hours).

    df needs author, created_at, closed_at (NaT = open), resolution_hours. Only issues created
    strictly before t_q count; a resolution counts only if closed_at <= t_q.
    """
    out = np.full((len(df), 3), np.nan)
    out[:, :2] = 0
    for _, g in df[df["author"].notna()].groupby("author"):
        idx, cr = g.index.to_numpy(), g["created_at"].to_numpy()
        cl, rh = g["closed_at"].to_numpy(), g["resolution_hours"].to_numpy()
        for a, t in zip(idx, cr, strict=True):
            prior = cr < t
            done = prior & (cl <= t)  # NaT compares False: an open issue never counts as done
            out[a, 0], out[a, 1] = prior.sum(), done.sum()
            if done.any():
                out[a, 2] = np.median(rh[done])
    return pd.DataFrame(out, columns=["a_n_prior", "a_n_done", "a_med_res_h"], index=df.index)


def open_load(created: np.ndarray, closed: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """Issues created before t_q and not closed by t_q, in the row's own predicted component."""
    out = np.zeros(len(created))
    for c in np.unique(comp):
        m = comp == c
        cr, cl = np.sort(created[m]), np.sort(closed[m][~pd.isna(closed[m])])
        out[m] = np.searchsorted(cr, created[m], "left") - np.searchsorted(cl, created[m], "right")
    return out


def knn_features(emb: np.ndarray, created: np.ndarray, closed: np.ndarray, res_h: np.ndarray,
                 ref: np.ndarray, k: int = K_NN) -> np.ndarray:  # fmt: skip
    """Median log1p resolution and share closed within 7 days of the k nearest reference issues.

    Reference = rows with ref True that are closed at or before the query's created_at (so they
    are earlier than it and the row itself can never be its own neighbour). NaN if none exist.
    """
    ref_i = np.flatnonzero(ref & ~pd.isna(closed))
    sims, out = emb @ emb[ref_i].T, np.full((len(emb), 3), np.nan)
    for i in range(len(emb)):
        ok = closed[ref_i] <= created[i]
        if not ok.any():
            continue
        s = np.where(ok, sims[i], -np.inf)
        top = np.argpartition(-s, min(k, ok.sum()) - 1)[: min(k, ok.sum())]
        r = res_h[ref_i[top]]
        out[i] = np.median(np.log1p(r)), (r <= WEEK_H).mean(), len(top)
    return out


def text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Own-text and calendar features known at creation."""
    body, title = df["body"].fillna(""), df["title"].fillna("")
    tmpl = np.zeros(len(df), dtype=int)
    for code, hdr in reversed(TEMPLATES):  # first listed header wins
        tmpl[body.str.contains(hdr, regex=False).to_numpy()] = code
    return pd.DataFrame({
        "assoc": df["author_assoc"].map(ASSOC).fillna(-1).to_numpy(),
        "title_len": title.str.len().to_numpy(), "body_len": body.str.len().to_numpy(),
        "has_code": body.str.contains("```", regex=False).astype(int).to_numpy(),
        "has_trace": body.map(lambda s: int(bool(TRACE_RE.search(s)))).to_numpy(),
        "n_urls": body.map(lambda s: len(URL_RE.findall(s))).to_numpy(),
        "hour": df["created_at"].dt.hour.to_numpy(), "dow": df["created_at"].dt.dayofweek.to_numpy(),
        "template": tmpl,
    }, index=df.index)  # fmt: skip


def component_probs(root: Path, slug: str, nums: pd.DataFrame, text: pd.Series, cache: Path):  # noqa: ANN201
    """(probs [n, C], classes): out-of-fold for train rows, candidate 1 refit for validation/test.

    Candidate 1 = TF-IDF + multinomial LR, C=64 balanced (qp1 grid.json). Train rows: 5-fold
    (seed 42) over the labelled train rows; labelled rows get their held-out fold's prediction, the
    train rows without a sig/* label (never fitted on) get the mean of the 5 fold models. Validation
    and test rows (all of them, labelled or not) come from one refit on all labelled train rows; the
    saved qp1 validation probabilities are asserted equal to the refit's.
    """
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["p"], list(z["classes"])
    d = t1.load_split(root / "derived" / slug, root / "splits" / slug, ("train", "validation"))
    modelled, classes = t1.define_classes(d[d["split"] == "train"])
    tr = d[(d["split"] == "train") & d["classifier_eligible"]].reset_index(drop=True)
    ytr = np.array(t1.gold(tr, modelled)[0])
    pos = pd.Series(np.arange(len(nums)), index=nums["number"].to_numpy())
    p, unl = np.zeros((len(nums), len(classes))), (nums["split"] == "train").to_numpy().copy()
    unl[pos[tr["number"]].to_numpy()] = False
    from sklearn.feature_extraction.text import TfidfVectorizer

    def fit(idx: np.ndarray):  # noqa: ANN202
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        m = t1.fit_lr(
            v.fit_transform(text.iloc[pos[tr["number"].iloc[idx]]]), list(ytr[idx]), 64, "balanced"
        )
        return lambda rows: t1.proba_full(m, v.transform(text.iloc[rows]), classes)  # fmt: skip

    for a, b in KFold(5, shuffle=True, random_state=SEED).split(tr):
        pred = fit(a)
        p[pos[tr["number"].iloc[b]].to_numpy()] = pred(pos[tr["number"].iloc[b]].to_numpy())
        p[unl] += pred(np.flatnonzero(unl)) / 5
    pred = fit(np.arange(len(tr)))
    later = np.flatnonzero((nums["split"] != "train").to_numpy())
    p[later] = pred(later)
    saved = pd.read_parquet(root / "models" / slug / "tfidf_lr" / "validation.parquet")
    rows = pos[saved["number"]].to_numpy()
    assert np.allclose(p[rows], saved[classes].to_numpy(), atol=1e-4), "refit != saved qp1 probs"
    np.savez(cache, p=p, classes=np.array(classes))
    return p, classes


def build(root: Path, slug: str, cache: Path) -> pd.DataFrame:
    """Feature table (one row per split issue, eligible or not) with split and survival targets."""
    sp = pd.concat(
        [
            pd.read_parquet(
                root / "splits" / slug / f"{s}.parquet",
                columns=["number", "split", "classifier_eligible"],
            )
            for s in ("train", "validation", "test")
        ]  # fmt: skip
    )
    iss = pd.read_parquet(root / "derived_v2" / slug / "issues.parquet")
    emb = np.load(root / "embeddings" / slug / f"{slug}__issues_emb.npy").astype(np.float32)
    ids = pd.read_parquet(root / "embeddings" / slug / f"{slug}__issues_ids.parquet")["number"]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idpos = pd.Series(np.arange(len(ids)), index=ids.to_numpy())
    for c in ("created_at", "closed_at"):  # naive UTC so numpy datetime64 comparisons work
        iss[c] = iss[c].dt.tz_localize(None)
    iss = iss.merge(sp, on="number").sort_values("created_at").reset_index(drop=True)
    cutoff = iss["scraped_at_max"].iloc[0].tz_localize(None)
    emb = emb[idpos[iss["number"]].to_numpy()]
    legacy = pd.read_parquet(
        root / "derived" / slug / "issues.parquet", columns=["number", "title", "body"]
    )
    ltext = legacy.assign(t=legacy["title"].fillna("") + "\n" + legacy["body"].fillna("").str.slice(0, 4000)).set_index("number")["t"]  # fmt: skip
    p, classes = component_probs(
        root, slug, iss[["number", "split"]], iss["number"].map(ltext), cache
    )
    top = p.argmax(1)
    cr, cl = iss["created_at"].to_numpy(), iss["closed_at"].to_numpy()
    tr = (iss["split"] == "train").to_numpy()
    pca = PCA(N_PCA, random_state=SEED).fit(emb[tr])
    f = pd.concat([
        text_features(iss), author_history(iss),
        pd.DataFrame(p, columns=[f"p_{c}" for c in classes]),
        pd.DataFrame({"comp_top1": top, "comp_maxp": p.max(1), "load_comp": open_load(cr, cl, top),
                      "load_all": open_load(cr, cl, np.zeros(len(iss), dtype=int))}),
        pd.DataFrame(knn_features(emb, cr, cl, iss["resolution_hours"].to_numpy(), tr), columns=["knn_med_log", "knn_share7", "knn_n"]),  # fmt: skip
        pd.DataFrame(pca.transform(emb), columns=[f"pca{i}" for i in range(N_PCA)]),
    ], axis=1)  # fmt: skip
    hours = np.where(iss["censored"], (cutoff - iss["created_at"]).dt.total_seconds() / 3600, iss["resolution_hours"])  # fmt: skip
    meta = iss[["number", "split", "classifier_eligible", "created_at"]].assign(
        T=hours,
        event=~iss["censored"].to_numpy(),
        follow_h=(cutoff - iss["created_at"]).dt.total_seconds() / 3600,
    )
    return pd.concat([meta, f], axis=1)
