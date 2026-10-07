from __future__ import annotations

"""Runs inside a tiq-sk:<ver> container. Dumps every served-path output for the prod artifacts
to /out/equiv_<ver>.npz so a second container's dump can be compared bit-for-bit."""

import json
import sys
import warnings

import faiss
import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, "/work/src")
sys.path.insert(0, "/work")
from triage_iq.models.component_classifier import _build_text, load_classifier  # noqa: E402
from triage_iq.models.resolution import ResolutionTimePredictor, engineer_features  # noqa: E402

warnings.filterwarnings("ignore")
V = sklearn.__version__
SPL = "/splits/processed"  # regenerated splits (classifier_{train,val,test}, temporal_*)
MODELS = "/work/data/models"
REPOS = {"microsoft_vscode": "microsoft/vscode", "kubernetes_kubernetes": "kubernetes/kubernetes"}
out: dict[str, np.ndarray] = {}

gold = pd.read_parquet("/work/data/gold_triage_plans.parquet")
evalset = pd.DataFrame([json.loads(line) for line in open("/work/eval/eval_set.jsonl")])

for slug, repo in REPOS.items():
    # ---------------- classifier ----------------
    clf = load_classifier(MODELS, slug)
    texts: dict[str, pd.Series] = {}
    for sp in ("train", "val", "test"):
        d = pd.read_parquet(f"{SPL}/{slug}_classifier_{sp}.parquet")
        texts[f"cls_{sp}"] = _build_text(d["title"], d["body_clean"])
        out[f"{slug}|cls_{sp}|y"] = d["component"].astype(str).to_numpy()
    g = gold[gold["repo"].str.replace("/", "_") == slug]
    texts["gold119"] = g["title"].astype(str) + ". " + g["body_clean"].astype(str)
    e = evalset[evalset["repo"] == repo]
    texts["eval64"] = e["title"].astype(str) + ". " + e["body"].astype(str)
    for name, X in texts.items():
        raw = clf.predict_proba(X)
        cal = clf.predict_proba_calibrated(X)
        out[f"{slug}|{name}|raw"] = raw
        out[f"{slug}|{name}|cal"] = cal
        out[f"{slug}|{name}|label"] = np.asarray(clf.predict(X)).astype(str)
        out[f"{slug}|{name}|classes"] = np.asarray(clf.classes_()).astype(str)
    # ---------------- resolution predictor ----------------
    pr = ResolutionTimePredictor.load(f"{MODELS}/resolution_predictor_{slug}.pkl")
    imp = pr.feature_importance("gain")
    emb_share = float(imp[[c for c in imp.index if c.startswith("emb_")]].sum() / imp.sum())
    print(f"[{V}] {slug}: n_features={len(pr.feature_names)} emb_* share of point-model gain={emb_share:.4f}")
    # training-path: PCA.transform on real BGE embeddings (index.faiss) for all temporal rows
    meta = joblib.load(f"{MODELS}/similar_issue_index_{slug}_bge/meta.pkl")
    index = faiss.read_index(f"{MODELS}/similar_issue_index_{slug}_bge/index.faiss")
    n2i = {int(n): i for i, n in enumerate(meta["issue_numbers"])}
    train = pd.read_parquet(f"/work/data/processed/{slug}_temporal_train.parquet")  # GCS prod copy
    train = train[train["resolution_hours"] > 0]
    for sp in ("gcsprod_train", "regen_train", "regen_val", "regen_test"):
        d = pd.read_parquet(
            f"/work/data/processed/{slug}_temporal_train.parquet" if sp == "gcsprod_train"
            else f"{SPL}/{slug}_temporal_{sp.split('_')[1]}.parquet")
        d = d[d["resolution_hours"] > 0]
        embs = np.zeros((len(d), index.d), dtype=np.float32)
        for pos, num in enumerate(d["number"]):
            j = n2i.get(int(num))
            if j is not None:
                embs[pos] = index.reconstruct(int(j))
        feats, _ = engineer_features(d, train_df=train, embeddings=embs, pca=pr.pca)
        feats = feats[pr.feature_names]
        k = f"{slug}|res_{sp}"
        out[f"{k}|pca"] = pr.pca.transform(embs)
        out[f"{k}|feats"] = feats.to_numpy(dtype=np.float64)
        out[f"{k}|point_log"] = pr.predict_log(feats)
        out[f"{k}|point_hrs"] = pr.predict(feats)
        lo, hi = pr.predict_intervals(feats)
        out[f"{k}|lo"], out[f"{k}|hi"] = lo, hi
        out[f"{k}|bucket_proba"] = np.asarray(pr.model_bucket.predict(feats))
        out[f"{k}|bucket_label"] = np.asarray(pr.predict_bucket(feats)[0]).astype(str)
        out[f"{k}|y"] = d["resolution_hours"].to_numpy(dtype=np.float64)
    # served-path (triage.py:836-860): engineer_features WITHOUT embeddings, emb_* zero-filled
    sg = g.rename(columns={"gold_component": "component", "gold_priority": "priority"})
    rows = []
    for _, r in sg.iterrows():
        idf = pd.DataFrame([r])
        feats, _ = engineer_features(idf, train_df=train)
        for c in pr.feature_names:
            if c not in feats.columns:
                feats[c] = 0.0
        rows.append(feats[pr.feature_names])
    sf = pd.concat(rows)
    k = f"{slug}|res_served_gold"
    out[f"{k}|point_hrs"] = pr.predict(sf)
    lo, hi = pr.predict_intervals(sf)
    out[f"{k}|lo"], out[f"{k}|hi"] = lo, hi
    out[f"{k}|bucket_proba"] = np.asarray(pr.model_bucket.predict(sf))
    out[f"{k}|bucket_label"] = np.asarray(pr.predict_bucket(sf)[0]).astype(str)

np.savez(f"/out/equiv_{V}.npz", **out)
print(f"[{V}] wrote {len(out)} arrays")
