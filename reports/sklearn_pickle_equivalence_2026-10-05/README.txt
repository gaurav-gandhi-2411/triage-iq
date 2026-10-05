Provenance for the scikit-learn 1.6.1 vs 1.7.2 pickle-equivalence measurement (2026-10-05).
Base: origin/main 15ef0b3b14ccedc2f9ff3bc09b1227df421d16be. Artifacts: the production GCS bytes
(sha256 in inspect_pickles_under_*.txt, equal to data/models/MANIFEST.sha256).
Environments: two python:3.11-slim containers installing requirements.lock verbatim, differing only in
scikit-learn (1.6.1 vs 1.7.2); numpy 2.4.4, pandas 3.0.2, lightgbm 4.6.0, scipy 1.17.1, joblib 1.5.3,
torch 2.11.0 identical in both.
Populations: classifier_{train,val,test}.parquet regenerated from data/raw via scripts/02_preprocess.py +
scripts/03_split.py (seed 42; test n=423 vscode / 671 k8s, matching ADR-0057), eval/eval_set.jsonl,
data/gold_triage_plans.parquet, temporal splits (GCS prod temporal_train + regenerated train/val/test).
Run: equiv.py inside each container (writes npz), then compare.py; inspect_pickles.py for stored versions.
Note: regenerated vscode temporal_train has 9793 rows vs 4923 in the prod GCS copy (corpus grew since the
predictor was trained); the prod GCS copy is included as its own population (res_gcsprod_train).
