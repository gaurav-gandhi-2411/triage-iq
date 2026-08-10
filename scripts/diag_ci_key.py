from __future__ import annotations
"""Diagnostic: compute synthesis key for vscode-2093 and print prompt excerpt."""
import sys, json
sys.path.insert(0, "src")
sys.path.insert(0, "eval")
import numpy as np, pandas as pd, sklearn, torch
print(f"numpy={np.__version__} sklearn={sklearn.__version__} torch={torch.__version__}", flush=True)
from triage_iq.models.component_classifier import load_classifier
from triage_iq.models.similar_issues import SimilarIssueRetriever
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant
from triage_iq.cache.llm_cache import LLMCache
from cassette import CassettePlayer
issues = [json.loads(l) for l in open("eval/eval_set.jsonl").read().splitlines() if l.strip()]
issue = issues[0]
clf = load_classifier("data/models", "microsoft_vscode")
det = SimilarIssueRetriever.load("data/models/similar_issue_index_microsoft_vscode_bge")
pred = ResolutionTimePredictor.load("data/models/resolution_predictor_microsoft_vscode.pkl")
train = pd.read_parquet("data/processed/microsoft_vscode_temporal_train.parquet")
# Patch compute_key to capture first call
orig = LLMCache.compute_key
captured = []
def patched(*a, **kw):
    key = orig(*a, **kw)
    if not captured:
        msgs = a[2] if len(a) > 2 else kw.get("messages", [])
        captured.append((key, msgs[0]["content"] if msgs else ""))
    return key
LLMCache.compute_key = staticmethod(patched)
cassette = CassettePlayer("eval/cassettes/eval_cassette.json", strict=False)
asst = TriageAssistant(repo="microsoft/vscode", classifier=clf, detector=det,
    predictor=pred, train_df=train, groq_api_key="noop", cache=cassette)
row = pd.Series({"title": issue["title"], "body_clean": issue["body"],
    "number": issue["number"],
    "created_at": pd.Timestamp(issue["created_at"]) if issue.get("created_at") else pd.Timestamp("now", tz="UTC")})
try: asst.triage_with_metadata(row)
except: pass
if captured:
    key, prompt = captured[0]
    print(f"SYNTHESIS_KEY={key}", flush=True)
    print(f"PROMPT_LINES_600_900:", flush=True)
    print(prompt[600:900], flush=True)
else:
    print("NO KEY CAPTURED", flush=True)
