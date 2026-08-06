# ADR-0040 — Corpus token-truncation fix + per-repo BGE query instruction

**Status:** Accepted — code + measurement + candidate-index verification done; GCS publish and
deploy are separate, explicitly-gated next steps, not covered by this ADR's acceptance
**Date:** 2026-08-06
**Decider:** Gaurav Gandhi

## Context

Every prior retrieval-improvement attempt on this project (ADR-0031's three levers, the D2
fine-tune of ADR-0034/0035) was measured against a corpus with a real, previously undiagnosed
bug: `similar_issues.py::_build_text()` truncated the CORPUS side of the index to 512
**characters** before encoding, while `BAAI/bge-base-en-v1.5`'s actual sequence limit is 512
**tokens** (`model.max_seq_length`). Query-side truncation was already fixed to use full,
untruncated text (ADR-0035) — the corpus side was flagged in `_retrieval_eval_common.py`'s own
comment as "disclosed, not fixed."

Measured first, before touching any code (`reports/lever1_truncation_measurement.json`, BGE's
own tokenizer, no char/4 approximation):

| Repo | Issues truncated today | Mean content lost among truncated | Median tokens dropped |
|---|---|---|---|
| kubernetes/kubernetes | 17.8% | 36.7% | 78 |
| microsoft/vscode | 43.2% | 47.9% | 165 |

Separately, BGE-v1.5's model card documents a required query-side instruction prefix
("Represent this sentence for searching relevant passages: ") for retrieval use — never applied
in this codebase.

## Decision

**Lever 1 (corpus truncation): fix for both repos, unconditionally.** `_build_text()` now
truncates by real token count via the model's own tokenizer (`AutoTokenizer.encode()`,
reserving room for the title and `[CLS]`/`[SEP]`), not a fixed character cut. This is a strict
improvement with no plausible downside — it only recovers content the model could always use but
was never shown.

**Lever 2 (query instruction): per-repo, not uniform.** Re-measured against D1's frozen,
hand-verified eval sets, paired bootstrap CI, same methodology as `_retrieval_eval_common.py`
(`reports/lever12_eval_results.json`):

| Eval set (n) | Baseline R@5 | +Lever1 | +Lever1+2 | Lever2-only delta (vs Lever1) |
|---|---|---|---|---|
| k8s_related (150, gate) | 18.00% | 21.33% (Δ+3.33pp CI[+0.67,+6.0], excl. 0) | **24.67%** (Δ+6.67pp CI[+2.67,+10.67], excl. 0) | **+3.33pp CI[0.0,+6.67]** |
| vscode_duplicate (200, gate) | 50.50% | 53.50% (Δ+3.00pp CI[-1.0,+7.0], crosses 0) | 51.50% (Δ+1.00pp CI[-3.5,+5.0], crosses 0) | **-2.00pp CI[-5.0,+1.0]** |
| vscode_related (19, directional, never gated) | 63.16% | 57.89% | 47.37% | -10.53pp — too underpowered to read |

k8s: the instruction is a real, additive, statistically significant win on top of Lever 1.
vscode_duplicate: the instruction's isolated effect is directionally **negative** and it erases
most of Lever 1's own (non-significant) positive trend (53.5%→51.5%).

**The instruction ships ON for k8s, OFF for vscode** (`QUERY_INSTRUCTION_REPO_OVERRIDE` in
`similar_issues.py`), not uniformly on or off. Rejected the "ship uniformly, vscode's CI crosses
zero anyway" framing: a CI crossing zero means the *evidence* isn't strong enough to prove harm
at this sample size, not that the direction is a coin flip — the point estimate is consistently
negative across both the isolated Lever2-only delta and the vscode_related directional set, and
it cancels a gain we'd otherwise keep. Shipping the uniform default for code simplicity would be
accepting a known-direction cost to avoid a small, one-time config addition.

**Working hypothesis for the asymmetry, explicitly not confirmed:** vscode's D1 eval task is
near-duplicate matching (near-identical issue detection), where exact lexical/surface overlap is
plausibly the dominant relevance signal; BGE's "searching relevant passages" framing is tuned for
topical/semantic retrieval and may dilute that lexical signal. k8s's task is genuine
issue-to-issue semantic relatedness (not duplicates), where the framing fits the task BGE's
instruction was designed for. **If a future eval shows the instruction helps vscode, flip
`QUERY_INSTRUCTION_REPO_OVERRIDE["microsoft_vscode"]` to `True`** — this record exists
specifically so that flip isn't blocked by "but we decided OFF" inertia, and so a future session
doesn't quietly "simplify" the config back to a uniform default without re-deriving why it was
split in the first place.

## Consequences

- **What changes:** `_build_text()` gains a `tokenizer`/`max_tokens` path (the char-based path
  stays as an explicit legacy/no-tokenizer fallback). `retrieve()`/`retrieve_batch()` gain an
  `apply_query_instruction` parameter (`None` = resolve via repo override → model default;
  explicit `True`/`False` = eval-only A/B isolation, never used by prod code).
- **What becomes easier:** any future per-repo retrieval config decision has a precedent and a
  slot (`QUERY_INSTRUCTION_REPO_OVERRIDE`) instead of needing a new mechanism.
- **What becomes harder:** nothing structurally; one more per-repo dict to keep in sync when a
  new repo is added (mitigated by falling back to the model default when a repo isn't listed).
- **No production change from the code merge alone.** Lever 1 has zero effect on the currently
  served `dup_index_*_bge` artifacts until they're explicitly rebuilt and re-published — `load()`
  reads pre-built FAISS files, it doesn't call `build_index()`. Lever 2 (query-side) DOES take
  effect on the next deploy once merged, independent of any index rebuild, since it's a runtime
  code path — this is expected and fine: BGE's instruction is asymmetric by design (queries only,
  regardless of how the corpus was built), so it isn't a "mismatch" with an unrebuilt corpus.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Ship the query instruction uniformly for both repos | vscode's isolated delta is consistently negative in direction (not just noisy); shipping it anyway trades a known-direction cost for config simplicity. |
| Hold Lever 2 entirely, pending more data on both repos | k8s's result is already statistically significant (CI excludes zero) — holding a proven win on one repo to avoid a config asymmetry is the wrong trade. |
| A generic per-(repo, model) config table instead of a repo-only override | No current need — `model_key` is already effectively fixed per repo in this project (bge is the shipped retriever everywhere; minilm is evaluation-only, ADR-0006). Repo-only override matches the actual decision surface; extend later if that changes. |

---

## UPDATE (2026-08-06) — candidate served-index rebuilt + cutover verification PASSED

Built `data/models/dup_index_{repo}_bge_candidate/` from the current full corpus
(`data/processed/issues_{repo}.parquet`, the same source `scripts/08_build_similar_issue_index.py`
uses for the real served index) with Lever 1's tokenizer-based `_build_text()` already live in
the code. Two checks (`scripts/lever12_verify_candidate.py`,
`reports/lever12_candidate_verification.json`), both required before any publish:

**1. Reproduction via the REAL, un-overridden code path** (`retrieve()` called with no
`apply_query_instruction` argument, exactly like `triage.py` does in production — the per-repo
default in `QUERY_INSTRUCTION_REPO_OVERRIDE` resolves on its own, not forced by a test):

| Eval set | Instruction applied (per repo default) | R@5 | Matches manual-override measurement above? |
|---|---|---|---|
| k8s_related (n=150) | True (k8s) | 24.67% [18.0, 31.3] | Yes — exact match to the Lever1+2 row |
| vscode_duplicate (n=200) | False (vscode) | 53.50% [46.5, 60.5] | Yes — exact match to the Lever1-only row |
| vscode_related (n=19) | False (vscode) | 57.89% [36.8, 78.9] | Yes — exact match to the Lever1-only row |

Confirms the per-repo config isn't just correct in isolated unit tests — it produces the exact
intended numbers when exercised through the same code path a live `/triage` request uses.

**2. Index/query construction consistency** — re-derived `_build_text()` on a random 200-issue
sample per repo (fixed seed) using the loaded candidate index's own tokenizer/`max_seq_length`,
and asserted byte-identical output against what's actually stored in the index's `meta.pkl`
`texts`. **200/200 matched for both repos, zero mismatches.** This is the check GG asked for
explicitly: proof, not assumption, that the corpus text embedded in the candidate index and what
current code would produce today are the same thing.

**Status: candidate index verified, ready to publish. NOT yet published to GCS, NOT yet
deployed.** Remaining steps, each requiring GG's explicit go per the standing escalation policy:

1. Replace `data/models/dup_index_{repo}_bge/` with the verified `_candidate/` contents (both
   repos).
2. `python scripts/publish_models.py` — uploads to `gs://triageiq-models/models/dup_index_*/`,
   updates `MANIFEST.sha256` (project-identity-gated, per that script's own hard-stop check).
3. Commit the updated `MANIFEST.sha256`, merge this branch to `main` — `deploy.yml` triggers,
   downloads the just-published index fresh from GCS, builds and deploys with this ADR's
   per-repo query-instruction code already in the same commit (code and index move together,
   by construction of merging one branch that contains both).
4. Live-verify post-deploy: real `/triage` calls for both repos, confirm `similar_issues` output
   changes in the expected direction (not just a 200 response) — same live-verification
   discipline as the ADR-0036 classifier cutover and ADR-0038 migration.
5. Rollback anchor: `deploy.yml`'s existing smoke-test-triggered auto-rollback applies
   unchanged; no new rollback mechanism needed for this cutover specifically.
