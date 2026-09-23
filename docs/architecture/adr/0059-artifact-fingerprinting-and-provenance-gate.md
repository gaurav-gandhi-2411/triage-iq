# ADR-0059 — Model-artifact fingerprinting and the cassette provenance gate

Status: Accepted
Date: 2026-09-23 (implemented 2026-09-23 in `db282d9`; incident 2026-09-05)
Decider: Gaurav Gandhi

## Context

`eval/record_cassettes.py` resolves its model directory relative to whichever checkout's copy
of the script executes (`ROOT = Path(__file__).parent.parent`). On 2026-09-05 a "full 64-issue
re-record" was run against the main repo checkout (`C:\Users\gaura\ml-projects\triage-iq`)
instead of the bake-off worktree. That checkout still held the **pre-retrain** component
classifier, so every recorded prompt embedded the OLD classifier's `classifier_top3`, while this
worktree held the ADR-0057 retrained 47-class classifier.

Nothing noticed. The recording checkpoint was keyed by `(issue_id, model, prompt_hash)`
(ADR-0054/0055), and a classifier retrain changes neither the model name nor the
prompt/schema text — it changes what the prompt *says* for each issue, via the classifier
signals block. The mismatch surfaced only later as an unexplained `CassetteMissError` on replay
from the worktree. The baseline written from that recording (ADR-0052's 2026-09-05 table,
judge mean 12.0938) therefore measured a configuration that never shipped — see the
2026-09-23 correction appended to ADR-0052.

Generalisation: any artifact that changes prompt content (classifier, resolution predictor,
temporal train parquet used for feature engineering, similar-issue FAISS index + metadata,
conformal adjustment store) is part of the recording's identity, exactly as the model name and
prompt text are.

## Decision

1. **Fingerprint every prompt-affecting artifact.** `eval/artifact_fingerprint.py` SHA-256s the
   per-repo classifier, resolution predictor, temporal-train parquet, BGE FAISS index and
   `meta.pkl`, plus the shared `cqr_conformal_adjustments.json`. `combined_hash()` reduces them
   to one 16-hex identifier (`artifact_hash`).
2. **Committed expectation, fail closed.** `eval/cassettes/EXPECTED_ARTIFACT_HASHES.json` is the
   explicit, human-reviewed statement of which artifacts this branch intends to record against.
   `record_cassettes.py` prints the resolved absolute paths + hashes at startup and exits unless
   they match. A **missing** expectation file is a refusal, never "no expectation, proceed"
   (rule 98a). Regenerating it is a deliberate act, never automatic reconciliation.
3. **Wrong-checkout tripwire.** A cheap, specific check exits if `ROOT` resolves to the known
   other checkout — defence in depth for the exact incident shape on top of the general gate.
4. **Checkpoint key extended** to `(issue_id, model, prompt_hash, artifact_hash)`. An entry
   missing any tag halts the run (cannot be proven to belong to the current config); entries
   under a different config are ignored for resume, not deleted.
5. **Per-entry cassette provenance.** `CassettePlayer.set_provenance()` stamps each synthesis
   entry with the triaged repo's own artifact hashes at write time. Per-entry, not per-file,
   because a resumable multi-day campaign can span an artifact change. The cassette is what
   ships and replays in CI, so it must be checkable without the checkpoint file.
6. **Invariant.** `eval/test_invariants.py::test_cassette_provenance_matches_current_artifacts`
   fails on any synthesis entry that is unstamped or whose stamped hashes don't match the
   artifacts currently on disk. Judge entries carry `judge_provenance` inherited from their
   parent synthesis entry (ADR-0060 §B3), checked by the same test.

## Consequences

- The 2026-09-05 incident class is now caught at the first line of a recording run, not days
  later at replay. Verified RED before the fix (256/256 stale entries unstamped) and against a
  scratch negative control with a genuinely mismatched hash (not only an absent one).
- Any intentional retrain now requires three deliberate steps before recording: retrain,
  regenerate `EXPECTED_ARTIFACT_HASHES.json`, re-record. That friction is the point.
- `EXPECTED_ARTIFACT_HASHES.json` is distinct from `data/models/MANIFEST.sha256`: the manifest
  tracks what is published to GCS for serving (`test_model_manifest_clean`, still failing until
  the retrained classifier is published); the expectation file tracks what a recording is
  allowed to be conditioned on. They must agree once the publish happens; nothing enforces that
  yet.
- Did not fix: `ROOT = Path(__file__).parent.parent` itself. Any checkout can still run the
  script; it just can no longer do so silently against the wrong artifacts.

## Alternatives considered

- **Fold artifact identity into `prompt_hash`.** Rejected: `prompt_hash` covers prompt text and
  wire schema, which are static; artifact identity is data, and a per-issue prompt differs for
  every issue anyway. Separate tags keep "what changed" diagnosable.
- **Per-file (cassette-level) provenance only.** Rejected: a campaign spanning an artifact
  change would carry one stamp for mixed content.
- **Resolve `ROOT` from an env var / refuse unless run from the worktree.** Narrower — fixes the
  checkout-confusion instance but not an in-place artifact change in the right checkout.
