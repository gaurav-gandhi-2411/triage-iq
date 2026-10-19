# ADR-0066 — Retrieval targets for the quality push: vscode duplicates, k8s fix-linked

Status: Accepted
Date: 2026-10-20
Decider: Gaurav Gandhi (decision of 2026-10-20 on spec-quality-push.md section 7.1)
Numbered 0066 because 0065 is reserved by the open Cloud Run recording-job PR (#184).

## Context

The quality-push spec sets one retrieval target: duplicate Recall@5 >= 80% per repo. The fresh-data
scrape and the first derivation passes (scripts/qp0_derive_*.py, 2023 kubernetes slice, 2,317
issues on disk at the time) show how few duplicate pairs kubernetes has: 27 pairs from comments and
labels, and 1 `MarkedAsDuplicateEvent` in 2,317 issues. Extrapolated to the 7.7k k8s issues that is a
few dozen pairs in total, far too few for a measured target with a usable interval. vscode has the
`marked_as_duplicate` events and a bot that closes duplicates, so its duplicate set is expected to be
orders of magnitude larger (not yet measured: vscode is still being scraped).

## Decision

1. **vscode duplicate Recall@5 >= 80% stays the target** (spec section 7, unchanged).
2. **k8s duplicate retrieval is reported, not targeted.** Report the pair count, Recall@1/5/10 and MRR
   with their intervals, and say plainly that n is small.
3. **The k8s retrieval headline is fix-linked retrieval** (spec section 7.1: issues fixed by the same
   merged PR), **subject to fix-link precision**: fix links may be used for training or evaluation only
   when an independent audit against the GitHub API (>= 100 links, stratified by repo and year,
   Wilson interval) shows >= 95% precision; otherwise the extraction is fixed first.
4. The old k8s "related" task stays retired as a headline (spec section 7.1).

## Consequences

- No pass/fail claim is made about k8s duplicate retrieval; its numbers are descriptive.
- The k8s headline depends on the fix-link audit; if the audit fails or the same-PR groups are too
  few, the k8s retrieval result is reported as the best achievable number with the evidence, per the
  spec's rule for infeasible targets.
- Targets are not edited after seeing results; this ADR records a scope decision taken before any
  retrieval model is scored on the fresh test split.

## Alternatives

- Keep a k8s duplicate target: rejected, n too small for an interval that supports a claim.
- Pool the two repos for one duplicate number: rejected, it would hide that the k8s sample is tiny and
  mix two label cultures.
