# Eval infrastructure

## What's here

| File | Purpose |
|------|---------|
| `eval_set.jsonl` | 64-issue frozen eval set (11 vscode + 53 k8s), stratified by resolution bucket |
| `cassette.py` | JSON-backed cassette player for offline LLM replay in CI |
| `cassettes/eval_cassette.json` | Recorded LLM interactions (synthesis + judge) for replay |
| `test_invariants.py` | 7 deterministic structural invariants (no LLM) — runs in CI, informational |
| `test_fabrication_gate.py` | Deterministic grounding ratchet (no LLM) — **HARD, BLOCKING** (ADR-0029) |
| `test_quality_regression.py` | Cassette-replayed judge vs baseline — runs in CI, informational |
| `record_cassettes.py` | One-time recording script (run locally with live creds; never in CI) |
| `run_eval.py` | Compute current scores from cassettes; used to refresh baseline |

## Running the structural suite locally

```bash
pytest eval/test_invariants.py -v --no-cov
```

Requires production models in `data/models/` and `data/processed/`. Download via:

```bash
gsutil -m cp "gs://triageiq-portfolio-495022-models/models/..." data/models/
```

## How to intentionally change behavior (baseline + cassette update procedure)

When you make an intentional change that alters LLM interactions (prompt edit, model swap,
retrieval change), the gate will fail with `CassetteMissError` — the recorded request no
longer matches what the code sends. This is correct behavior.

To update:

1. Make your code change.
2. Re-record cassettes locally with live Groq creds:
   ```bash
   python eval/record_cassettes.py
   ```
   This uses the checkpoint; pass `--fresh` to start over.

3. Compute new scores and update the baseline:
   ```bash
   python eval/run_eval.py --update-baseline
   ```

4. Review `reports/eval_baseline.json` — confirm per-repo mean scores look sane and the
   delta vs the previous baseline is intentional.

5. Commit `eval/cassettes/eval_cassette.json`, `reports/eval_baseline.json`, and your
   code change in the same PR. Include the score delta in the PR description.

Baseline updates must be human-reviewed — there is no auto-update automation. A PR that
drops the quality score below the threshold must explain why (deliberate tradeoff, different
model tier, etc.).

## CI behavior

`structural-invariants` and `quality-regression` are **non-blocking**
(`continue-on-error: true`). They will be promoted to required status checks after one
confirmed green cycle on `main`.

`fabrication-gate` (ADR-0029) is **hard-blocking** (no `continue-on-error`) — a
fabricated component or similar-issue claim beyond the pinned, measured baseline in
`eval/test_fabrication_gate.py` fails the workflow. It was promoted directly to
blocking, skipping the informational stage the other two jobs are still in, because it
gates a specific, precisely-measured, near-zero-rate correctness failure mode (see
ADR-0029), not because the general promotion criterion above changed.

Every job runs zero live LLM calls. A cassette miss = hard fail (the request changed
without a cassette update). Fix: re-record + commit the new cassette.

## Known limitation

The gate verifies code against recorded interactions, not against the live API. It catches
unintended code changes; it does not detect model drift or provider-side changes. See
ADR-0011 for the full tradeoff discussion.
