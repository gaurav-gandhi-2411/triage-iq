# Pre-registration: vscode index v2 latency and memory on Cloud Run (supersedes criterion 5)

Written and merged before any measurement. Approved in principle by GG's advisor on 2026-11-02 (GG instruction, item 1).
Earlier prereg: `docs/QP_VSCODE_INDEX_V2_PREREG_2026-11.md`. Earlier results: `docs/QP_VSCODE_INDEX_V2_RESULTS_2026-11.md`.

## Why criterion 5 is superseded

Criterion 5 of the earlier prereg (vscode retrieval p50 <= 300 ms on the laptop) was applied as written: v2 measured 611.6 ms,
the gate failed, and there was no cutover (decision recorded 2026-11-01). The same laptop run measured the **current served
index** at 585.3 ms and a dated copy of it at 609.6 ms. A criterion that also rejects the configuration already in production
does not measure the difference that matters, which is what v2 adds over served on the production platform. It is replaced by
the comparison below. Criterion 4b (Cloud Run memory < 85%), never measured, is closed by gate M below.

Not blind: the laptop numbers above are known. v2's laptop retrieval p50 was within 2 ms of the dated served copy.

## Configurations

| Name | vscode index | Dated | Notes |
|---|---|---|---|
| served (control) | `similar_issue_index_microsoft_vscode_served_dated` | yes | the served vectors minus 2 undated issues (#5123, #29238); index sha256 5000aaaa..., meta 47dd068f... |
| v2 (treatment) | `similar_issue_index_microsoft_vscode_bge_v2_dated` | yes | 109,537 issues; index sha256 b5be4bb1..., meta ae52dcf5... |

Both use the created-before filter. Everything else is production: the same image, the five production flags, 4 GiB, 2 vCPU,
concurrency 8, region us-central1.

## Prerequisites (none changes production behaviour)

1. **Loader flag** `TRIAGE_VSCODE_INDEX`. Unset keeps today's undated served index. `served_dated` and `v2` load the dirs above
   and fail closed if absent, like `TRIAGE_K8S_INDEX=v2`. Merged under the normal gates; production never sets it.
2. **Artifacts** uploaded to the models bucket with `gcloud storage cp --no-clobber`, plus MANIFEST entries for the drift guard.
3. **Image.** The deploy workflow's download step gains the two dirs. This is a workflow change, so it is a draft PR for GG to
   merge. The next main deploy then carries both dirs (about +0.5 GB of image) while production keeps its env, so serving is
   unchanged. If v2 is not adopted, a follow-up draft PR removes them.

## Candidate revisions

Each candidate is deployed from that image with `--no-traffic` and a tag, with production env plus:

| Tag | Extra env | Stage |
|---|---|---|
| `vsidx-served` | `TRIAGE_VSCODE_INDEX=served_dated`, `GROQ_BASE_URL=http://127.0.0.1:9` | A |
| `vsidx-v2` | `TRIAGE_VSCODE_INDEX=v2`, `GROQ_BASE_URL=http://127.0.0.1:9` | A |
| `vsidx-v2-llm` | `TRIAGE_VSCODE_INDEX=v2` | B |

`GROQ_BASE_URL` points the Groq client at a closed local port. Every LLM call then fails at once and takes the production
provider-outage path (`degraded_provider_error`), after the classifier, retrieval and resolution steps have run and been timed.
Stage A therefore spends 0 Groq tokens.

**Template safety.** `gcloud run deploy --no-traffic` also changes the service template, and the CI deploy keeps template env
it does not set. No merge to main is made while a measurement is in progress. Each measurement ends by restoring the template
env to the live revision's env, verified with `gcloud run services describe`, before any later merge. Traffic stays 100% on
the live revision throughout; tags are removed after the run.

## Request set (fixed now)

The 75 vscode issues of `eval/sets/quality_push_2026-10` (set rows 76-150), in set order, each sent with its own title, body,
issue number and `created_at`, so the filter is active. Requests run one at a time.

## Stage A: retrieval latency and memory (0 Groq tokens)

- Warm-up: the first 5 issues to each revision, discarded. Both revisions then hold one warm instance.
- Timed: for i = 1..75, issue i to `vsidx-served`, then issue i to `vsidx-v2`, alternating so that drift affects both alike.
- Retrieval time is `system2_latency_ms` from each request's log entry, joined on the request ID. It covers encode, search with
  the filter, and resolution evidence (none for vscode).
- Validity: every timed request must return HTTP 200 with `llm_status = degraded_provider_error`. Any other status, or a missing
  log entry, makes the run invalid; it is re-run once in full, and a second invalid run is reported and decided by GG.

**Gate R:** p50(v2) <= p50(served) + 100 ms over the 75 timed requests each. Reported: p95 of both, and the paired median
difference with a bootstrap 95% CI (seed 42, 10,000 resamples).

## Stage B: end-to-end latency on v2 (Groq, booked in the ledger)

- Revision `vsidx-v2-llm`, real Groq. Warm-up: issue 31 (discarded). Timed: issues 1-30 in order.
- End-to-end is `total_latency_ms` from the request log.
- Every call goes through `tools/groq_budget/groq_budget.py wait`/`record` under project `triageiq-latency`. The 40,000-token
  production reserve is never allocated. Estimate: 31 calls x about 7,000 tokens = about 220,000, more than one day's 160,000
  allocatable, so stage B may span two budget windows. It runs after the k8s recording finishes.
- A timed request with `llm_status` other than `ok` is reported and replaced by the next unused issue (32, 33, ...).

**Gate E:** p50 of `total_latency_ms` over 30 timed `ok` requests <= 6,000 ms (`spec-quality-push.md`: end-to-end warm p50 <= 6 s).
p95 reported. Reported for context, not gated: the LLM share of each request (`system4_latency_ms`).

Known risk, stated before measuring: most of the end-to-end time is the Groq call, which does not depend on the index, and
earlier production requests took 16 s or more when rate-limited (D73). Gate E can therefore fail for a reason the served
configuration shares, which is the flaw that retired criterion 5. If E fails while R and M pass, the result is still a fail
under this prereg (no cutover), and the report says whether the LLM share explains it; any re-test is GG's decision.

## Memory

**Gate M:** Cloud Monitoring `run.googleapis.com/container/memory/utilizations` for `vsidx-v2` and `vsidx-v2-llm`, the maximum of
the 99th-percentile series over each stage's window, is < 0.85. The served candidate's value is reported.

## Also reported, not gated

Cold start (deploy to first healthy `/health?deps=1`) per candidate; the image size before and after the workflow change.

## Decision

All of R, E and M pass: cut over vscode v2 (GG item 2), via candidate, smoke, comparison, promote and live check, with the
rollback written first; then re-freeze the vscode rows on v2 with the filter, re-run the leakage check, and record vscode on v2.
Any gate fails: no cutover; record vscode on the served-index re-freeze already prepared
(`similar_issue_index_microsoft_vscode_served_dated`). The vscode recording starts only after this decision.

The known trade-off is reported either way: on the 65 queries both indexes answer, v2 hits 30 against served's 45; its overall
gain is coverage of older originals that the served index lacks.
