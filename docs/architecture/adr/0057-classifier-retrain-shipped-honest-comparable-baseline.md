# ADR-0057: Classifier retrain shipped (local); honest comparable-population re-measurement; downstream invalidation inventory

Status: Accepted
Date: 2026-09-04
Decider: Gaurav Gandhi — explicit "ship the retrain" decision, overriding ADR-0056's Phase 3
addendum recommendation ("reported and NOT recommended for shipping as-is").

## Context

ADR-0056 disclosed that the deployed component classifiers (28 vscode / 35 k8s classes) were
trained on a corpus snapshot roughly 2.3x smaller than what is currently in `data/raw/`, and
cannot structurally emit 9 labels present in the eval set's gold data. Its Phase 3 addendum
retrained a candidate classifier on the fresh, full corpus (47 classes both repos) and staged it
(`component_classifier_{repo}_multilabel_staged.pkl`) without promoting it, reporting what it
described as a genuine accuracy trade-off:

| Repo | Metric | Old (deployed) | New (staged) | Δ |
|---|---|---:|---:|---:|
| vscode | Top-3 | 89.84% [84.7, 93.4] | 85.82% [82.17, 88.82] | -4.02pp |
| k8s | Top-3 | 87.06% [82.7, 90.5] | 85.99% [83.16, 88.41] | -1.07pp |

against a gold-in-top-3 grounding gain (overall 62.5% → 93.75%). ADR-0056 flagged, but did not
resolve, that this was **not a paired comparison** — the old figures came from the old
classifier's own held-out test set (n=187 vscode / n=286 k8s, drawn from the stale, smaller
corpus snapshot); the new figures came from a much larger fresh test set (n=423 / n=671).

This session's explicit instruction: ship the retrain anyway. The stated reasoning was that the
4-point/1-point "regression" is not comparable to the 31-point gold-reachability gain, and that
gold-reachability is the metric that decides whether the product actually works. **That
reasoning turns out to understate the case** — see below.

## Decision

Ship the retrain. Executed as a **local artifact swap only** (`data/models/`):

- Archived the previously-deployed artifacts, unmodified, alongside the new ones:
  `component_classifier_microsoft_vscode_PRE_RETRAIN_2026-09-04.pkl` (sha256
  `25812ec99d13a54a270933ecb1bf3b0db73f490fec7ce6e9694a16681f40b20f`, matches the
  currently-committed `MANIFEST.sha256` entry — fully recoverable),
  `component_classifier_kubernetes_kubernetes_PRE_RETRAIN_2026-09-04.pkl` (sha256
  `9689243921ca7c558b0db0a6fa521e53cf0371a26cc646b85b24213473d1b31c`, same).
- Copied the staged artifacts onto the deployed path:
  `component_classifier_microsoft_vscode.pkl` (new sha256
  `5e286b1f75218373feceb43703d2a1a8a249d972393def659139eb308475313f`),
  `component_classifier_kubernetes_kubernetes.pkl` (new sha256
  `a7caf13b08e7e9cb3e7b8ae25413243345dd723a233556ca328641d606394f66`).
- Verified `load_classifier()` (`src/triage_iq/models/component_classifier.py:253`) dispatches
  the new artifacts correctly: both load as `MultiLabelTFIDFComponentClassifier`, 47 classes
  each.

**Explicitly NOT done, and why:** `MANIFEST.sha256` was not updated, and
`scripts/publish_models.py` was not run. `MANIFEST.sha256` is the committed source of truth CI's
drift guard (`scripts/verify_model_manifest.py`) checks against live GCS content — updating it
without also uploading the new bytes to GCS would make that guard fail the moment this branch
reaches `main` (manifest says one hash, GCS still holds the old one), and running the actual
upload is a live-production write gated to the working agreement's Phase 5 ("Do not deploy").
Promoting the manifest + running the GCS publish is deferred to that phase, alongside the actual
Cloud Run deploy.

## The comparable measurement (this ADR's central finding)

ADR-0056's -4.02pp/-1.07pp figures compare two different populations without saying so plainly
enough to prevent citing them as a regression. Built
`scripts/measure_old_classifier_on_fresh_taxonomy.py` (zero live calls, zero retraining — pure
local inference against the archived old-classifier pkls) to re-measure the **old** classifier
against the exact same fresh test rows (`data/processed/{repo}_classifier_test.parquet`, n=423 /
n=671) that scored the new classifier:

| Repo | Population | Old top-3 | New top-3 | Δ |
|---|---|---:|---:|---:|
| vscode | old's own stale test (n=187) — as previously cited, **not a fair comparator** | 89.84% [84.7, 93.4] | — | — |
| vscode | fresh shared test (n=423) — apples-to-apples | 68.09% [63.5, 72.4]* | 85.82% [82.17, 88.82] | **+17.73pp** |
| vscode | fresh test, in-old-taxonomy subset only (n=347) | 83.00%* | — | old classifier's own ceiling, excluding rows it could never answer |
| k8s | old's own stale test (n=286) — as previously cited, **not a fair comparator** | 87.06% [82.7, 90.5] | — | — |
| k8s | fresh shared test (n=671) — apples-to-apples | 83.61% [80.6, 86.2]* | 85.99% [83.16, 88.41] | **+2.38pp** |
| k8s | fresh test, in-old-taxonomy subset only (n=634) | 88.49%* | — | old classifier's own ceiling, excluding rows it could never answer |

\*Newly measured this session; `reports/old_classifier_on_fresh_taxonomy.json` has the full
output including CIs and exact ns.

**The previously reported "regression" does not survive an apples-to-apples measurement — on
the same population, the retrained classifier is strictly better on standalone top-3 for both
repos, not merely a wash against a large reachability gain.** ADR-0056's Phase 3 addendum table
is retracted as a basis for any "regression" framing (the underlying numbers in that table are
not disputed, only their use as a paired before/after).

**Mechanism, confirmed directly, not inferred:** 17.97% of vscode's fresh test rows (76/423) and
5.51% of k8s's (37/671) carry a gold label the old classifier cannot structurally ever predict
(not in its 28/35-class list) — every one of those rows scores zero for the old classifier by
construction, and that alone accounts for most of the population-level gap. Restricting to only
the rows the old classifier could ever get right (the in-taxonomy subset), it still underperforms
the new classifier's full-population score on vscode (83.00% vs. 85.82%) and is only modestly
ahead of it on k8s (88.49% vs. 85.99%, on a subset that already excludes 5.51% of the true
incoming population) — the new classifier is not winning "only" because it gets credited for
labels the old one was never asked about.

## Reachability metric re-confirmation (working agreement Phase 2a)

With the classifier now promoted, re-ran the gold-in-top-3 check
(`reports/grounding_measurement.json`'s prior working-tree diff, produced against the OLD
classifier on the fresh 64-issue cassette, already showed 4/11 vscode `component_ungrounded`).
Per ADR-0056's addendum, the NEW classifier resolves 3 of those 4 (the LLM's own correct override
now lands inside the new classifier's top-3): `#239838`, `#311284`, `#311878`. `#311836` (the one
genuine model error) stays correctly flagged. This needs re-running end-to-end against the now
NEW-classifier-produced `classifier_top3` (Phase 2 of the working agreement, next) — not done as
part of this ADR, since `reports/grounding_measurement.json`'s current working-tree state still
reflects the old classifier's `classifier_top3_labels` values and will need regenerating.

## Coverage gap: k8s `system-requirement` (Phase 1d)

Directly re-verified this session (not carried forward from ADR-0056's estimate): counted
`system-requirement` across the fresh `kubernetes_kubernetes_classifier_{train,val,test}.parquet`
splits combined — **zero occurrences** (0/6,710 rows), more decisively excluded than ADR-0056's
"raw count exactly 10, falls under `min_class_samples=10` post-collapse" estimate. This is the
one label neither this retrain nor a future one on the current corpus fixes without new data
collection. Recorded as a genuine, still-open coverage gap — not a staleness artifact this ADR
resolves.

## Conformal / CQR calibration validity (Phase 1f)

**VERIFIED unaffected — no regeneration needed.** `src/triage_iq/models/resolution.py:131-136`
explicitly excludes component/priority features from the resolution-time model's inputs (ADR-0009
T1.4: these are triage-assigned, not available at issue-creation time, and including them would
leak). The `top_components` attribute (`resolution.py:211,434`) is declared and serialized but
never populated or read as a feature anywhere in the class — confirmed by exhaustive grep, only
those two lines reference it. `data/models/cqr_conformal_adjustments.json`'s calibration has no
dependency on the component classifier in any form, so swapping the classifier artifact does not
invalidate it.

## Downstream numbers invalidated (Phase 1e — enumerated, not edited)

| Location | Claim | Status |
|---|---|---|
| `README.md:51` | "vscode: 89.8% top-3 acc (76.5% top-1)" (architecture diagram) | Invalidated — old classifier, old test population |
| `README.md:93-96` | Evaluation table: vscode 89.8%/76.5%, k8s 87.1%/60.5% | Invalidated — same |
| `README.md:264` | "classifier top-3 82.5%/90.4%" (historical retrieval-audit aside) | Stale historical figure, predates ADR-0036; now doubly stale |
| `README.md:296` | "88.2% vs. TF-IDF's 90.4% (vscode) and 74.5% vs. 82.5% (k8s)" (DistilBERT comparison) | The TF-IDF comparator side is invalidated; DistilBERT was never retrained/compared against the new classifier |
| `README.md:309` | "k8s top-3 82.5%→87.1%", "vscode top-3 flat (90.4%→89.8%)" (ADR-0036 shipped-fix table) | Historical record of that decision, endpoint numbers now superseded |
| `reports/classifier_results.json` | Old classifier's own stale-test report (train/val/test 1488/187/187) | Superseded by `reports/multilabel_classifier_final_training.json` + this ADR's fresh-population numbers; left in place as historical record, not edited |
| `docs/architecture/adr/0044-fabrication-rate-hard-gate-bound.md:21` | `_GROUNDING_BASELINE` figures (0/53, 0/11) | Already flagged invalid by ADR-0056 (taxonomy gap); now also stale on the classifier dimension |
| `eval/test_invariants.py:69-90` `_GROUNDING_BASELINE` | ungrounded_count 0/0 for both repos | Ratcheted against the old classifier + a pre-ADR-0055-schema cassette; not updated by this ADR — Phase 4's job, after the full 64-issue re-record |
| `reports/eval_baseline.json` | `fabrication_rate: 0.0` both repos, BLOCKING gate basis | Measured against the old classifier and the pre-ADR-0055 cassette; superseded, not yet replaced — ADR-0052 remains open, Phase 4 sets the real baseline |
| `reports/grounding_measurement.json` (current working-tree diff) | 4/11 vscode ungrounded, 0/53 k8s | Reflects the OLD classifier's `classifier_top3` against the fresh 64-issue cassette; itself now stale post-promotion — needs regenerating against the NEW classifier (Phase 2) |
| `data/models/cqr_conformal_adjustments.json` | Q-adjustment values | **NOT invalidated** — see Phase 1f above |
| Judge quality mean (`reports/eval_baseline.json`'s judge-mean fields) | 10.26/15 k8s, 8.64/15 vscode | **NOT invalidated** — judge never sees `classifier_top3` or gold `component` (ADR-0028), independent of which classifier is deployed |
| Resolution-time / bucket-classifier numbers | various | **NOT invalidated** — component features excluded from that model (Phase 1f) |

None of these were edited as part of this ADR, per the working agreement's explicit instruction
to report, not fix, at this phase.

## Consequences

- **What changes:** the locally-deployed component classifiers now cover 47 classes each
  (vscode/k8s), sourced from the current `data/raw/` corpus rather than a ~2.3x smaller stale
  snapshot. Any code path reading `data/models/component_classifier_{repo}.pkl` (the live API via
  `load_classifier`, the eval harness, `scripts/w5_t3_generate_candidates.py`) now gets the new
  classifier the next time it runs in this worktree.
- **What does NOT change:** production. The GCS-published artifacts and `MANIFEST.sha256` are
  untouched — production still serves the old classifier until a future, explicitly-approved
  Phase 5 publish + deploy.
- **What becomes easier:** the "is this a regression" question is now settled with a fair
  comparison — future discussion of this retrain does not need to relitigate whether -4pp is an
  acceptable cost, because there is no cost on this measurement.
- **What remains open:** `reports/grounding_measurement.json` needs regenerating against the new
  classifier (Phase 2); `_GROUNDING_BASELINE` and `reports/eval_baseline.json` still need a
  genuine re-baseline after the full 64-issue re-record (Phase 4, ADR-0052); k8s
  `system-requirement` remains uncovered by any classifier trainable on the current corpus.

## Phase 2 addendum (2026-09-04, same session): grounding metric re-confirmed, one new genuine catch found

**Working agreement Phase 2 asked to confirm the new classifier resolves 3 of the 4 known
vscode override cases while `#311836` (the genuine error) stays flagged, and to report
whether the metric needs redesigning.**

**Mechanism problem found first, before any result:** `scripts/measure_grounding.py`
re-invokes the full synthesis pipeline (`_collect_signals` → `_call_llm_verbose`), and the
LLM's prompt embeds `classifier_top3` as text — since the classifier changed, that text no
longer matches the committed cassette's recorded requests, so every replay hits
`CassetteMissError`. This is the guard working correctly (failing closed on a real content
change, not silently returning stale data) — not a bug to route around by weakening the
cassette's strictness. Built `scripts/measure_grounding_new_classifier.py` instead: it
reconstructs the request against the OLD (archived) classifier specifically to get a
cassette hit and recover the LLM's already-recorded plan (the model's synthesis judgment
doesn't need a new call just because a downstream feature classifier changed), then
separately computes the NEW classifier's top-3 for the same issue text and re-runs
`compute_grounding_status` — the same function production and `measure_grounding.py` both
use — against that. Zero live calls. `enable_validated_override_rescue` explicitly held at
`False` (working agreement 2d — not this session's decision to make).

**Result, full 64-issue eval set, not just the 4 previously-known cases:**

| | before (old classifier, this ADR's Phase 1 baseline) | after (new classifier) |
|---|---:|---:|
| vscode component-ungrounded | 4/11 (36.36%) | **1/11 (9.09%)** |
| k8s component-ungrounded | 0/53 (0%) | **1/53 (1.89%)** |
| overall | 4/64 (6.25%) | **2/64 (3.12%)** |

**vscode: confirmed exactly as expected.** `#239838`, `#311284`, `#311878` all resolve — the
LLM's own (unchanged) prediction now lands inside the new classifier's top-3. `#311836` stays
flagged: predicted `webview`, new top-3 `[ux, extensions, api]`, gold `perf` — a genuine
model error, not a taxonomy artifact, matches ADR-0056's addendum exactly.

**k8s: one new, previously-unflagged case appeared — `#14711`.** Not one of the working
agreement's 4 anticipated cases. Investigated directly: gold component is `kubectl`. Old
classifier's top-3 for this issue was `[introspection, logging, usability]` — the LLM
predicted `usability`, which happened to be in that top-3, so the case registered as
grounded even though `usability` does not match gold. The new classifier's top-3 is
`[kubectl, introspection, monitoring]` — gold (`kubectl`) is now correctly in top-3, but the
LLM's prediction (still `usability`, unchanged, an unchanged genuine wrong answer) no longer
matches any of the three, so it is now correctly flagged ungrounded. **This is the metric
catching a real LLM error that the OLD, less accurate classifier was accidentally masking**
— the new classifier didn't introduce a new failure, it removed a false negative. Strengthens
the case that the metric is functioning correctly, not evidence against the retrain.

**2b: no metric redesign needed — confirmed, reported plainly.** Both post-retrain ungrounded
cases (`#311836`, `#14711`) are genuine model errors (LLM prediction matches neither gold nor
either classifier's top-3), and the metric correctly separates them from the 3 vscode cases
that were classifier-reachability artifacts. `verify_plan_grounding`'s existing definition
does exactly what ADR-0056 said it does — no change to the function itself is warranted by
this data.

**2c: vscode's n=11 arm still cannot support a hard zero-tolerance gate — reporting the n
needed, as instructed, not deciding to change the gate.** Current observed rate 1/11 = 9.09%,
Wilson 95% CI **[1.6%, 37.7%]** — wide enough that this single observation is consistent with
a true rate anywhere from "very rare" to "over a third of cases," not actionable as a signal
on its own. Computed the sample size a **zero-failure observation** would need to bound the
true rate with reasonable confidence (Wilson upper bound, 95%):

| Target ceiling on true rate | n needed (0 observed failures) |
|---|---:|
| ≤20% | 16 |
| ≤10% | 35 |
| ≤5% | 73 |
| ≤2% | 189 |

vscode's current n=11 doesn't clear even the loosest useful ceiling (20% needs n=16). k8s's
n=53 clears the 10% ceiling (35) but not the 5% one (73) — its 1/53 result (CI [0.3%, 9.9%])
is a materially more informative signal than vscode's, though neither is fully resolved at a
5% ceiling. This is the same eval-set-expansion gap ADR-0056 already flagged as a
precondition for a defensible vscode gate — not a new finding, but now with a concrete
target (n≈60-75 for a 5%-ceiling gate, matching this project's other statistical bars) rather
than a qualitative "too small."

**Explicitly not done: no gate was loosened, tightened, or redefined.** `_GROUNDING_BASELINE`
in `eval/test_invariants.py` is untouched by this addendum — it remains stale (ratcheted
against the old classifier and a pre-current cassette) and will be re-derived properly in
Phase 4, against a real re-recorded cassette, not this diagnostic workaround.

Artifacts: `scripts/measure_grounding_new_classifier.py`,
`reports/grounding_measurement_new_classifier.json`.

## Phase 3 addendum (2026-09-04, same session): declared_attribution restored, live-validated

**Working agreement Phase 3 asked to land `declared_attribution` — restore it to the wire
schema as optional (never a hard `required`-with-no-escape), fix the prompt-variant gap
that left it permanently null, and validate on the 4 known override issues.**

**3a — schema restoration.** `declared_attribution` had `default=None` like the 6 truly
post-hoc fields (`resolution_bucket`, etc.), so ADR-0055's generic
default-vs-default_factory `_strip_post_hoc_fields` mechanism stripped it too — correctly
by that mechanism's own logic, but wrongly in effect: unlike the other 6 (fixed values the
app overwrites regardless of what the model emits), `declared_attribution` is real,
LLM-elicited signal (ADR-0020) with no other source. Added an explicit carve-out,
`_NEVER_STRIP_DESPITE_DEFAULT = frozenset({"declared_attribution"})`
(`src/triage_iq/models/triage.py`), rather than weakening the general mechanism for the
other 6 or changing `declared_attribution`'s Pydantic default (which would also weaken its
parsing-safety contract). Result: 11 → 12 required fields; `declared_attribution` is back
in `properties` as `type: [object, "null"]` (Groq's strict mode still forces `required` to
equal every property — there is no way to make a key literally absent from `required` and
still send it — "optional" here means the model can satisfy that requirement with `null`,
exactly the pattern the other four `X | None` fields used before ADR-0055). `$defs` stays
`{SimilarIssue}` only: `DeclaredAttribution`'s own $def becomes unreachable once its schema
is inlined into the property (it has no nested BaseModel refs of its own), confirmed
directly rather than assumed. `tests/test_wire_schema_excludes_post_hoc_fields.py` updated:
6 fields still stripped (was 7), a new `test_declared_attribution_restored_as_nullable_object`
pins the restored shape, required-count pin moved 11→12. Full suite: 309/309 pass.

**3b — the prompt-variant gap.** `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION` (ADR-0020) defaults
off — the 64-issue re-record ran without it, so every response used `SYSTEM_PROMPT_LEGACY`
(no attribution instructions, no attribution few-shot exemplars) regardless of the schema
change, which is why `declared_attribution` was `None` on all 64 entries even before 3a's
schema fix was diagnosed. **Not a bug**: ADR-0020 designed this default deliberately
("What reaches production: nothing, until `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1` is
explicitly set... a separate, deliberate deploy decision, not part of this ADR"). This
session does not flip that default in code — doing so would silently change what every
future synthesis call sends without a corresponding Cloud Run env var change, which is a
live-production-behavior change gated to Phase 5 approval, not a Phase 3 modeling
decision. Instead: the flag is set explicitly (`TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1`) for
this session's own validation call and will need the same explicit setting for Phase 4's
real re-record — a local/CI-invocation choice, not a code or Cloud Run change.
`eval/record_cassettes.py`'s `_compute_prompt_hash()` already reads this same env var
(confirmed by direct code read, no change needed) — the checkpoint-hash mechanism already
correctly distinguishes an attribution-on recording from an attribution-off one.

**Token cost, measured offline (zero live calls, `scripts/measure_attribution_token_cost.py`,
same estimator the live per-request guard uses):** mean +46.2 prompt tokens/call (min 46,
max 47) — the attribution-rules prompt section + extended few-shot exemplars. Against the
guard's actual ceiling (8,000 TPM, 200-token margin, 800-token completion floor): **0/64
issues cross the input-shrinking threshold either with or without attribution** — this cost
is small relative to the ~500-600 tokens of headroom `gpt-oss-120b` carries per ADR-0054,
not a comparability break on the token-budget dimension.

**Is it a comparability break, more broadly?** Yes, but not a NEW one: Phase 4's full
re-record already invalidates every existing baseline on three independent axes (model,
schema, classifier — ADR-0054/0055/0057). Turning attribution on for that same re-record
adds a fourth axis of change happening in the same event, not a separate incomparability to
manage on its own. Nothing currently comparable is broken by this, because nothing
currently exists that this change would need to stay comparable to (ADR-0052: no valid
baseline exists yet).

**3c — live validation on the 4 known override issues** (vscode #239838, #311284,
#311878, #311836), current production path (`openai/gpt-oss-120b`, new 47-class
classifier, restored schema, `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1`), `cache=None` to force
genuinely new calls (`scripts/scratch/validate_declared_attribution_4_overrides.py`,
gitignored scratch, one-time validation):

| Issue | predicted_component (this live draw) | declared_attribution | component_source | component_override_reason |
|---|---|---|---|---|
| #311284 | scm | non-null | classifier_top3 | "" |
| #311836 | api | non-null | classifier_top3 | "" |
| #311878 | terminal | non-null | classifier_top3 | "" |
| #239838 | terminal | non-null | classifier_top3 | "" |

**declared_attribution non-null: 4/4 (100%).** The restoration mechanism works end to end
— schema + prompt fix together produce well-formed, non-null attribution on every call.

**All 4 reasons are verbatim empty strings — expected, not a gap.** Per `DeclaredAttribution`
(ADR-0020), `component_override_reason` is populated only when `component_source ==
"model_override"`; all 4 calls declared `"classifier_top3"` instead. This is consistent
with — not contradicting — Phase 2's finding: the new classifier now places 3 of these 4
issues' correct predictions inside its own top-3 (`#239838`, `#311284`, `#311878`), so the
model is not overriding anything on this draw and correctly self-reports `classifier_top3`
sourcing. `#311836` drew `api` this time (a different live sample than the earlier
cached `webview` — Groq synthesis has documented replica-level nondeterminism even at
`seed=42`, ADR-0019/0020) — `api` also happens to sit in the new classifier's top-3
(`[ux, extensions, api]`, per Phase 2's measurement), so it too self-reports
`classifier_top3`, not `model_override`. **The working agreement's framing anticipated
these as override cases; the classifier retrain already resolved most of them out of that
category before this validation ran** — a real, coherent convergence across phases, not a
validation gap. The `model_override` branch (and non-empty
`component_override_reason` text) remains untested by this specific 4-issue draw; it is
exercised by `#311836`'s gold-mismatch class more generally (whichever component the model
guesses that ISN'T in top-3 on a given draw) and by ADR-0020's original n=65 measurement
(2 misattributed cases out of 65, both pre-existing classifier misses), not newly at risk
here.

Artifacts: `scripts/measure_attribution_token_cost.py`, `reports/attribution_token_cost.json`,
`scripts/scratch/validate_declared_attribution_4_overrides.py`,
`reports/declared_attribution_4_override_validation.json`.

## Alternatives considered

- **Accept ADR-0056's framing (small regression, large reachability gain) and ship on that
  basis, as instructed.** Not what happened — the instruction assumed a trade-off existed;
  checking it directly found the trade-off doesn't exist on a fair comparison. Reporting the
  stronger, correct finding rather than silently proceeding on the weaker one the instruction
  assumed, per the working agreement's explicit "surface findings that contradict a shipped
  conclusion" clause — this doesn't contradict the decision to ship, it removes the one
  argument that could have been used against it.
- **Run `publish_models.py` and update `MANIFEST.sha256` now, since the retrain is approved.**
  Rejected: conflates "ship the retrain" (a modeling decision, approved) with "deploy to
  production" (an infrastructure action, explicitly gated to Phase 5 with its own "do not
  deploy" instruction). Doing both together would make the model swap live before the
  re-baseline, contradicting ADR-0052/the working agreement's own sequencing.
