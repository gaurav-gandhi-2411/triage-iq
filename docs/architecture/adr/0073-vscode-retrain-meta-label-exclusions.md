# ADR-0073 — vscode retraining: exclude three meta labels from class generation

Status: Accepted (GG decision D92.2, `docs/DECISION_LOG_2026-10.md` entry D92, 2026-10-29)
Date: 2026-10-29
Decider: Gaurav Gandhi (D92.2)

Amends the exclusion handling of the retrain in ADR-0072 (which reuses the frozen ADR-0070 exclusion list unchanged). Code:
`RETRAIN_META_EXCLUDE` in `scripts/vscode_retrain_lib.py`, tests in `tests/test_vscode_retrain.py`.

## Context

ADR-0070 froze an exclusion list (`EXCLUDE` plus a regex in `scripts/qp2_vscode_taxonomy.py`) written after reading the 2023-2026 train and validation
label names. The monthly retrain (ADR-0072) regenerates the class list on a rolling window, so it meets labels the frozen list never saw. The dry run
of 2026-10-29 (as-of 2026-10-22, `D:\triageiq-data\retrain_dryrun\run1`) regenerated 121 classes, three of which are not components:

| label | what it says | train issues carrying it (dry run window, 30,205 train issues) |
|---|---|---|
| `ai-generated` | the issue text was written by an AI | 208 |
| `ai-triaged` | the issue was triaged by an AI workflow | 108 |
| `2026-themes` | a planning theme for the year | 61 |

They describe how an issue was produced or triaged, or a roadmap theme, not the product area, which is the same stance ADR-0070 took for
`bug`, `*duplicate` and bot or process markers. Left in, the model would learn to predict "was this AI-written" as a component and spend top-3 slots
on it. Counts above are from the dry-run window and are labels (an issue can carry several).

## Decision

1. The labels `ai-generated`, `ai-triaged` and `2026-themes` are excluded from class generation in retraining. They are held in one frozen constant,
   `RETRAIN_META_EXCLUDE` in `scripts/vscode_retrain_lib.py`, matched case-insensitively (the same comparison `is_excluded` uses for ADR-0070).
2. **How the exclusion works.** The retrain's label predicate is `tax.not_excluded(label) and label.lower() not in RETRAIN_META_EXCLUDE`. It is passed to
   `qp0_ceiling_labels.per_issue` exactly where `tax.not_excluded` was, so the three labels are dropped from an issue's label set before anything else
   happens: before the `parent_of` prefix rule and the >= 30-issue class count (`window_counts`, `regenerate_classes`), and before gold labels and the
   primary label are computed. Consequences by case:
   - An issue carrying the three labels plus a real component label keeps the component label; the meta labels disappear from its gold set.
   - An issue whose only labels are excluded ones ends with an empty label set. Retraining keeps only issues with >= 1 non-excluded label
     (`assemble`: `d[d["final"].map(bool)]`, the ADR-0070 population), so such an issue is **dropped from train, calibration and holdout**, i.e. it is
     handled like an issue with no label at all, not as `other`. In the dry-run window that is 70 train, 6 calibration and 2 holdout issues.
   - Other labels are unaffected, including their prefix grouping.
3. The frozen ADR-0070 list, its regex and the prefix list are not changed. **Adding any further exclusion needs a new ADR**; the constant is not
   edited to pick up a label that happens to look odd in a given month.

Observed in the regenerated class list and **not excluded** (for GG to rule on; nothing here is excluded by this ADR): `recent-regression` (200 train
issues in the dry-run window; marks a regression, arguably a kind of issue), `stable-anomaly` (74; arguably a kind of issue), `tree-sitter-autoindent-candidate`
(37; reads as a work-queue marker, though it names a component), `UBB` (111; billing programme name, may be a component), `unc`, `auto-model`, `otel`.
The first three are the likeliest meta labels; the rest look like components.

## Consequences

- The class list is smaller by three and the training population by the issues that carried only those labels (about 0.2% of train in the dry
  run); neither changes the ADR-0072 gate, which is re-run on the new window.
- Exclusion is by name, so a renamed or newly introduced meta label is not caught until a new ADR names it. This is the same limitation as ADR-0070
  and is why the observed list above exists.
- Metrics are not comparable to the dry run of run1 (different class space and population); the champion and challenger are scored on the same
  rows in the same class space, so the gate itself is unaffected. The champion (the served OVR pickle) is not retrained and may still emit these labels;
  they map to `other` via `to_class` when scored in the new class space, which costs the champion coverage (the real drift cost, per ADR-0072).
- A model trained before this ADR (run1) must not be served; the candidate to serve is the one produced after it.

## Alternatives

- **Keep them as classes** (status quo of run1): rejected by D92.2; they are not components and would occupy top-3 slots.
- **Edit the frozen ADR-0070 `EXCLUDE` list**: rejected; that would change the taxonomy that the ADR-0070 and ADR-0071 numbers (coverage, conformal) were
  measured against. A retrain-only constant leaves those numbers valid.
- **Map issues with only meta labels to `other`**: rejected; the issue's component is unknown, not "none of the families", and ADR-0070's population
  is "has a non-excluded label". `other` stays reserved for issues whose labels are real components outside the family.
- **Exclude all observed meta-like labels now** (`recent-regression`, `stable-anomaly`, ...): not decided here; D92.2 names three labels only.

## Addendum 2026-10-29 (GG decision D93): the meta-label rule

Written before any label was reviewed in detail and before any score of the retrained model was looked at.

**Rule (verbatim from GG):** "labels describing an issue's status or workflow rather than the code area it concerns".

**Operational tests.** A label is a meta label (EXCLUDE) if its name or GitHub description says any of:

1. a triage state or triage outcome (needs info, confirmed, under discussion, triaged by a process);
2. a release or verification status (verified, fixed in a release, insiders/stable-only, regression marker);
3. a process or bot workflow (stale, duplicate, assigned to a queue, applied by automation, AI-authored or AI-triaged marker);
4. a time-boxed theme or campaign (a year's plan, an iteration, a one-off initiative);
5. a candidate or work-queue marker (an issue is a candidate for something, not a place in the product).

A label is a component label (KEEP) if it names a feature, a UI area, a language or language service, an extension, a platform or OS, a subsystem
or a product programme whose issues are about one code area. A label that matches both a test above and names a code area is BORDERLINE; the
decision then follows what the label is used for: if removing it from the issue would lose information about *where* the code is, keep it
(BORDERLINE-KEEP); if it mainly says *when or in what state* the issue is, exclude it (BORDERLINE-EXCLUDE). Every borderline call is listed with its reason
for review by GG.

### Application of the rule (D93)

The rule was applied to all 703 distinct labels found in the scraped vscode data (`issues.parquet` `labels_final` and `labels_t72h`, and `label_events.parquet`),
not only the 121 classes of the dry run. The scraped data holds no GitHub label descriptions, so each decision rests on the label name and its counts;
no per-class or per-label model score, prediction or accuracy was looked at. The full table, with counts, the decision, whether the label is NEW relative to the
frozen ADR-0070 list and a reason, is `0073-vscode-label-decisions.md`. Summary: 215 EXCLUDE (195 already excluded by ADR-0070, 20 NEW), 9 BORDERLINE-EXCLUDE (all
NEW), 14 BORDERLINE-KEEP, 465 KEEP.

`RETRAIN_META_EXCLUDE` is now the 29 NEW exclusions (EXCLUDE plus BORDERLINE-EXCLUDE that ADR-0070 does not already exclude); it supersedes the three-label set of the
Decision section above and the "Observed ... not excluded" paragraph, which GG's D93 ruled on. A test parses the table file and checks the constant equals its NEW rows. The table is regenerable from the repo: `python scripts/adr0073_label_table.py` (`--check` exits non-zero if the committed file differs); the hand-made decisions are data in `scripts/adr0073_label_decisions.json`.
Labels with zero training examples that the rule does not exclude stay in the class list and evaluation as honest misses.

BORDERLINE calls, for GG's review:

- BORDERLINE-EXCLUDE: `modernization` (264 issues; an internal work stream with no single code area), `agent-host-mitigated` ("mitigated" status; ADR-0070 already excludes
  `mitigated`), `test-disabled` (status of a test), `service-outage` (incident state), `a11y-partner` (origin marker), `ack` (acknowledgement state), `on-unit-test`
  (sibling of the excluded `on-testplan`), and two outside the literal rule, taken for consistency with ADR-0070: `performance` (spelling variant of the excluded
  `perf`) and `papercut 🩸` (emoji variant of the excluded `papercut :drop_of_blood:`).
- BORDERLINE-KEEP: `UBB` (billing programme, issues are about the billing flow), `chat-request-failed`, `chat-infinite-response-loop`, `chat-rate-limiting`,
  `chat-lm-unavailable` (failure modes inside the chat subsystem), `chat-perf-regression`, `model-upgrade`, `smoke-test`, and `AI`, `canjd`, `hotspot`,
  `mock-feature`, `edge`, `chat-showcase` (meaning not determinable from the name; at most 5 issues each; exclusion needs a positive reason).
