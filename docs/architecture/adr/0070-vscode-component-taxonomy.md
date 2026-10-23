# ADR-0070 — vscode component taxonomy: rule-generated parent areas, abstention scored as a miss (DRAFT)

Status: Proposed (unattended run; frozen before any vscode model is scored)
Date: 2026-10-23
Decider: to be confirmed by Gaurav Gandhi (merge of this ADR is the freeze; the unattended run only proposes it)

Evidence: `reports/qp2_vscode_taxonomy_2026-10.json`, produced by `scripts/qp2_vscode_taxonomy.py` at commit `f9e9634`
(design pre-registered in its docstring). Every number below comes from issues created before 2026-01-01
(train 2023-01-01..2025-06-30 and validation 2025-07-01..2025-12-31, the boundaries of ADR-0067 / `qp0_make_splits.py`);
rows created on or after 2026-01-01 are filtered at read time and the script asserts none survive (tested). The only
test-period figure quoted is prior context from `docs/HANDOFF_2026-10-20_quality_push_pause.md` (31% of 2026 labelled
primaries are labels never seen in train), and it is not recomputed. Timelines are the complete direct-scraped ones
(`D:\triageiq-data\derived_v2\microsoft__vscode`, 74,576 train + validation issues).

## Context

vscode has no `sig/*`-style routing family. Components are feature-area labels (`terminal-*`, `editor-*`, `chat-*`,
`notebook-*`, `git`, `scm`, ...), mixed with type, workflow and bookkeeping labels. Two facts shape the decision.

1. **The vocabulary drifts quickly.** Only 36.3% of train and 26.9% of validation issues carry any non-excluded
   component label. New Copilot/AI areas appear in validation that are absent or sparse in train (`chat-billing` 10 train /
   600 validation issues, `chat-terminal` 21 / 459, `chat-agents-view` 1 / 257, `NES` 16 / 171, `capi` 2 / 104).
2. **A class list of exact labels goes stale within a quarter** (tables below). The superseded freeze on
   `feat/qp0-vscode-audits` (100 classes, built on incomplete timelines) used a class family of labels with >= 100 issues over
   ALL years, i.e. it used validation and test counts; it is discarded together with its reports.

### Exclusion list (non-component labels), derived from label semantics

Exclusion is by meaning, not by frequency (same stance as ADR-0067): a label is excluded if it says what kind of issue it is,
where it is in the workflow, or what platform or dependency it touches, rather than which product area owns it. Categories
(counts of exact entries; the full list and the regex are in the report and in `EXCLUDE` in the script, sha256
`41cd2bfe1475162e61874a2dad2beb92e0a11da6d0830c5586c99ca0b8902d9b`):

| Category | Entries | Examples |
|---|---|---|
| issue type | 11 | `bug`, `feature-request`, `debt`, `polish`, `regression`, `api-proposal` |
| workflow / resolution state | 34 | `info-needed`, `triage-needed`, `verified`, `insiders-released`, `confirmed`, `stale`, `invalid`, `spam`, `help wanted` |
| test-plan / release bookkeeping | 22 | `testplan-item`, `plan-item`, `endgame-plan`, `integration-test-failure`, `ai-translated` |
| platform / OS | 18 | `windows`, `linux`, `macos`, `wayland`, `snap`, `deb` |
| upstream dependency | 7 | `upstream`, `electron`, `chromium` |
| cross-cutting attribute | 10 | `accessibility`, `ux`, `perf`, `freeze-slow-crash-leak`, `error-telemetry`, `chat-ext-issue` |
| language without an owning team | 10 | `python`, `go`, `ruby`, `c++`, `latex` |
| model vendor | 6 | `gemini`, `claude`, `gpt` |
| regex | | `^[*~]` (process markers such as `*duplicate`, `*out-of-scope`, `~info-needed`), `translation-required-*`, `electron-NN-update`, `fixed-in-electron-*` |

Judgement calls, stated so they can be overruled: `accessibility` and `ux` are treated as cross-cutting attributes because they
co-occur with a component label rather than naming an owner (an `accessibility`-only issue is unlabelled for this task);
`electron`/`chromium` as upstream dependencies; `typescript`, `javascript`, `json`, `html`, `css-less-scss`, `emmet` kept because
they route to the language-feature owners. Changing any of these changes the hash and needs a new ADR before scoring.

### Primary label

As ADR-0067 and `qp0_ceiling_labels.per_issue`: among the non-excluded labels still present at the end, the label applied
earliest (labels with no event were applied by the issue template at creation and rank first), ties alphabetical. Population =
issues with at least one non-excluded label.

## Evidence (train and validation only)

**(a) Training window.** Class family = non-excluded labels carried by >= 30 window issues (labels_final, issue level);
coverage = share of the 6,520 labelled validation issues whose primary label is in the family (Wilson 95%).

| Train window start | Window issues | Labelled in window | Exact-label classes | Validation coverage | Parent-area classes | Validation coverage |
|---|---|---|---|---|---|---|
| 2023-01-01 | 50,363 | 18,268 | 158 | 53.5% [52.3, 54.7] | 85 | **88.7% [87.9, 89.5]** |
| 2024-01-01 | 28,981 | 10,539 | 98 | 48.4% [47.1, 49.6] | 52 | 85.4% [84.5, 86.2] |
| 2024-07-01 | 19,053 | 7,039 | 74 | 45.8% [44.5, 47.0] | 45 | 84.7% [83.8, 85.5] |
| 2025-01-01 | 9,764 | 3,529 | 26 | 33.3% [32.1, 34.4] | 20 | 77.8% [76.8, 78.8] |

A more recent window is **worse**, not better: a shorter window loses classes below the 30-issue floor faster than it drops stale
ones, and it also cuts the training size by 2x to 5x. The full 2023 window is kept. Minimum-count sensitivity (2023 window,
validation coverage, exact / parent): >= 10: 79.7% (257 classes) / 92.6% (125); >= 20: 66.6% (201) / 89.6% (107); >= 30: 53.5% (158) /
88.7% (85); >= 50: 46.9% (100) / 86.1% (53). Lowering the floor to 10 does not reach 95% either, and it creates classes with too few
examples to learn or score.

**(b) Open-set share and staleness.** With the 2023-window exact family, the abstention rate on validation is 46.5%: 373 of the
3,030 outside primaries are labels never seen in the window, 2,657 are seen but sparse (< 30 issues), i.e. mostly labels that
exploded after the cut. Per month it is 44.5%-49.0% with no recovery. A frozen family decays with age (coverage of a family frozen
at 2025-01-01 on later quarters, exact / parent): 79.7 / 91.3% in the next quarter, 65.4 / 91.4% two quarters out, 44.8 / 90.3% three
out, 46.1 / 85.0% four out. Even a family frozen the day before the evaluated quarter covers only 91-93% at parent level. Refitting the family helps materially: for validation Q4 (3,434 labelled), the train-only family covers
53.1% exact / 86.3% parent; adding validation Q3 to the family lifts it to 82.2% / 90.4%.

**(c) Parent areas.** A mechanical prefix rule (`parent(L) = P` if `L == P` or `L` starts with `P-`, for `P` in `chat`, `terminal`,
`notebook`, `editor`, `workbench`, `authentication`, `debug`, `settings`, `remote`, `search`, `keybindings`, `extension(s)`, `file`;
else `L` itself) gives 85 parent classes at >= 30 window issues and 88.7% validation coverage (row 1 of the table). It also maps
labels unseen in train by the same rule (`chat-billing`, `chat-terminal`, `chat-agents-view` fall into `chat`), which is why the
coverage gap closes. A naive first-token rule was tried first and rejected before scoring anything: it merged unrelated labels
(`tree-sitter` with `tree-views`, `code-cli` with `code-lens`, `multi-monitor`, `simple-browser`), so the prefix list is explicit; it
was chosen from label names seen in train and validation only. What remains outside the parent family on validation (11.3%) is
new product areas that share no prefix: `NES` 167, `capi` 100, `agent-behavior` 61, `copilot-cli-agent` 65, `model-byok` 65,
`model-picker` 52, `auto-model` 32.

## Decision

1. **The vscode target is a parent-area taxonomy generated by a frozen rule**, not a hand-listed class set:
   non-excluded labels (spec hash above) -> `parent_of` (prefix list above) -> classes = parents carried by >= 30 training-period
   issues (labels_final, issue level) -> everything else is `other`. Training window starts 2023-01-01; the window is not shortened.
2. **Frozen class list** for the train period <= 2025-06-30: the 85 classes below, sha256 of the newline-joined sorted list
   `95129c95809c21574ed34d41a46358149322680a7ccb0e42e90f4b62e7b761ea`. The Q3 split manifest records the hash of the list
   generated by this rule from the actual training split used for the final fit; if the final fit adds validation data, the list is
   regenerated by the same rule (no manual edit) and its hash recorded before any test scoring.
3. **Primary label, population, multi-label handling** as ADR-0067 (top-1 / top-3 against the primary label; any-valid-label
   accuracy against the full set; both reported).
4. **Abstention semantics (recommended): scored as a miss.** An issue whose primary label's parent is outside the class list
   (truth `other`) is a miss for top-1 and top-3 in the headline metric regardless of what the model outputs; predicting `other`
   never earns a hit. The headline therefore has an upper bound equal to coverage. `other` still exists as a class so that
   conformal sets and abstention are well defined, and these are reported next to the headline, never instead of it:
   (i) in-family accuracy (top-1 / top-3 over issues whose primary is in the family), (ii) the declared-`other` variant in which
   a correct `other` prediction counts, (iii) abstention precision/recall, (iv) the abstention rate itself.
5. Evaluation population: issues with at least one non-excluded label at their final state (26.9% of validation issues). The
   unlabelled share is reported next to every result.
6. **Scheduled retraining is part of the product design (section below)**; a frozen-forever family is rejected.
7. No vscode model is scored on this taxonomy before this ADR is merged.

Frozen classes (85, train <= 2025-06-30): `WSL`, `accessibility-signal`, `api`, `authentication`, `breadcrumbs`, `chat`, `code-cli`,
`code-lens`, `command-center`, `comments`, `config`, `cross-file-editing`, `css-less-scss`, `custom-editors`, `debug`, `dictation`,
`diff-editor`, `editor`, `emmet`, `error-list`, `extensions`, `file`, `font-rendering`, `formatting`, `getting-started`, `git`,
`github`, `gpu`, `grammar`, `html`, `inlay-hints`, `inline-chat`, `inline-completions`, `install-update`, `interactive-window`,
`issue-reporter`, `javascript`, `json`, `keybindings`, `l10n-platform`, `languages-basic`, `languages-diagnostics`, `layout`, `log`,
`markdown-ext`, `menus`, `merge-editor`, `multi-diff-editor`, `native-file-dialog`, `network`, `notebook`, `open-editors`, `outline`,
`output`, `panel-chat`, `proxy`, `quick-chat`, `quick-open`, `quick-pick`, `remote`, `scm`, `search`, `settings`,
`simple-file-dialog`, `snippets`, `suggest`, `tasks`, `telemetry`, `terminal`, `testing`, `themes`, `titlebar`, `tree-sitter`,
`tree-sitter-autoindent-candidate`, `tree-views`, `tree-widget`, `typescript`, `unc`, `user-profiles`, `vscode-build`, `vscode.dev`,
`web`, `webview`, `workbench`, `workspace-trust`.

### Feasibility of top-3 >= 95%

**Not feasible as pre-registered (top-3 over all labelled issues, abstention = miss).** Top-3 is bounded by coverage; the best
validation coverage of any option tried is 88.7% [87.9, 89.5] (parent areas, 2023 window, family frozen 0-6 months before the
evaluated period), and 92.6% even with a floor of 10 issues. Refit evidence (90.4% for a Q4 parent family refitted with Q3) shows
retraining narrows but, on these data, does not close the gap to 95%; the test period is 6-12 months after the train end and the
2026 vocabulary shift is already known to be large (31% of 2026 labelled primaries are labels never seen in train, handoff doc, not
recomputed here). The spec target is NOT changed by this ADR. Best achievable framing, to be used in reports next to the
unchanged target: top-3 on in-family issues (the model's real skill, with its coverage stated), top-3 over all labelled issues
(bounded above by coverage), and the declared-`other` variant. Whether in-family top-3 reaches 95% is unmeasured: no model has been
scored. Amendments needed (not made here): `spec-quality-push.md` section 1 target row and ADR-0067's wording that the
component target is "per repo >= 95%" should state that the vscode figure is coverage-bounded; ADR-0069 conformal coverage for
vscode is defined w.r.t. the primary label including `other`.

## Scheduled retraining (design only)

- **Cadence:** monthly refresh of the class list and of the model; the incremental scraper already runs, so the labelled stream is
  available. Quarterly is too slow: exact-label coverage of a family one quarter old fell 14 points (79.7% -> 65.4%) and parent coverage fell 5 points by the fourth quarter (section (b)).
- **Trigger on drift, not only on the calendar:** each week compute, on issues older than 30 days (labels have settled), the share
  of labelled issues whose primary parent is in the served class list. Retrain immediately when this falls below 85% or drops
  by 3 points against the value measured at the last fit; also when a non-class label is carried by >= 30 issues in the last 90 days.
- **What is retrained:** (1) the class list, regenerated by the frozen rule (never hand-edited; the hash is logged), (2) the
  classifier and the conformal calibration on the most recent labelled window with a time-ordered holdout (conformal calibration is
  redone every time the class list changes, since coverage is class-list dependent), (3) not the exclusion spec or the prefix list,
  which change only through a new ADR.
- **Gate before promotion:** champion-challenger on the newest 30-day-settled holdout; promote on non-inferior in-family top-3 and
  coverage-bounded headline not below the champion; log `classes_sha256`, window, coverage and abstention rate in the model
  metadata. Serving shows "unknown area, route to manual triage" for abstentions instead of forcing a class.
- **Cost:** training is TF-IDF/embedding-head scale (minutes on CPU or a Kaggle kernel); no spend decision is implied here.

## Consequences

- vscode accuracy is reported as coverage-bounded; the headline number will be lower than the k8s `sig/*` one for a structural
  reason, and the README must say so (new task, never compared with the old 85.8% vscode figure, which used the old taxonomy).
- Parent areas are coarser than exact labels (`editor` and `workbench` each pool ~20 sub-labels, so a correct `editor` says less
  than a correct `editor-hover`). The sub-label can be a secondary output later; it is not scored here.
- Roughly 11% of labelled validation issues sit outside the family because new product areas (Copilot/agent features) share no
  prefix; this share grows with time and is the reason the retraining trigger exists.
- The 2026 test set will be scored against a class list at least 6 months stale unless the final fit includes validation data;
  that choice is a Q3 (splits) decision to be made before any scoring and recorded in the manifest.
- Honest limits: the exclusion list and prefix list were written after reading train and validation label names (not test); the
  `accessibility`, `ux` and `electron` calls are judgement; `tree-sitter-autoindent-candidate` slipped in as a class because its name
  is not a workflow label.

## Alternatives

- **Exact labels with >= 30 train issues (158 classes):** rejected, 53.5% validation coverage, 46.5% abstention.
- **A more recent training window (2024-01-01, 2024-07-01, 2025-01-01):** rejected on evidence, coverage falls monotonically
  (48.4%, 45.8%, 33.3% exact; 85.4%, 84.7%, 77.8% parent) and training size shrinks.
- **Lower floor (10 issues):** rejected, 92.6% parent coverage still below 95% and 125 classes with 10-29 examples each.
- **Open-set detector with `other` counted as a hit:** kept as a reported variant, rejected as the headline because a correct
  `other` is not a routing success and a model could inflate the metric by predicting a catch-all.
- **Freeze the exact-label class list of `feat/qp0-vscode-audits` (100 classes):** rejected, incomplete timelines and a leaking
  family definition (counts over all years).
- **Hand-curated parent hierarchy:** rejected, the unattended run cannot justify hand-made groupings; the prefix rule is
  reproducible and extends to unseen labels.
- **Single frozen list forever:** rejected, see (b).
