# TriageIQ Data Card

**Version:** Day 2  
**Last updated:** 2026-04-28  
**Maintainer:** Gaurav Gandhi

---

## 1. Data Sources

| Repo | URL | License | Primary use |
|---|---|---|---|
| microsoft/vscode | https://github.com/microsoft/vscode | MIT | Component classifier, resolution predictor, duplicate detection |
| kubernetes/kubernetes | https://github.com/kubernetes/kubernetes | Apache-2.0 | Component classifier (Kubernetes-specific), cross-repo transfer |
| tensorflow/tensorflow | https://github.com/tensorflow/tensorflow | Apache-2.0 | Component classifier (ML framework domain) |
| pytorch/pytorch | https://github.com/pytorch/pytorch | BSD-3-Clause | Component classifier (ML framework domain) |
| apache/airflow | https://github.com/apache/airflow | Apache-2.0 | Component classifier (data pipeline domain) |

**Access method:** GitHub Issues REST API v3 (authenticated, 5,000 req/hour).  
**Data collected via:** `src/triage_iq/data/github_scraper.py` — paginated GET with comment fetch per issue.  
**GitHub Terms of Service:** Data collected under GitHub TOS § acceptable use (research/non-commercial). Issue content is user-generated and publicly available.

---

## 2. Schema Documentation

All processed data is stored as Apache Parquet at `data/processed/issues_{owner}_{repo}.parquet`.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `id` | int64 | No | GitHub internal issue ID (globally unique) |
| `number` | int64 | No | Issue number within the repo (e.g., #1234) |
| `title` | string | No | Issue title (raw, not cleaned) |
| `body_clean` | string | No | Issue body with HTML comments removed, code blocks replaced with `[CODE_BLOCK]`, whitespace normalized, truncated at 10,000 chars |
| `code_blocks` | string | No | Extracted fenced code blocks (raw content), concatenated, capped at 5,000 chars |
| `labels_raw` | list[string] | No | All GitHub labels applied to the issue at scrape time |
| `component` | string | Yes | Normalized component extracted from labels. Null if no component label matched |
| `type` | string | Yes | Normalized type: "bug", "feature-request", etc. Null if no type label matched |
| `priority` | string | Yes | Normalized priority extracted from labels. Null if no priority label (common for vscode) |
| `state` | string | No | "open" or "closed" |
| `created_at` | datetime (UTC) | No | Issue creation timestamp |
| `closed_at` | datetime (UTC) | Yes | Issue close timestamp. Null if state="open" |
| `resolution_hours` | float64 | Yes | `(closed_at - created_at).total_seconds() / 3600`. Null if open |
| `author` | string | No | GitHub login of the issue creator |
| `num_comments` | int64 | No | Number of comments on the issue at scrape time |
| `num_assignees` | int64 | No | Number of users assigned to the issue |

---

## 3. Label Normalization Rules

Labels are mapped to structured facets (component/type/priority) per repo. See full patterns in `src/triage_iq/data/preprocess.py: LABEL_FACET_PATTERNS`.

| Repo | Component extraction | Type extraction | Priority extraction |
|---|---|---|---|
| kubernetes/kubernetes | `area/(.+)` regex | `kind/(.+)` regex | `priority/(.+)` regex |
| tensorflow/tensorflow | `comp:(.+)` regex | `type:(.+)` regex | `stat:(.+)` regex |
| pytorch/pytorch | `module:\s*(.+)` regex | exact match (bug/feature/enhancement) | — |
| microsoft/vscode | allowlist of ~50 known labels | allowlist (bug/feature-request/etc.) | — |
| apache/airflow | `area:(.+)` regex | `kind:(.+)` regex | — |

**Note on vscode:** vscode uses flat, unprefixed labels (no `area/` prefix). Component extraction relies on a manually-maintained allowlist. The 2015-2016 issues have ~37% component coverage (hard ceiling due to many workflow-only labels like `verified`, `info-needed`, `*duplicate`). Recent 2025 issues may have higher coverage.

---

## 4. Known Biases and Limitations

### 4.1 vscode temporal bias
The initial 5,000 vscode issues span only Oct 2015 – Apr 2016 (launch period). A supplemental scrape of 2,028 most recent issues (2025-2026) is also included. The combined dataset has a 9-year gap (2016-2025) with no issues.

Historical slice characteristics:
- Top 10 authors are Microsoft employees (not community)
- Issues are terse and developer-style
- Component fill: 37% (ceiling set by many workflow-only labels)
- 99.2% closed

Recent slice characteristics:
- Community-heavy authorship (2,000+ unique authors in combined dataset)
- Labels shifted to new conventions (`triage-needed`, `agents`, `copilot-cli-agent`) not in historical allowlist
- Component fill only 6.1% — reflects labeling practice change, not extraction failure
- 56.9% closed (many issues still open, filed recently)

**Combined fill:** 28.1% component fill overall — below either slice in isolation due to the large number of unlabeled recent issues dragging down the mean.

### 4.2 PR/issue conflation
GitHub's Issues API returns both issues AND pull requests when `state=all`. The dataset contains PRs (which are a different artifact type). The `labels_raw` and `title` fields allow post-hoc filtering, but no column explicitly flags PRs vs issues. The `pull_request` key in the raw JSON can be checked if needed.

### 4.3 Author concentration
In the 2015-2016 vscode slice, top 10 authors account for ~1,700 of 5,000 issues (34%). These are team members filing internal bugs. Training a model on author features would not generalize to new external contributors.

### 4.4 kubernetes/kubernetes scale mismatch
Kubernetes uses 400+ label values for `area/`, `kind/`, and `priority/`. The long tail of components has very few examples per class, which will require minimum-class filtering in the classifier split.

### 4.5 Resolution time extreme outliers
vscode combined median resolution: 1.8 days. Mean: 104.9 days. p95: 666 days (historical alone: 3.9 / 129.7 / 708 days). The distribution is log-normal, not normal. All resolution time models must work on log-transformed targets.

### 4.6 Comment fetch completeness
Comments are fetched with `per_page=100`. Issues with >100 comments have only the first 100 captured. In the 2015-2016 vscode dataset, the maximum is 588 comments (issue #519) — these are truncated.

### 4.7 Known Preprocessing Limitations

**Image-only and link-only bodies stripped to empty.** Issue bodies consisting solely of image URLs or external links are stripped to empty during preprocessing (e.g., vscode #2093 — body is a single `https://cloud.githubusercontent.com/...` URL). This produces information-impoverished inputs to downstream models: `build_triage_prompt` renders empty bodies as `(no body)`, leaving the LLM with only the issue title. Future work: replace stripped media with placeholder tokens like `[image content]` to preserve the signal that visual context exists.

---

## 5. Cleaning Steps

1. **HTML comment removal:** `<!-- ... -->` blocks stripped from body
2. **Code block extraction:** Fenced ` ``` ` blocks moved to `code_blocks` column, replaced with `[CODE_BLOCK]` in `body_clean`
3. **Inline code normalization:** Single-tick inline code replaced with `[INLINE_CODE]`
4. **Whitespace normalization:** Multiple blank lines collapsed to double newline; multiple spaces collapsed
5. **Body truncation:** `body_clean` capped at 10,000 chars; `code_blocks` capped at 5,000 chars
6. **Malformed JSON skipping:** Files that fail JSON parse are logged and skipped
7. **Label normalization:** Per-repo regex or allowlist matching; first-match wins per facet

---

## 6. Split Methodology

Two split strategies implemented in `src/triage_iq/data/splits.py`:

### Time-based split (System 3 — Resolution Time Predictor)
- Split by `closed_at` timestamp ascending
- 80% train / 10% val / 10% test
- Open issues (null `closed_at`) excluded
- **Rationale:** Prevents leakage where a test issue was opened while training issues were being closed. A model that sees future issue metadata would be unrealistically optimistic.

### Stratified split (System 1 — Component Classifier)
- Split by stratified random sampling preserving label distribution
- 80% train / 10% val / 10% test
- Classes with < 10 examples dropped
- Random seed: 42
- **Rationale:** The component classifier needs representative class distributions in val/test. Temporal split would over-represent early components (e.g., vscode's initial feature set) in training.

---

## 7. Per-Repo Statistics

*(Both repos fully processed as of 2026-04-28)*

| Metric | microsoft/vscode (combined) | microsoft/vscode (historical) | microsoft/vscode (recent) | kubernetes/kubernetes |
|---|---|---|---|---|
| Issues scraped | 7,028 | 5,000 | 2,028 | 15,000 |
| Date range | Oct 2015 – Apr 2026 | Oct 2015 – Apr 2016 | 2025–2026 | Jun 2014 – Oct 2015 |
| % closed | 87.6% | 99.2% | 56.9% | 99.8% |
| Component fill % | 28.1% | 37.0% | 6.1% | 19.4% |
| Type fill % | 37.8% | 48.3% | ~12% | 14.7% |
| Priority fill % | — | — | — | 27.3% |
| Median resolution (days) | 1.8 | 3.9 | — | 1.9 |
| Mean resolution (days) | 104.9 | 129.7 | — | 83.3 |
| p95 resolution (days) | 666 | 708 | — | 677 |
| Unique component labels | 58 | ~45 | ~5 | 54 |
| Unique authors | 2,402 | ~500 (team-heavy) | ~2,000 | 1,175 |

**Note on vscode component fill:** Recent (2025-2026) vscode issues use different label conventions (`triage-needed`, `agents`, `copilot-cli-agent`) not in the historical allowlist. 6.1% fill for recent reflects labeling practice change, not extraction failure.

**Note on kubernetes era:** The 15,000 kubernetes issues are the oldest 15k by creation date (Jun 2014 – Oct 2015). This is early-kubernetes when the project was under rapid development. Label practices have changed significantly since. Component fill (19.4%) reflects sparse early labeling; recent kubernetes issues would likely have higher fill.

**Note on kubernetes component distribution:** Top components: api (399), test (390), test-infra (324), kubectl (261), usability (196), kubelet (138), introspection (116), security (111). Long tail of 54 unique components; 19 classes dropped (< 10 samples) for classifier split.

### Train/val/test split sizes — microsoft/vscode

| Split strategy | Train | Val | Test | Notes |
|---|---|---|---|---|
| Temporal (resolution predictor) | 4,923 | 615 | 616 | Train cutoff: Nov 2020 |
| Stratified (component classifier) | 1,488 | 187 | 187 | 1,862 labeled rows; 30 rare classes dropped |

### Train/val/test split sizes — kubernetes/kubernetes

| Split strategy | Train | Val | Test | Notes |
|---|---|---|---|---|
| Temporal (resolution predictor) | 11,974 | 1,496 | 1,498 | Train cutoff: Sep 2015 |
| Stratified (component classifier) | 2,284 | 286 | 286 | 2,856 labeled rows; 19 rare classes dropped |

---

## 8. Reproducibility

To regenerate all data from scratch:
```bash
# 1. Scrape
python scripts/01_scrape_issues.py --repo microsoft/vscode --max-issues 5000
python scripts/01_scrape_issues.py --repo microsoft/vscode --max-issues 2000 --sort created --direction desc
python scripts/01_scrape_issues.py --repo kubernetes/kubernetes --max-issues 15000

# 2. Preprocess
python scripts/02_preprocess.py

# 3. Split (run after Day 2 script is complete)
python scripts/03_split.py
```

Raw JSON files are gitignored (`data/raw/`). Parquet files are gitignored (`data/processed/*.parquet`). Only the code, scripts, and reports are versioned.
