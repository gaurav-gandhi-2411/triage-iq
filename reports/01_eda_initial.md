# EDA Report — microsoft/vscode (First 5,000 Issues)

**Generated:** 2026-04-28  
**Data source:** `data/raw/microsoft_vscode/` (5,000 JSON files)  
**Processed parquet:** `data/processed/issues_microsoft_vscode.parquet`  
**Shape:** 5,000 rows × 16 columns

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| Total issues | 5,000 |
| Date range | 2015-10-13 → 2016-04-06 (first **6 months** of vscode's public life) |
| Open | 39 (0.8%) |
| Closed | 4,961 (99.2%) |
| Issues with type label | 2,415 (48.3%) |
| Issues with component label | 0 (0%) — see §6 |
| Issues with resolution_hours | 4,961 |

---

## 2. Open vs Closed Split

```
closed    4,961  (99.2%)
open         39   (0.8%)
```

The dataset is dominated by closed issues because these are the oldest 5,000 (by creation order). Nearly all early-era issues have since been resolved.

---

## 3. Top 20 Labels by Frequency

| Rank | Label | Count |
|---|---|---|
| 1 | verified | 1,780 |
| 2 | bug | 1,352 |
| 3 | feature-request | 1,069 |
| 4 | debug | 397 |
| 5 | important | 246 |
| 6 | *duplicate | 220 |
| 7 | info-needed | 213 |
| 8 | upstream | 180 |
| 9 | *question | 160 |
| 10 | api | 136 |
| 11 | javascript | 136 |
| 12 | ux | 122 |
| 13 | testplan-item | 103 |
| 14 | tasks | 98 |
| 15 | accessibility | 96 |
| 16 | languages-basic | 95 |
| 17 | typescript | 93 |
| 18 | error-telemetry | 90 |
| 19 | on-testplan | 80 |
| 20 | file-explorer | 76 |

**Notes:**
- `verified` is a QA workflow label (bug confirmed), not a classification target
- `*duplicate` and `*question` use asterisk prefix — vscode's convention for meta-labels
- `debug`, `api`, `javascript`, `typescript`, `file-explorer` are de-facto component labels (no prefix)

---

## 4. Type Distribution (Extracted)

| Type | Count | % of total |
|---|---|---|
| `None` (no type label) | 2,585 | 51.7% |
| `bug` | 1,352 | 27.0% |
| `feature` | 1,063 | 21.3% |

52% of issues carry no type label — either they're workflow items, duplicates, questions, or upstream issues that weren't given a primary classification.

---

## 5. Resolution Time Distribution

| Metric | Hours | Days |
|---|---|---|
| Mean | 3,113 | **129.7** |
| Median | 92.7 | **3.9** |
| p25 | 12.3 | 0.5 |
| p75 | 1,706.6 | 71.1 |
| p95 | 16,995.8 | 708.2 |

**Distribution is extremely right-skewed.** Median resolution is under 4 days (fast team), but the mean is pulled to 130 days by ~200+ "open" issues that were only resolved years later. The p95 of 708 days highlights a long tail of contentious or deprioritized issues.

For the resolution-time predictor (System 3), log-transform of `resolution_hours` will be essential.

---

## 6. Text Length Distribution

| Field | Mean (chars) | Median (chars) | p95 (chars) |
|---|---|---|---|
| `body_clean` | 434 | 302 | 1,192 |
| `title` | 51 | 49 | 86 |

Issue bodies are concise (median 302 chars). Titles are very short (median 49 chars). This is consistent with developer-filed bugs rather than user-submitted support requests. The 10,000-char truncation in `clean_text` never activates on this corpus.

---

## 7. Top 10 Authors by Issue Count

| Author | Issues |
|---|---|
| bpasero | 266 |
| joaomoreno | 192 |
| isidorn | 191 |
| Tyriar | 171 |
| alexdima | 151 |
| weinand | 147 |
| egamma | 142 |
| jrieken | 135 |
| dbaeumer | 97 |
| vscodeerrors | 88 |

**All top authors are Microsoft vscode team members.** The first 5,000 issues were largely self-reported by the dev team. `vscodeerrors` is an automated bot filing crash telemetry issues. This has implications for the model: training on this period may not generalize to community-reported issues from later years.

---

## 8. Sample Titles per Type

**Bug samples:**
- `[php] missing user code auto-complete on 0.10.1`
- `generated launch config is erroneous`
- `Link decoration bleeds to next line when wrapping`

**Feature-request samples:**
- `Improve the explorer view to match full VS design`
- `Always show path information even for non-focus editor`
- `Scroll markers should be focusable and snap into position`

Labels look well-calibrated to the titles. No obvious mislabelings in the sample.

---

## 9. Most-Commented Issues

| # | Title | Comments | State |
|---|---|---|---|
| 519 | Allow to change the font size and font of the workbench | 588 | open |
| 3130 | Allow customization of mouse shortcuts | 452 | open |
| 224 | Proper tabs for open files | 411 | closed |
| 396 | Add support for opening multiple project folders | 380 | closed |
| 4490 | Macro recording | 276 | open |

Most-commented issues are high-demand feature requests (tabs, multi-root workspace, font customization). These remained open with hundreds of votes for years — a strong signal for the resolution-time predictor.

---

## 10. Actionable Findings for Day 2+

| Finding | Implication |
|---|---|
| **Component column: 100% null** | Must extend `LABEL_FACET_PATTERNS["microsoft_vscode"]` to capture unprefixed labels (`debug`, `api`, `javascript`, `typescript`, `file-explorer`, `ux`, etc.) as components |
| **Date range is only 6 months** | The 5,000-issue limit captures only the early team-internal phase; for better diversity, the classifier will need issues from later periods or other repos |
| **52% unlabeled for type** | Need to decide: drop these rows for classifier training, or treat as a third class ("other") |
| **Resolution hours: extreme skew** | Use log-transform for LightGBM regression target; report MAE on log scale and original scale |
| **Author concentration** | Do not use author as a model feature — too sparse and not predictive for new issues filed by external users |
| **`verified` label (1780 items)** | Exclude from type/component classification; it's a workflow state, not a classification target |
