# Gold Set Labeling Protocol — W5 Expansion

**Version:** 1.0  
**Date:** 2026-05-31  
**Maintainer:** Gaurav Gandhi

This document defines the procedure for GG to label the 120 candidate issues in `data/gold_expansion_candidates.csv` and select the final 90 to add to the gold evaluation set.

---

## 1. Goal

Select ~90 issues from the 120 candidates (45 per repo) to expand the gold eval set from n=60 to n=150. The candidates are already stratified by resolution bucket and component diversity. GG reviews each and marks it `accept` or `reject`. A `notes` column is available for per-issue observations.

---

## 2. Working file

Open `data/gold_expansion_candidates.csv` in a spreadsheet editor or any tool that can edit CSVs.

Add two columns if not already present:
- `label_decision` — values: `accept` | `reject` | `pending`
- `label_notes` — free text (optional, e.g. "bot-generated", "empty body", "duplicate theme")

Target selection: 9 per resolution bucket per repo (45 per repo). The 12-per-bucket pool gives 3 slots of margin per bucket.

---

## 3. Acceptance criteria

Accept an issue if ALL of the following hold:

| Criterion | Rule |
|---|---|
| **Has meaningful body** | `body_excerpt` is not empty, not just a URL, not just `[CODE_BLOCK]` placeholders. At least one sentence of human-readable description. |
| **Human-authored** | Not a bot-generated issue (e.g. no `[bot]` in title, no automated release notes, no Renovate/Dependabot content). |
| **English** | Issue title and body are primarily in English. |
| **Unambiguous component** | The `component` column matches what a human would assign from reading the title + body. If the issue feels mislabeled, reject it. |
| **Non-trivial** | Not a one-word title like "test" or a meta-issue like "add me to contributors". Needs enough context for the judge to evaluate triage quality. |
| **Closed with known resolution** | All candidates already have `resolution_hours > 0`; verify the issue wasn't closed as `wontfix` or `invalid` if you can check GitHub (optional — the label column may have clues). |

---

## 4. Rejection reasons (use in label_notes)

| Code | Meaning |
|---|---|
| `empty-body` | Body is blank or stripped to URL/code only |
| `bot` | Automated/bot-generated issue |
| `non-english` | Non-English content |
| `mislabeled` | Component label doesn't match issue content |
| `trivial` | Too short or too little content to judge meaningfully |
| `duplicate-theme` | This stratum already has enough accepted issues; rejecting to rebalance |
| `wontfix` | Issue was closed as "will not fix" rather than resolved |

---

## 5. Judge dimension rubric (for orientation)

These are the 6 dimensions the Cohere/Groq judge evaluates. Keeping them in mind while selecting helps ensure candidates cover the full rubric:

| Dimension | Max | What it tests |
|---|---|---|
| `component_match` | 2 | Does the predicted component match the gold? |
| `similar_issues_relevance` | 3 | Are the retrieved similar issues actually related? |
| `resolution_estimate_reasonableness` | 3 | Is the resolution time estimate plausible given the issue content? |
| `priority_alignment` | 1 | Does the LLM's priority judgment match inferred gold priority? |
| `next_steps_actionability` | 3 | Are the suggested next steps concrete and appropriate? |
| `overall_quality` | 3 | Holistic quality of the triage plan |

When in doubt between two candidates, prefer the one whose resolution behavior or component is more diagnostic for these dimensions.

---

## 6. Priority inference

`gold_priority` will be re-inferred at integration time using the same `infer_priority()` logic as the current gold set:
- Explicit `priority/` label on the issue → mapped to high/medium/low
- No explicit label: resolution < 24h → high; < 7d → medium; else → low

You don't need to assign priority during labeling. The `priority` column in the CSV shows any existing label if present.

---

## 7. Generating LLM triage plans for accepted issues

After selecting the final 90, run the triage pipeline to generate LLM plans (System 4) before the follow-up eval run:

```bash
# Step 1: clear the triage checkpoint so new issues get planned
python scripts/11_evaluate_triage.py \
  --repos microsoft/vscode kubernetes/kubernetes \
  --skip-judge \
  --n-samples 0
```

This is optional for the candidate review but required before the expanded eval can be scored by the judge. The pipeline will use the LLM response cache for any issues previously triaged.

---

## 8. Handing off the labeled file

When done:
1. Save the updated CSV as `data/gold_expansion_candidates_labeled.csv`
2. Confirm counts: `label_decision.value_counts()` should show ~90 `accept` across both repos with ~9 per bucket per repo
3. In a new session: "W5 labeling done — run the integration step"

The follow-up session reads `gold_expansion_candidates_labeled.csv`, validates the decisions, and runs `scripts/10_curate_triage_gold.py --extend` to produce the final n=150 gold set.

---

## 9. Expected effort

~2–3 hours for 120 issues at 1–2 minutes each. The pre-populated TF-IDF predictions and BGE similar-issue numbers help calibrate whether the issue is typical or anomalous for its component. Most rejections will be obvious (empty body, bot, trivial). Ambiguous cases get a quick check of the `body_excerpt`.
