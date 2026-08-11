# Investigation: structural confidence representation (ADR-0037's untested lever)

**Date:** 2026-08-10
**Status:** Two candidates built and probed locally (zero Groq cost). **Neither shows a clean
enough local signal to recommend a confirming Groq recording right now.** Candidate 2 additionally
introduces a new, concrete failure mode (component-label fabrication) not present in the live
prompt. No live Groq calls made; no cassette re-recorded; no baseline touched.
**Scope:** `src/triage_iq/prompts/triage_prompt.py` (not modified — both candidates live only in
the new probe script, see §5); `scripts/probe_confidence_structural_variants.py` (new).
Reproduce: `python scripts/probe_confidence_structural_variants.py` (requires local Ollama with
`llama3.1:8b` + `qwen3:8b` pulled, and `data/models/*kubernetes_kubernetes*` +
`data/processed/kubernetes_kubernetes_temporal_train.parquet` present — gitignored, not in this
worktree by default, copied in locally for this investigation).

---

## 1. Context — why this lever, and why now

ADR-0037 diagnosed the ADR-0036 multi-label classifier's tightly-clustered per-class confidence
scores (e.g. `0.546 / 0.459 / 0.448`) as a trigger for LLM-synthesis hedging: the judge reads a
close spread as "the model is unsure" and writes hedged prose across dimensions fed by completely
unrelated, unchanged models (`resolution_estimate_reasonableness`, `next_steps_actionability`).
Four prompt-**wording** variants (v1–v3, see ADR-0037) tried to talk the LLM out of that reading —
none closed the gap; v3 (live today) is a partial, incomplete fix. ADR-0043 later showed ~61% of
the *original* regression was actually upstream-signal quality (retrieval, resolution model), now
fixed — but the remaining ~39% is confirmed still attributable to confidence-framing/hedging and
explicitly not addressed by any wording change tried so far.

ADR-0037's own Consequences section named the untested lever: not a fifth wording variant, but a
**structural** change to how confidence is represented — a different transform, categorical bands,
or showing less of it. This investigation builds and locally probes two structural candidates.

## 2. Why banding was not attempted (candidate 1's "no numbers" route was chosen instead)

The task allowed either "no numbers at all" or "qualitative bands" for candidate 1. Banding was
rejected as under-justified, not merely skipped:

- `eval/test_invariants.py::_compute_ece` bins by top-1 confidence into 5 **equal-width**
  `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` bins for a reliability diagram — a calibration-quality
  diagnostic, not a decision threshold. Reusing those edges for "high/moderate/low" would put
  nearly the entire live confidence distribution (ADR-0037's own examples cluster
  `0.44–0.65`) inside a single bin (`[0.4, 0.6)`), collapsing the 3-way band to 1-way for the
  common case — the opposite of useful.
- `src/triage_iq/models/abstention.py::COMPONENT_CONFIDENCE_THRESHOLD` (k8s 0.45, vscode 0.29) is
  the only other threshold in the codebase near this shape. Its own comment marks it **STALE as of
  ADR-0036**: tuned to the old single-label softmax classifier, and confirmed (in that same
  comment, citing `reports/tfidf_multilabel_calibration_and_threshold_check.json`) to fire at a
  "wildly different rate" under the new multi-label classifier — k8s 59.8%→0.0%. Reusing it for a
  *display* band would be worse than reusing it for its original gating purpose: at least gating
  is a binary yes/no where "fires almost never" fails safe; a 3-way display band built on the same
  stale cutoffs would silently mislabel most predictions into whichever band happens to catch the
  compressed 0.44–0.65 range.

No other calibration machinery in `component_classifier.py` (temperature scaling only, a single
scalar dividing logits — recalibrates confidence values, not confidence *categories*) offered a
principled 3-way cutoform. Rather than invent one, candidate 1 uses a rank-ordered list with no
numbers at all — the option the task brief offered as the alternative, and the one that doesn't
require picking a defensible threshold from a distribution that doesn't cleanly support one.

## 3. The two candidates — exact prompt text

Both candidates modify only the **SYSTEM 1** section of `build_triage_prompt()`'s user turn and the
`CLASSIFIER CONFIDENCE GUIDANCE` paragraph of `SYSTEM_PROMPT_LEGACY` (the confirmed-live prompt,
per ADR-0037). `classifier_top3` — the actual data structure passed to `TriageAssistant` /
`verify_plan_grounding()` — is **unchanged** in both candidates; only the LLM-facing prompt *text*
differs, exactly as the task brief required. Full source: `scripts/probe_confidence_structural_variants.py`.

**Baseline (v3, live today):**
```
CLASSIFIER CONFIDENCE GUIDANCE:
The component classifier in SYSTEM 1 reports independent per-component probabilities, not a single
normalized distribution, so several components scoring similarly is expected and does not mean the
classifier is unsure. Weigh it together with the issue text and the similar issues below, the same
way you always would — a close spread is additional context, not an instruction toward or away from
any particular entry. [...]

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
These are independent per-component probabilities, not a single normalized distribution: [...]
Top-3 predictions:
  1. kube-proxy (confidence: 0.575)
  2. test (confidence: 0.468)
  3. test-infra (confidence: 0.468)
```

**Candidate 1 — rank-ordered list, no numbers:**
```
CLASSIFIER CONFIDENCE GUIDANCE:
The component classifier in SYSTEM 1 reports a ranked list of candidate components, most to least
likely, with no numeric scores shown. Treat #1 as the classifier's lead assessment. [...]

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
Ranked component predictions (most to least likely), from an automated classifier. Rank order is
the classifier's signal here — treat #1 as its lead assessment.
  1. kube-proxy
  2. test
  3. test-infra
```

**Candidate 2 — top-1 + single calibrated confidence only:**
```
CLASSIFIER CONFIDENCE GUIDANCE:
The component classifier in SYSTEM 1 reports its single top prediction with a calibrated confidence
score. Weigh it together with the issue text and the similar issues below, the same way you always
would. [...]

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
Predicted component (automated classifier): kube-proxy (confidence: 0.575)
```

The appended 4th few-shot example (ADR-0037's "clustered confidence" demonstration, not one of the
ADR-0020-frozen three) was rewritten per candidate to match its SYSTEM 1 format — mirroring exactly
what ADR-0037's v3 iteration did when it last changed this format. The three frozen LOW/MEDIUM/HIGH
examples are reused byte-identical across all three variants, unedited (ADR-0020 freeze) — this
means, in both candidates, the model still sees three demonstrations in the *old* scored-top-3
format before seeing the new format in the 4th example and the live task turn. This mismatch is a
real property of what shipping either candidate would look like today, not a probe artifact, and is
called out here rather than smoothed over.

## 4. Method

`scripts/probe_confidence_structural_variants.py` (new file). Zero Groq cost: local Ollama only —
`llama3.1:8b` for synthesis, `qwen3:8b` for judging (same judge model production/CI use, ADR-0019),
both `temperature=0.0`, `seed=42`. Same load-bearing caveat as
`scripts/probe_prompt_structure_local.py`: **this is a cheap filter, not a validation gate** —
ADR-0037's v3 iteration showed a clean local "go" signal that inverted on the real 64-issue Groq
recording. This probe tests a structural (not wording) question, which is a different, more
mechanical hypothesis class than v3's — but that is a reason to trust this result *slightly* more,
not a proven exemption from the same local/production gap.

**Subset:** 12 of the 24 k8s issues in `probe_prompt_structure_local.py::TARGET_ISSUE_IDS` — chosen
over the full 24 because 3 variants × 24 issues × 2 calls = 144 local calls was a meaningful time
cost for a filtering step; 12 was chosen to include every issue ADR-0037 named specifically
(`k8s-13270`, `k8s-14756` — the two hedging examples; `k8s-14281` — the worst score drop/near-tie
flip; `k8s-12703`, `k8s-14363` — flips that recovered under v3; `k8s-14895` — the confirmed
classifier-error case) plus 6 more for general spread (`k8s-12224`, `k8s-12665`, `k8s-12828`,
`k8s-13435`, `k8s-14135`, `k8s-14711`). Each (issue, variant) pair: 1 synthesis call + 1 judge call
+ `verify_plan_grounding()` (pure Python, using the real unedited `classifier_top3`, independent of
what the LLM was shown).

## 5. Results

Full log: `reports/probe_confidence_structural_variants_result.json`. Raw per-variant means as
reported by the script (**note baseline_v3 is n=11**, not 12 — see §5.3):

| metric | baseline_v3 (n=11) | candidate1_no_numbers (n=12) | candidate2_top1_only (n=12) |
|---|---|---|---|
| mean_total (/15) | 9.727 | 9.583 | 9.500 |
| resolution_estimate_reasonableness | 1.182 | 1.167 | 1.250 |
| next_steps_actionability | 2.182 | 2.167 | 2.167 |
| component_match | 1.091 | 1.083 | 1.000 |
| grounding_pass_rate | **1.000** | **1.000** | **0.917** |
| issues with hedge-phrase hits (rationale text) | 1/11 | 2/12 | 3/12 |

### 5.1 Apples-to-apples (same 11 issues, excluding k8s-14711 which failed to parse under baseline)

Summed judge totals over the 11 issues every variant scored:

| variant | sum /165 max | mean |
|---|---|---|
| baseline_v3 | 107 | 9.727 |
| candidate1_no_numbers | 107 | 9.727 |
| candidate2_top1_only | 106 | 9.636 |

**Candidate 1 is an exact tie with baseline on total judge score at this n** (107=107) — individual
issues moved in both directions and cancelled out (baseline scored k8s-13270 3 points higher,
candidate1 scored k8s-12828 4 points higher, k8s-13435 1 point lower — net zero). Candidate 2 is
0.7% lower, within noise at n=11. **Read: no candidate shows a total-score win over the live
prompt at this sample size** — the differences are inside a coin-flip's worth of judge-scoring
jitter (ADR-0019 measured qwen3:8b judge std=0.748/15 on repeated scoring of identical input).

### 5.2 Candidate 1 (no numbers) — hedge language, closely read

Two issues flagged by the crude hedge-phrase detector (`"imprecise"`, `"lacks confidence"`, etc. in
`judge_rationale`), one more than baseline's one. Reading the actual rationale text
(`scripts/_probe_rationale_dump.py` output, reproduced verbatim):

- **k8s-14756**, baseline: *"the resolution estimate is imprecise and lacks confidence, which
  slightly lowers the overall quality."* Candidate 1: *"the resolution estimate is imprecise and
  spans a wide range, which reduces its reasonableness."* Both hedge on the same dimension with the
  same score (`res=1` both). The wording differs but the underlying critique (the 0.1–138.6 day
  interval is too wide to be useful) is the *same complaint under both prompts* — this is not new
  hedging introduced by removing numbers, it's a pre-existing resolution-interval-width problem
  (separately logged in ADR-0037's "Separately logged, not fixed here" section) that both prompts
  surface identically.
- **k8s-12224** (new hedge hit under candidate1, not present under baseline): rationale says *"the
  resolution estimate is imprecise and the priority is misaligned with the gold standard."*
  `component_match=2` (correct) and `next_steps=2` both variants; the score didn't change
  (baseline judge=10, candidate1 judge=10 — verified in §5.1's per-issue baseline log, both scored
  10/15) — only the rationale's phrasing crossed the crude word-match filter. **This is a false
  positive in the hedge-phrase detector, not a real behavior difference** — a reminder that a
  1-2 issue swing in a keyword-matched count at n=12 is not a reliable signal either direction.

**Candidate 1 verdict: a wash.** Tied on total score, no grounding regression (12/12, matching
baseline's 11/11), and the "more hedging" signal evaporates on close reading of the actual text —
one case is a pre-existing shared problem, the other is a keyword-match false positive with no
score change behind it.

### 5.3 Candidate 2 (top-1 only) — the fabrication finding

The task brief specifically flagged this risk to check: *"a structural change that silently breaks
grounding is worse than the problem it's meant to fix."* It happened, once, in 12 issues:

**k8s-14363.** Actual `classifier_top3` (real classifier output, unaffected by prompt text):
`kubectl (0.490) / usability (0.478) / cloudprovider (0.458)`. Baseline_v3 (shown all three)
predicted `cloudprovider` — grounded, and scored 14/15 ("excellent... accurate component
labeling"). Candidate 2 (shown only `kubectl (confidence: 0.490)` as top-1) predicted
**`kubernetes-provider`** — a label that does not exist anywhere in `classifier_top3` and was never
shown to the model. `verify_plan_grounding()` correctly flagged it: `component_grounded=False`.
The judge, working only from the plan text with no visibility into grounding, still scored it
12/15 and called the labeling "accurate" — the judge cannot see this failure at all; only the
pipeline's own grounding check (deterministic Python, independent of the LLM) catches it. This
confirms the caveat from the task brief in the opposite direction from how it's usually framed:
grounding *verification* still works correctly (it did its job — it's a pure function over the real
`classifier_top3`, unaffected by what candidate 2 shows the LLM), but the underlying *plan quality*
degraded in a way judge-score alone would have missed entirely.

Mechanism read: with only one label and no visible alternates, the model appears to
free-associate a plausible-sounding paraphrase ("kubernetes-provider" for an issue about the
vSphere cloud provider) rather than staying anchored to the classifier's literal output — exactly
the opposite of what removing the "several plausible" framing was meant to achieve. Two more hedge
hits also appeared under candidate 2 (k8s-12703 already hedges under all three variants per the
component_match=0 pattern discussed below; k8s-12665's rationale: *"the resolution estimate is in
the right ballpark but imprecise"* — same shared pre-existing pattern as §5.2, not new).

component_match also drops slightly (1.000 vs 1.091 baseline) — consistent with, though not fully
explained by, the one fabricated-label case (which the judge still scored comp=1 on, since it
judged the *plan text* as reasonable even though grounding failed structurally).

**Candidate 2 verdict: not a wash — actively worse on a structural-integrity axis the other two
variants don't have.** Total judge score is statistically indistinguishable from baseline (§5.1),
but it introduces a new failure mode (component-label fabrication) that baseline and candidate 1
did not produce even once across the same 12 issues.

## 6. Recommendation

**Neither candidate is recommended for the confirming Groq spend right now.**

- **Candidate 1 (no numbers):** a clean structural wash at this n — tied total score, zero
  grounding regression, no genuine new hedging behavior on close reading. It is *not* actively
  bad, and is the safer of the two to revisit later, but it also doesn't clear the bar of "clearly
  better than v3 locally" that would justify spending ~215K Groq tokens to confirm. Given
  ADR-0037's own documented case of a clean local signal (v3's) inverting at full Groq scale, a
  merely-tied local signal is well below the threshold where spending the confirmation budget is
  worth it.
- **Candidate 2 (top-1 only):** actively not recommended. Independent of the judge-score tie, it
  introduced a concrete, observed label-fabrication failure the other two variants did not, at
  zero-Groq cost, on a 12-issue sample — exactly the risk the task brief asked to be checked for.
  Shipping this without a design change (e.g., forcing the model to only ever emit the shown label
  verbatim, which would need output-schema enforcement beyond what this prompt-only lever can do)
  would be trading one problem (hedging) for a worse one (fabricated attribution) on the one
  dimension (`component_grounded`) the pipeline treats as a hard trust boundary.

**No further prompt-lever work is recommended from this investigation** unless a new diagnostic
finding changes the picture — mirroring ADR-0037's own closing call after four wording variants.
The "untested lever" ADR-0037 flagged is now tested: at local-probe fidelity, restructuring *how
much* confidence information is shown does not produce a clear win, and in the top-1-only case
produces a new, worse failure. This does not prove no structural fix exists — a probe at n=12 on
one local model is a narrow instrument — but it does not clear the bar to spend real budget
confirming it either.

## Appendix: files

- `scripts/probe_confidence_structural_variants.py` — the probe (3 variants × 12 issues, committed).
- `scripts/_probe_rationale_dump.py` — one-off helper used to pull full rationale text for the 8
  (issue, variant) pairs quoted in §5.2/§5.3 (committed for reproducibility, not part of the eval
  suite).
- `reports/probe_confidence_structural_variants_result.json` — machine-readable summary (committed).
- Raw run logs (not committed, reproducible via the scripts above): full 12×3 run and the targeted
  rationale dump.
