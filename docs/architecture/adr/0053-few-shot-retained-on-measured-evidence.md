# ADR-0053: Few-shot examples retained on measured evidence, not precedent

Status: Accepted -- **basis confounded, re-measurement needed before this decision can
be trusted on its stated evidence; decision NOT reopened, reliability question flagged
for a future session.** See ADR-0055: Arm B's 3 failures (the entire basis for this
ADR) were examined directly and show the identical schema-field-omission pattern
ADR-0055 fixes -- not confirmed to be specific to the no-few-shot condition. ADR-0055's
fix was NOT re-tested against a no-few-shot arm (out of scope, explicitly not
reopened). Re-measuring costs roughly a full Arm B re-run (n=8, matching the original,
~0.3-0.4 days of quota at the ~20-25 calls/day pace observed elsewhere in this
engagement; n=20 for a comparison as robust as Arm A's original, ~1 day) under the
reduced schema. Until re-measured, treat this ADR's few-shot-is-necessary conclusion
as unconfirmed, not false -- the original observation (95% vs 62.5%) is real and
unretracted; only its causal attribution (few-shot vs. schema) is now in question.
Date: 2026-08-30

## Context

ADR-0020 and ADR-0037 established few-shot examples in the triage synthesis prompt as
precedent — content that demonstrably improved judge-scored quality and calibrated the
model to the classifier's clustered-confidence behavior, at the time those decisions were
made. Neither measured what happens to the model's ability to **complete the output
schema at all** when few-shot is removed; both treated few-shot purely as a content/style
lever.

The 2026-08-29/30 model/prompt bake-off (`docs/eval/bakeoff_prereg_2026-08-29.md`)
pre-registered a no-few-shot arm (Arm B) specifically to test whether stripping few-shot
lowers cost without hurting quality. It did not survive the pre-registered elimination
rule, and the failure mode it produced is a different, more fundamental finding than the
prereg anticipated.

## Decision

**Few-shot examples are retained in the default synthesis prompt, on measured evidence
from this bake-off, not solely on ADR-0020/0037's original precedent.**

Same model (`openai/gpt-oss-20b`), same system prompt (`SYSTEM_PROMPT_PROSE`), only the
few-shot set changed:

| Condition | Parse-success rate | n |
|---|---:|---:|
| With few-shot (Arm A) | 19/20 (95%) | 20 |
| Without few-shot (Arm B) | 5/8 (62.5%) — eliminated | 8 |

Arm B was eliminated per the pre-registration's own rule (§3: *"Parse-success rate below
90%"*) — 3 failures at n=8 makes a 90%-of-20 floor mathematically unreachable even with a
perfect remaining 12/12. All 4 observed failures across both arms (3 in Arm B, 1 in Arm
A's #12254) share one failure shape: the completion is syntactically valid JSON up to the
point it stops, then ends early — missing anywhere from 5 to 15 of 18 required schema
fields. Not malformed syntax, not a Groq-side content rejection: the model's own
generation stopped (emitted an end token) before satisfying the schema, something Groq's
post-hoc `required` validation then correctly catches as a 400.

**This is evidence that few-shot examples are carrying format compliance — teaching the
model to traverse the entire schema before stopping — not just content/style quality.**
That is a stronger, more load-bearing role than either ADR-0020 or ADR-0037 credited them
with, and it is the pre-registered condition for keeping few-shot (bake-off prereg §4:
*"the only valid reason to prefer [no few-shot] is that [it] scores equal to or better,
measured fairly"* — it does not, on the more basic prior gate of finishing the output at
all).

## Consequences

- **What changes:** Few-shot's justification moves from "ADR-0020/0037 decided this" to
  "measured directly, 2026-08-30, n=20 vs n=8, mechanically decisive via the
  pre-registered elimination rule." Future prompt-shrinking proposals must budget for
  this reliability cost explicitly, not just a token-cost tradeoff.
- **What becomes harder — stated plainly, this is a finding about the tier, not a
  license to drop few-shot:** `gpt-oss-20b` **with** few-shot does not fit the Groq free
  tier with headroom. Real usage this session: 168,970–182,338 tokens consumed against a
  documented 200,000 daily/rolling ceiling by roughly 25 successful calls before hitting
  `RateLimitError`, and the same wall recurred again partway through a later 20-call
  re-run (9/20 succeeded before exhausting retries again). The corrected fix is **not**
  "drop few-shot to fit" — that reopens exactly the reliability gap this ADR documents.
  The tier constraint is a separate, still-open problem (see the bake-off's Part C token-
  budget-margin findings for the mechanism-level discussion).
- **What becomes easier:** The no-few-shot question is closed for this model/prompt
  combination — no need to re-litigate it without new evidence (e.g., a different base
  model, or a schema simplification that removes the fields most often missing).

## Alternatives considered

- **Keep treating few-shot as a pure cost/quality tradeoff, decide via judge-mean
  alone.** Rejected: the judge-mean comparison never got the chance to run for Arm B at
  scale — the arm failed a more fundamental, prior gate (finishing the schema) before
  quality could even be assessed on the failing 37.5% of calls. Judging on judge-mean
  alone would have silently thrown away information about a real reliability defect.
- **Treat the 3 Arm-B failures as noise and re-run for more data before deciding.**
  Rejected: the pre-registered elimination rule exists precisely so a result like this
  doesn't get re-litigated after the fact once it's already mathematically decisive (a
  90%-of-20 floor cannot be reached from 5/8). Re-running would spend real, scarce
  free-tier quota to re-confirm an outcome that cannot change.
- **Fix the underlying schema-completion defect (widen `_call_llm_verbose`'s exception
  handling to catch this failure shape and retry/degrade gracefully) instead of an ADR.**
  Not rejected, but out of scope for this ADR — flagged as a real, separate reliability
  gap (the existing fallback branch only matches a literal `"response_format"` substring
  in the error, which this failure shape doesn't contain, so it currently propagates
  uncaught). Worth a follow-up PR; this ADR records the few-shot decision the bake-off
  data supports today, independent of whether that defect gets fixed later.
