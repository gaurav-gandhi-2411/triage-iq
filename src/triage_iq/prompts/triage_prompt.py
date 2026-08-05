"""Prompt template for the LLM triage assistant.

Builds a structured prompt from System 1–3 signals and requests a JSON
TriagePlan response. The schema is embedded in the prompt so the model
can be instructed to follow it without function-calling support.
"""



SYSTEM_PROMPT = """\
You are an expert GitHub issue triager for large open-source software projects.
You will be given an issue along with signals from automated classification and retrieval systems.
Your job is to produce a structured triage plan in valid JSON.

Output ONLY valid JSON matching the schema below. Do not include any prose before or after the JSON block.

PRIORITY GUIDELINES:
1. low — cosmetic or non-blocking; affects an edge case or niche workflow; trivial workaround exists; or long-standing minor annoyance.
2. medium — reproducible regression with a workaround available; feature gap affecting an active workflow; or visual inconsistency in a core feature.
3. high — crash, data loss, auth failure, or broken core workflow with NO workaround for any user.
4. Resource-leak / memory-leak framing does NOT automatically imply high. Assign medium unless the leak also causes a crash or completely blocks usage.
5. Empty or image-only body: assign priority based on the title alone. If the title is ambiguous and does not indicate a crash or data loss, default to medium.

CLASSIFIER CONFIDENCE GUIDANCE:
The component classifier in SYSTEM 1 reports independent per-component probabilities, not a single
normalized distribution, so several components scoring similarly is expected and does not mean the
classifier is unsure. Weigh it together with the issue text and the similar issues below, the same
way you always would — a close spread is additional context, not an instruction toward or away from
any particular entry. The resolution estimate and next steps are produced by separate, independent
models — do not soften or hedge them because the classifier's scores happen to be close together.

ATTRIBUTION RULES:
1. predicted_component should normally be one of the classifier's top-3 predictions. If you deviate, you MUST set component_source to "model_override" and explain why in component_override_reason.
2. Every issue number you cite anywhere in your plan MUST be one of the numbers listed in SYSTEM 2. Never invent issue numbers.
3. In declared_attribution, cite ONLY the SYSTEM-2 issues that actually support the specific claim. Citing every retrieved issue indiscriminately is wrong. Empty lists are honest when no retrieved issue supports the claim.

Schema:
{
  "predicted_component": "string — the single best component label for this issue",
  "component_confidence": "number 0.0–1.0 — your confidence in the component assignment",
  "similar_issues": [
    {
      "number": "integer — issue number",
      "similarity": "number 0.0–1.0 — semantic similarity score",
      "relevance_note": "string — one sentence on why this is related"
    }
  ],
  "expected_resolution_summary": "string — human-readable estimate (e.g., '2–7 days typical for this component')",
  "expected_resolution_lower_days": "number — optimistic estimate in days",
  "expected_resolution_upper_days": "number — conservative estimate in days",
  "priority_guess": "one of: low | medium | high",
  "priority_rationale": "string — 1–2 sentences explaining priority assignment",
  "suggested_assignee_class": "string — team or role best suited (e.g., 'core-runtime team', 'documentation team', 'first-time-contributor friendly')",
  "suggested_next_steps": ["string — ordered list of 2–4 actionable next steps"],
  "triage_summary": "string — 2–3 sentence executive summary of the issue and recommended action",
  "declared_attribution": {
    "component_source": "one of: classifier_top3 | model_override — whether predicted_component is taken from the classifier's top-3 list",
    "component_override_reason": "string — required when component_source is model_override; empty string otherwise",
    "summary_cited_issues": ["integer — issue numbers from SYSTEM 2 that directly support claims in triage_summary; [] if none"],
    "next_steps_cited_issues": ["integer — issue numbers from SYSTEM 2 that informed suggested_next_steps; [] if none"]
  }
}
"""


def build_triage_prompt(
    issue_title: str,
    issue_body: str,
    classifier_top3: list[dict],
    similar_issues: list[dict],
    resolution_point_days: float,
    resolution_lower_days: float,
    resolution_upper_days: float,
    repo: str,
    resolution_bucket: str | None = None,
    resolution_confidence_pct: float | None = None,
) -> str:
    """Build the user-turn prompt for the triage assistant.

    Args:
        issue_title: Issue title string.
        issue_body: Cleaned issue body (truncated to 800 chars).
        classifier_top3: Top-3 component predictions from TF-IDF classifier.
            Each dict has keys: label, confidence.
        similar_issues: Top-5 similar issues from BGE retrieval.
            Each dict has keys: number, score, text.
        resolution_point_days: Point estimate from predictor in days.
        resolution_lower_days: Q10 lower bound in days.
        resolution_upper_days: Q90 upper bound in days.
        repo: Repository slug (e.g., "microsoft/vscode").
        resolution_bucket: Optional coarse bucket (hours/days/weeks/months/long).
            When provided (Config C), appended to System 3 section alongside floats.
            When None (Config A, default), only float signals are shown.
        resolution_confidence_pct: Bucket confidence 0–100%. Used only when bucket provided.

    Returns:
        Formatted user-turn string.
    """
    body_preview = issue_body[:800].strip() if issue_body else "(no body)"

    classifier_lines = "\n".join(
        f"  {i+1}. {c['label']} (confidence: {c['confidence']:.3f})"
        for i, c in enumerate(classifier_top3)
    )

    similar_lines = "\n".join(
        f"  #{s['number']} (similarity: {s['score']:.3f}): {s['text'][:120]}..."
        for s in similar_issues[:5]
    )

    # Config C: optionally append coarse bucket alongside floats
    bucket_line = ""
    if resolution_bucket is not None:
        bucket_descs = {
            "hours": "< 1 day", "days": "1–7 days", "weeks": "1–4 weeks",
            "months": "1–6 months", "long": "> 6 months",
        }
        desc = bucket_descs.get(resolution_bucket, resolution_bucket)
        conf = f"(confidence: {resolution_confidence_pct:.0f}%)" if resolution_confidence_pct and resolution_confidence_pct >= 40 else "(low confidence)"
        bucket_line = f"\nBucket estimate: {resolution_bucket} ({desc}) {conf}"

    return f"""\
Repository: {repo}

--- ISSUE ---
Title: {issue_title}
Body:
{body_preview}

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
These are independent per-component probabilities, not a single normalized distribution: each score
answers "does this component apply?" on its own, so it is normal and expected for two or three
components to score similarly when an issue plausibly touches more than one area — this does not by
itself mean the classifier is unsure. Weigh these scores together with the issue text and the similar
issues below, the same way you always would.
Top-3 predictions:
{classifier_lines}

--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
{similar_lines}

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: {resolution_point_days:.1f} days
80% prediction interval: [{resolution_lower_days:.1f}d, {resolution_upper_days:.1f}d]{bucket_line}
Note: These estimates are trained on historical data and may not account for current team velocity or issue priority changes.

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
Use the classifier signals, similar issues, and resolution estimate to inform your plan.
Be specific and actionable. Do not hallucinate issue numbers not listed above.
"""


# ADR-0020: TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1 switches synthesis to the attribution-augmented
# SYSTEM_PROMPT / build_few_shot_examples() above (same env-var-gated pattern as
# TRIAGE_PROMPT_INCLUDE_BUCKET in triage.py). Off by default so eval/cassettes/eval_cassette.json
# (recorded pre-attribution) stays replayable without re-baselining reports/eval_baseline.json --
# see ADR-0020 "Baseline decision". The two constants below are a frozen, byte-exact snapshot of
# the pre-attribution prompt and are not meant to evolve alongside SYSTEM_PROMPT.
SYSTEM_PROMPT_LEGACY = """\
You are an expert GitHub issue triager for large open-source software projects.
You will be given an issue along with signals from automated classification and retrieval systems.
Your job is to produce a structured triage plan in valid JSON.

Output ONLY valid JSON matching the schema below. Do not include any prose before or after the JSON block.

PRIORITY GUIDELINES:
1. low — cosmetic or non-blocking; affects an edge case or niche workflow; trivial workaround exists; or long-standing minor annoyance.
2. medium — reproducible regression with a workaround available; feature gap affecting an active workflow; or visual inconsistency in a core feature.
3. high — crash, data loss, auth failure, or broken core workflow with NO workaround for any user.
4. Resource-leak / memory-leak framing does NOT automatically imply high. Assign medium unless the leak also causes a crash or completely blocks usage.
5. Empty or image-only body: assign priority based on the title alone. If the title is ambiguous and does not indicate a crash or data loss, default to medium.

CLASSIFIER CONFIDENCE GUIDANCE:
The component classifier in SYSTEM 1 reports independent per-component probabilities, not a single
normalized distribution, so several components scoring similarly is expected and does not mean the
classifier is unsure. Weigh it together with the issue text and the similar issues below, the same
way you always would — a close spread is additional context, not an instruction toward or away from
any particular entry. The resolution estimate and next steps are produced by separate, independent
models — do not soften or hedge them because the classifier's scores happen to be close together.

Schema:
{
  "predicted_component": "string — the single best component label for this issue",
  "component_confidence": "number 0.0–1.0 — your confidence in the component assignment",
  "similar_issues": [
    {
      "number": "integer — issue number",
      "similarity": "number 0.0–1.0 — semantic similarity score",
      "relevance_note": "string — one sentence on why this is related"
    }
  ],
  "expected_resolution_summary": "string — human-readable estimate (e.g., '2–7 days typical for this component')",
  "expected_resolution_lower_days": "number — optimistic estimate in days",
  "expected_resolution_upper_days": "number — conservative estimate in days",
  "priority_guess": "one of: low | medium | high",
  "priority_rationale": "string — 1–2 sentences explaining priority assignment",
  "suggested_assignee_class": "string — team or role best suited (e.g., 'core-runtime team', 'documentation team', 'first-time-contributor friendly')",
  "suggested_next_steps": ["string — ordered list of 2–4 actionable next steps"],
  "triage_summary": "string — 2–3 sentence executive summary of the issue and recommended action"
}
"""


def build_few_shot_examples_legacy() -> list[dict]:
    """Pre-attribution few-shot examples; see SYSTEM_PROMPT_LEGACY.

    The first three (low/medium/high) are frozen (ADR-0020) -- do not edit their content, only
    append after them. A fourth was appended for ADR-0037: those three were written against the
    single-label softmax classifier's confidence shape (one dominant score, two near-zero) and no
    longer demonstrate what the current multi-label OvR classifier normally outputs (three
    similarly-scored components). Editing them would violate the freeze; appending doesn't -- their
    bytes are untouched -- and gives the model an in-context precedent for "clustered scores, still
    decisive" instead of leaving that only as instruction text in SYSTEM_PROMPT_LEGACY, which
    in-context demonstrations tend to outweigh.
    """
    return [
        # --- LOW ---
        {
            "role": "user",
            "content": """\
Repository: microsoft/vscode

--- ISSUE ---
Title: Background workers not terminated when extension is unloaded on Linux
Body:
When an extension is disabled or uninstalled on Linux, background worker processes spawned by the extension keep running until VS Code itself is closed. Verified on VS Code 1.87.0, Ubuntu 22.04. The workers are eventually killed when VS Code exits, so no persistent leak occurs.

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
Top-3 predictions:
  1. extension-host (confidence: 0.341)
  2. api (confidence: 0.112)
  3. workbench (confidence: 0.058)

--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
  #8821 (similarity: 0.803): Child processes not cleaned up when extension deactivates...
  #7144 (similarity: 0.694): Extension worker threads linger after disable on Linux...

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: 18.7 days
80% prediction interval: [3.1d, 74.2d]

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
""",
        },
        {
            "role": "assistant",
            "content": """\
{
  "predicted_component": "extension-host",
  "component_confidence": 0.65,
  "similar_issues": [
    {"number": 8821, "similarity": 0.803, "relevance_note": "Same pattern of child processes surviving extension deactivation — likely identical root cause."},
    {"number": 7144, "similarity": 0.694, "relevance_note": "Worker thread lingering on Linux disable — same OS-level lifecycle gap."}
  ],
  "expected_resolution_summary": "Extension lifecycle cleanup on Linux. Requires platform-specific process management fix; typical 2–8 weeks.",
  "expected_resolution_lower_days": 3.1,
  "expected_resolution_upper_days": 74.2,
  "priority_guess": "low",
  "priority_rationale": "Despite resource-leak framing, this is low priority: the workers are reclaimed on VS Code exit (no persistent leak), the issue affects only extension authors on Linux, and VS Code remains fully usable. Resource leak framing alone does not make this high.",
  "suggested_assignee_class": "extension-host team",
  "suggested_next_steps": [
    "Confirm reproduction with a minimal extension that spawns a worker on activate.",
    "Check extension-host teardown path for explicit worker/process cleanup calls.",
    "Review whether #8821 is a duplicate; close if so."
  ],
  "triage_summary": "Background workers from disabled extensions linger until VS Code exits on Linux. No persistent resource leak and no user-facing breakage — VS Code remains fully functional. Low priority; assign to extension-host team for lifecycle cleanup."
}""",
        },
        # --- MEDIUM ---
        {
            "role": "user",
            "content": """\
Repository: microsoft/vscode

--- ISSUE ---
Title: Terminal: cursor blink stops after switching tabs
Body:
After switching between editor tabs multiple times, the integrated terminal cursor stops blinking. Reopening the terminal panel fixes it. Reproducible on 1.87.0.

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
Top-3 predictions:
  1. terminal (confidence: 0.821)
  2. editor (confidence: 0.054)
  3. workbench (confidence: 0.031)

--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
  #12345 (similarity: 0.891): Terminal cursor blink regression after 1.85 update...
  #11902 (similarity: 0.762): Integrated terminal: animation stops when switching focus...

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: 4.2 days
80% prediction interval: [0.8d, 18.3d]

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
""",
        },
        {
            "role": "assistant",
            "content": """\
{
  "predicted_component": "terminal",
  "component_confidence": 0.82,
  "similar_issues": [
    {"number": 12345, "similarity": 0.891, "relevance_note": "Direct regression of cursor blink behavior reported after a prior release, same reproduction pattern."},
    {"number": 11902, "similarity": 0.762, "relevance_note": "Focus-switch animation stoppage in terminal — same root cause area as tab-switching."}
  ],
  "expected_resolution_summary": "Typical terminal UI regression: 1–3 weeks depending on whether root cause is a CSS animation state bug or a deeper renderer issue.",
  "expected_resolution_lower_days": 0.8,
  "expected_resolution_upper_days": 18.3,
  "priority_guess": "medium",
  "priority_rationale": "Reproducible regression affecting a highly visible UX feature (cursor blink). Workaround exists (reopen panel), so not blocking, but degrades daily workflow.",
  "suggested_assignee_class": "terminal team",
  "suggested_next_steps": [
    "Confirm reproduction on latest Insiders build to rule out already-fixed regression.",
    "Check xterm.js animation state management when the terminal loses and regains focus.",
    "Add a regression test that switches tabs 5× and asserts cursor blink state."
  ],
  "triage_summary": "Reproducible terminal cursor blink regression triggered by tab switching. Two closely related prior issues exist (#12345, #11902), suggesting a known weak point in terminal focus handling. Assign to terminal team; medium priority given the available workaround."
}""",
        },
        # --- HIGH ---
        {
            "role": "user",
            "content": """\
Repository: microsoft/vscode

--- ISSUE ---
Title: VS Code crashes on opening any folder with .git directory after 1.85.0 upgrade
Body:
After upgrading to 1.85.0, VS Code crashes immediately on opening any workspace folder that contains a .git directory. The crash is 100% reproducible on both macOS and Windows. This affects all users who upgraded. Downgrading to 1.84.x is the only option — there is no workaround within 1.85.0.

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
Top-3 predictions:
  1. scm (confidence: 0.612)
  2. workbench (confidence: 0.143)
  3. git (confidence: 0.098)

--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
  #19201 (similarity: 0.931): VS Code crash on startup with git repo after update...
  #18877 (similarity: 0.814): Fatal error opening workspace with .git folder on 1.85...

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: 1.8 days
80% prediction interval: [0.3d, 6.1d]

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
""",
        },
        {
            "role": "assistant",
            "content": """\
{
  "predicted_component": "git",
  "component_confidence": 0.72,
  "similar_issues": [
    {"number": 19201, "similarity": 0.931, "relevance_note": "Same crash-on-startup regression tied to git repo detection after a version upgrade."},
    {"number": 18877, "similarity": 0.814, "relevance_note": "Identical reproduction: fatal error opening any workspace with .git folder on 1.85."}
  ],
  "expected_resolution_summary": "Critical regression requiring hotfix; typically resolved within 1–3 days for a P0 crash.",
  "expected_resolution_lower_days": 0.3,
  "expected_resolution_upper_days": 6.1,
  "priority_guess": "high",
  "priority_rationale": "Crash affecting all users on 1.85.0 when opening any git-backed workspace with no in-product workaround. This blocks the core workflow (opening a project) for every upgraded user.",
  "suggested_assignee_class": "git / scm team",
  "suggested_next_steps": [
    "Immediately bisect the 1.84→1.85 git extension changes to isolate the crashing commit.",
    "Publish a hotfix release (1.85.1) or Insiders rollback patch.",
    "Add a smoke test that opens a repo folder on each supported platform in CI."
  ],
  "triage_summary": "VS Code crashes on opening any folder with a .git directory after the 1.85.0 upgrade, affecting all users on both macOS and Windows with no workaround. Two highly similar prior reports confirm the regression. High priority; requires an immediate hotfix release."
}""",
        },
        # --- CLUSTERED CONFIDENCE (appended, not a frozen original -- see the module docstring
        # note below build_few_shot_examples_legacy() for why) ---
        {
            "role": "user",
            "content": """\
Repository: microsoft/vscode

--- ISSUE ---
Title: Settings Sync intermittently drops extension-specific settings after profile switch
Body:
When switching between user profiles with Settings Sync enabled, per-extension settings (e.g. formatter configuration, linter rules) occasionally fail to reapply after the switch completes. Reproducible roughly 1 in 5 switches on VS Code 1.88.0. Global (non-extension) settings sync correctly every time. Re-triggering sync manually usually fixes it.

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
These are independent per-component probabilities, not a single normalized distribution: each score
answers "does this component apply?" on its own, so it is normal and expected for two or three
components to score similarly when an issue plausibly touches more than one area — this does not by
itself mean the classifier is unsure. Weigh these scores together with the issue text and the similar
issues below, the same way you always would.
Top-3 predictions:
  1. settings-sync (confidence: 0.579)
  2. extensions (confidence: 0.531)
  3. profiles (confidence: 0.492)

--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
  #15210 (similarity: 0.742): Extension settings not restored after switching profiles on sync...
  #14889 (similarity: 0.681): Intermittent settings sync failure specific to extension-contributed keys...

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: 9.4 days
80% prediction interval: [2.8d, 21.6d]
Note: These estimates are trained on historical data and may not account for current team velocity or issue priority changes.

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
Use the classifier signals, similar issues, and resolution estimate to inform your plan.
Be specific and actionable. Do not hallucinate issue numbers not listed above.
""",
        },
        {
            "role": "assistant",
            "content": """\
{
  "predicted_component": "settings-sync",
  "component_confidence": 0.579,
  "similar_issues": [
    {"number": 15210, "similarity": 0.742, "relevance_note": "Same failure mode -- extension-contributed settings not restored after a profile switch."},
    {"number": 14889, "similarity": 0.681, "relevance_note": "Same intermittent pattern isolated to extension-contributed keys specifically, not global settings."}
  ],
  "expected_resolution_summary": "Cross-cutting sync timing bug between profile switch and extension settings restore; typically 1-3 weeks once the race condition is isolated.",
  "expected_resolution_lower_days": 2.8,
  "expected_resolution_upper_days": 21.6,
  "priority_guess": "medium",
  "priority_rationale": "Reproducible regression with real user impact (extension settings silently drop), but a manual re-sync works around it and the failure rate (~1 in 5) is partial, not universal.",
  "suggested_assignee_class": "settings-sync team",
  "suggested_next_steps": [
    "Reproduce with a minimal profile containing a single extension with custom settings, sync, then switch profiles 10x to isolate the race window.",
    "Check whether extension-contributed settings restore is awaited before the profile switch reports complete.",
    "Confirm #15210 and #14889 are duplicates of this or distinct manifestations of the same race."
  ],
  "triage_summary": "Settings Sync drops extension-specific settings on roughly 1 in 5 profile switches while global settings sync reliably every time, pointing to a timing race in the extension-settings restore path specifically. Two closely related prior reports confirm this is a recurring pattern, not a one-off. Assign to the settings-sync team; medium priority given the available workaround."
}""",
        },
    ]


def build_few_shot_examples() -> list[dict]:
    """Return 3 static few-shot examples (low / medium / high) for in-context learning."""
    return [
        # --- LOW ---
        {
            "role": "user",
            "content": """\
Repository: microsoft/vscode

--- ISSUE ---
Title: Background workers not terminated when extension is unloaded on Linux
Body:
When an extension is disabled or uninstalled on Linux, background worker processes spawned by the extension keep running until VS Code itself is closed. Verified on VS Code 1.87.0, Ubuntu 22.04. The workers are eventually killed when VS Code exits, so no persistent leak occurs.

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
Top-3 predictions:
  1. extension-host (confidence: 0.341)
  2. api (confidence: 0.112)
  3. workbench (confidence: 0.058)

--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
  #8821 (similarity: 0.803): Child processes not cleaned up when extension deactivates...
  #7144 (similarity: 0.694): Extension worker threads linger after disable on Linux...

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: 18.7 days
80% prediction interval: [3.1d, 74.2d]

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
""",
        },
        {
            "role": "assistant",
            "content": """\
{
  "predicted_component": "extension-host",
  "component_confidence": 0.65,
  "similar_issues": [
    {"number": 8821, "similarity": 0.803, "relevance_note": "Same pattern of child processes surviving extension deactivation — likely identical root cause."},
    {"number": 7144, "similarity": 0.694, "relevance_note": "Worker thread lingering on Linux disable — same OS-level lifecycle gap."}
  ],
  "expected_resolution_summary": "Extension lifecycle cleanup on Linux. Requires platform-specific process management fix; typical 2–8 weeks.",
  "expected_resolution_lower_days": 3.1,
  "expected_resolution_upper_days": 74.2,
  "priority_guess": "low",
  "priority_rationale": "Despite resource-leak framing, this is low priority: the workers are reclaimed on VS Code exit (no persistent leak), the issue affects only extension authors on Linux, and VS Code remains fully usable. Resource leak framing alone does not make this high.",
  "suggested_assignee_class": "extension-host team",
  "suggested_next_steps": [
    "Confirm reproduction with a minimal extension that spawns a worker on activate.",
    "Check extension-host teardown path for explicit worker/process cleanup calls.",
    "Review whether #8821 is a duplicate; close if so."
  ],
  "triage_summary": "Background workers from disabled extensions linger until VS Code exits on Linux. No persistent resource leak and no user-facing breakage — VS Code remains fully functional. Low priority; assign to extension-host team for lifecycle cleanup.",
  "declared_attribution": {"component_source": "classifier_top3", "component_override_reason": "", "summary_cited_issues": [8821, 7144], "next_steps_cited_issues": [8821]}
}""",
        },
        # --- MEDIUM ---
        {
            "role": "user",
            "content": """\
Repository: microsoft/vscode

--- ISSUE ---
Title: Terminal: cursor blink stops after switching tabs
Body:
After switching between editor tabs multiple times, the integrated terminal cursor stops blinking. Reopening the terminal panel fixes it. Reproducible on 1.87.0.

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
Top-3 predictions:
  1. terminal (confidence: 0.821)
  2. editor (confidence: 0.054)
  3. workbench (confidence: 0.031)

--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
  #12345 (similarity: 0.891): Terminal cursor blink regression after 1.85 update...
  #11902 (similarity: 0.762): Integrated terminal: animation stops when switching focus...

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: 4.2 days
80% prediction interval: [0.8d, 18.3d]

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
""",
        },
        {
            "role": "assistant",
            "content": """\
{
  "predicted_component": "terminal",
  "component_confidence": 0.82,
  "similar_issues": [
    {"number": 12345, "similarity": 0.891, "relevance_note": "Direct regression of cursor blink behavior reported after a prior release, same reproduction pattern."},
    {"number": 11902, "similarity": 0.762, "relevance_note": "Focus-switch animation stoppage in terminal — same root cause area as tab-switching."}
  ],
  "expected_resolution_summary": "Typical terminal UI regression: 1–3 weeks depending on whether root cause is a CSS animation state bug or a deeper renderer issue.",
  "expected_resolution_lower_days": 0.8,
  "expected_resolution_upper_days": 18.3,
  "priority_guess": "medium",
  "priority_rationale": "Reproducible regression affecting a highly visible UX feature (cursor blink). Workaround exists (reopen panel), so not blocking, but degrades daily workflow.",
  "suggested_assignee_class": "terminal team",
  "suggested_next_steps": [
    "Confirm reproduction on latest Insiders build to rule out already-fixed regression.",
    "Check xterm.js animation state management when the terminal loses and regains focus.",
    "Add a regression test that switches tabs 5× and asserts cursor blink state."
  ],
  "triage_summary": "Reproducible terminal cursor blink regression triggered by tab switching. Two closely related prior issues exist (#12345, #11902), suggesting a known weak point in terminal focus handling. Assign to terminal team; medium priority given the available workaround.",
  "declared_attribution": {"component_source": "classifier_top3", "component_override_reason": "", "summary_cited_issues": [12345, 11902], "next_steps_cited_issues": []}
}""",
        },
        # --- HIGH ---
        {
            "role": "user",
            "content": """\
Repository: microsoft/vscode

--- ISSUE ---
Title: VS Code crashes on opening any folder with .git directory after 1.85.0 upgrade
Body:
After upgrading to 1.85.0, VS Code crashes immediately on opening any workspace folder that contains a .git directory. The crash is 100% reproducible on both macOS and Windows. This affects all users who upgraded. Downgrading to 1.84.x is the only option — there is no workaround within 1.85.0.

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
Top-3 predictions:
  1. scm (confidence: 0.612)
  2. workbench (confidence: 0.143)
  3. git (confidence: 0.098)

--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
  #19201 (similarity: 0.931): VS Code crash on startup with git repo after update...
  #18877 (similarity: 0.814): Fatal error opening workspace with .git folder on 1.85...

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: 1.8 days
80% prediction interval: [0.3d, 6.1d]

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
""",
        },
        {
            "role": "assistant",
            "content": """\
{
  "predicted_component": "git",
  "component_confidence": 0.72,
  "similar_issues": [
    {"number": 19201, "similarity": 0.931, "relevance_note": "Same crash-on-startup regression tied to git repo detection after a version upgrade."},
    {"number": 18877, "similarity": 0.814, "relevance_note": "Identical reproduction: fatal error opening any workspace with .git folder on 1.85."}
  ],
  "expected_resolution_summary": "Critical regression requiring hotfix; typically resolved within 1–3 days for a P0 crash.",
  "expected_resolution_lower_days": 0.3,
  "expected_resolution_upper_days": 6.1,
  "priority_guess": "high",
  "priority_rationale": "Crash affecting all users on 1.85.0 when opening any git-backed workspace with no in-product workaround. This blocks the core workflow (opening a project) for every upgraded user.",
  "suggested_assignee_class": "git / scm team",
  "suggested_next_steps": [
    "Immediately bisect the 1.84→1.85 git extension changes to isolate the crashing commit.",
    "Publish a hotfix release (1.85.1) or Insiders rollback patch.",
    "Add a smoke test that opens a repo folder on each supported platform in CI."
  ],
  "triage_summary": "VS Code crashes on opening any folder with a .git directory after the 1.85.0 upgrade, affecting all users on both macOS and Windows with no workaround. Two highly similar prior reports confirm the regression. High priority; requires an immediate hotfix release.",
  "declared_attribution": {"component_source": "classifier_top3", "component_override_reason": "", "summary_cited_issues": [19201, 18877], "next_steps_cited_issues": []}
}""",
        },
        # --- CLUSTERED CONFIDENCE (appended, ADR-0037 -- see build_few_shot_examples_legacy()) ---
        {
            "role": "user",
            "content": """\
Repository: microsoft/vscode

--- ISSUE ---
Title: Settings Sync intermittently drops extension-specific settings after profile switch
Body:
When switching between user profiles with Settings Sync enabled, per-extension settings (e.g. formatter configuration, linter rules) occasionally fail to reapply after the switch completes. Reproducible roughly 1 in 5 switches on VS Code 1.88.0. Global (non-extension) settings sync correctly every time. Re-triggering sync manually usually fixes it.

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
These are independent per-component probabilities, not a single normalized distribution: each score
answers "does this component apply?" on its own, so it is normal and expected for two or three
components to score similarly when an issue plausibly touches more than one area — this does not by
itself mean the classifier is unsure. Weigh these scores together with the issue text and the similar
issues below, the same way you always would.
Top-3 predictions:
  1. settings-sync (confidence: 0.579)
  2. extensions (confidence: 0.531)
  3. profiles (confidence: 0.492)

--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
  #15210 (similarity: 0.742): Extension settings not restored after switching profiles on sync...
  #14889 (similarity: 0.681): Intermittent settings sync failure specific to extension-contributed keys...

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: 9.4 days
80% prediction interval: [2.8d, 21.6d]
Note: These estimates are trained on historical data and may not account for current team velocity or issue priority changes.

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
Use the classifier signals, similar issues, and resolution estimate to inform your plan.
Be specific and actionable. Do not hallucinate issue numbers not listed above.
""",
        },
        {
            "role": "assistant",
            "content": """\
{
  "predicted_component": "settings-sync",
  "component_confidence": 0.579,
  "similar_issues": [
    {"number": 15210, "similarity": 0.742, "relevance_note": "Same failure mode -- extension-contributed settings not restored after a profile switch."},
    {"number": 14889, "similarity": 0.681, "relevance_note": "Same intermittent pattern isolated to extension-contributed keys specifically, not global settings."}
  ],
  "expected_resolution_summary": "Cross-cutting sync timing bug between profile switch and extension settings restore; typically 1-3 weeks once the race condition is isolated.",
  "expected_resolution_lower_days": 2.8,
  "expected_resolution_upper_days": 21.6,
  "priority_guess": "medium",
  "priority_rationale": "Reproducible regression with real user impact (extension settings silently drop), but a manual re-sync works around it and the failure rate (~1 in 5) is partial, not universal.",
  "suggested_assignee_class": "settings-sync team",
  "suggested_next_steps": [
    "Reproduce with a minimal profile containing a single extension with custom settings, sync, then switch profiles 10x to isolate the race window.",
    "Check whether extension-contributed settings restore is awaited before the profile switch reports complete.",
    "Confirm #15210 and #14889 are duplicates of this or distinct manifestations of the same race."
  ],
  "triage_summary": "Settings Sync drops extension-specific settings on roughly 1 in 5 profile switches while global settings sync reliably every time, pointing to a timing race in the extension-settings restore path specifically. Two closely related prior reports confirm this is a recurring pattern, not a one-off. Assign to the settings-sync team; medium priority given the available workaround.",
  "declared_attribution": {"component_source": "classifier_top3", "component_override_reason": "", "summary_cited_issues": [15210, 14889], "next_steps_cited_issues": [15210, 14889]}
}""",
        },
    ]
