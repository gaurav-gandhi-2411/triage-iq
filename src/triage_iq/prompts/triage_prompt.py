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
  "expected_resolution_summary": "string — human-readable resolution estimate, e.g. 'Typically resolves within days for this component; similar issues closed in 2-5 days'",
  "resolution_bucket": "one of: hours | days | weeks | months | long — your estimate of resolution time bucket",
  "resolution_confidence_pct": "number 0-100 — your confidence in the resolution_bucket estimate",
  "priority_guess": "one of: low | medium | high",
  "priority_rationale": "string — 1–2 sentences explaining priority assignment",
  "suggested_assignee_class": "string — team or role best suited (e.g., 'core-runtime team', 'documentation team', 'first-time-contributor friendly')",
  "suggested_next_steps": ["string — ordered list of 2–4 actionable next steps"],
  "triage_summary": "string — 2–3 sentence executive summary of the issue and recommended action"
}
"""


def build_triage_prompt(
    issue_title: str,
    issue_body: str,
    classifier_top3: list[dict],
    similar_issues: list[dict],
    resolution_bucket: str,
    resolution_confidence_pct: float,
    repo: str,
) -> str:
    """Build the user-turn prompt for the triage assistant.

    Args:
        issue_title: Issue title string.
        issue_body: Cleaned issue body (truncated to 800 chars).
        classifier_top3: Top-3 component predictions from TF-IDF classifier.
            Each dict has keys: label, confidence.
        similar_issues: Top-5 similar issues from BGE retrieval.
            Each dict has keys: number, score, text.
        resolution_bucket: Coarse bucket from ordinal classifier
            (hours/days/weeks/months/long).
        resolution_confidence_pct: Model confidence 0–100%.
        repo: Repository slug (e.g., "microsoft/vscode").

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

    bucket_desc = {
        "hours":  "< 1 day",
        "days":   "1–7 days",
        "weeks":  "1–4 weeks",
        "months": "1–6 months",
        "long":   "> 6 months",
    }.get(resolution_bucket, resolution_bucket)

    conf_note = (
        "(low confidence — use as a rough prior only)"
        if resolution_confidence_pct < 40
        else f"(model confidence: {resolution_confidence_pct:.0f}%)"
    )

    return f"""\
Repository: {repo}

--- ISSUE ---
Title: {issue_title}
Body:
{body_preview}

--- SYSTEM 1: COMPONENT CLASSIFIER (TF-IDF) ---
Top-3 predictions:
{classifier_lines}

--- SYSTEM 2: SIMILAR ISSUES (BGE retrieval) ---
{similar_lines}

--- SYSTEM 3: RESOLUTION TIME PREDICTOR ---
Estimated bucket: {resolution_bucket} ({bucket_desc}) {conf_note}
Note: Based on historical issue data. Use similar issues and component patterns to refine.

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
Use the classifier signals, similar issues, and resolution estimate to inform your plan.
Be specific and actionable. Do not hallucinate issue numbers not listed above.
"""


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

--- SYSTEM 3: RESOLUTION TIME PREDICTOR ---
Estimated bucket: weeks (1–4 weeks) (model confidence: 52%)

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
  "resolution_bucket": "weeks",
  "resolution_confidence_pct": 52,
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

--- SYSTEM 3: RESOLUTION TIME PREDICTOR ---
Estimated bucket: days (1–7 days) (model confidence: 61%)

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
  "resolution_bucket": "days",
  "resolution_confidence_pct": 61,
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

--- SYSTEM 3: RESOLUTION TIME PREDICTOR ---
Estimated bucket: hours (< 1 day) (model confidence: 71%)

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
  "resolution_bucket": "hours",
  "resolution_confidence_pct": 71,
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
    ]
