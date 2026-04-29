"""Prompt template for the LLM triage assistant.

Builds a structured prompt from System 1–3 signals and requests a JSON
TriagePlan response. The schema is embedded in the prompt so the model
can be instructed to follow it without function-calling support.
"""

from typing import Optional


SYSTEM_PROMPT = """\
You are an expert GitHub issue triager for large open-source software projects.
You will be given an issue along with signals from automated classification and retrieval systems.
Your job is to produce a structured triage plan in valid JSON.

Output ONLY valid JSON matching the schema below. Do not include any prose before or after the JSON block.

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


def build_triage_prompt(
    issue_title: str,
    issue_body: str,
    classifier_top3: list[dict],
    similar_issues: list[dict],
    resolution_point_days: float,
    resolution_lower_days: float,
    resolution_upper_days: float,
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
        resolution_point_days: Point estimate from LightGBM predictor in days.
        resolution_lower_days: Q10 lower bound in days.
        resolution_upper_days: Q90 upper bound in days.
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

--- SYSTEM 3: RESOLUTION TIME PREDICTOR (LightGBM) ---
Point estimate: {resolution_point_days:.1f} days
80% prediction interval: [{resolution_lower_days:.1f}d, {resolution_upper_days:.1f}d]
Note: These estimates are trained on historical data and may not account for current team velocity or issue priority changes.

--- TASK ---
Produce a triage plan as valid JSON matching the schema in the system prompt.
Use the classifier signals, similar issues, and resolution estimate to inform your plan.
Be specific and actionable. Do not hallucinate issue numbers not listed above.
"""


def build_few_shot_examples() -> list[dict]:
    """Return 2 static few-shot examples for in-context learning."""
    return [
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
    ]
