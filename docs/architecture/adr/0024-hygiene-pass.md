# ADR-0024 — Hygiene Pass: Closing Standing Loose Ends

**Status:** Accepted (items 1–4, closed) / Escalated (item 5, investigate-and-report only)
**Date:** 2026-07-10
**Decider:** Gaurav Gandhi

## Context

Across the eval + applied-AI arc (ADR-0015 through ADR-0023), several non-blocking loose ends
accumulated: a chronically-red security-audit job, a workflow file with a YAML parse error, a
stale scheduling artifact, and an open cross-project resource-contention question. This ADR
closes each one — fixed, honestly-suppressed-with-justification, or explicitly escalated —
before the next (model-performance) phase.

**Merge-state confirmed first, per instructions:** `feat/attribution-fidelity` (PR #18),
`feat/selective-prediction` (PR #19), and `feat/resolution-diagnosticity` (PR #20) are merged to
`main`. `feat/structured-verification` (ADR-0022, the semantic consistency verifier) is **not**
merged — it was never even pushed to `origin`, still local-only. This hygiene pass runs against
`main`'s actual current state, which does not yet include ADR-0022.

## Item 3 — Stale `/loop` wakeup

Investigated both scheduling mechanisms available: `CronList` (session-scoped cron jobs) and
`RemoteTrigger` list (cloud-hosted scheduled routines) — both returned empty, no jobs registered
either way. Found `.claude/scheduled_tasks.lock` in the project directory; verified via process
inspection (`Get-CimInstance Win32_Process`) that its PID is the live parent process of this very
session (not an orphaned/stale process from an old run) — did **not** touch it, killing it would
have terminated the active session.

The actual stale trigger was a pending dynamic-loop wakeup schedule from earlier in this session
(the `ScheduleWakeup` mechanism, which has no listing API). Called it with `stop: true`, which
confirmed a pending schedule existed ("Loop stopped — no further wakeups scheduled") and cleared
it. No `Monitor` was ever armed in this session (never used), so nothing further to stop there.
**Confirmed: no scheduled trigger remains that would auto-resume work.**

## Item 4 — Untracked scratch files

`git status --short` (and `--ignored`) on `main` before starting this branch showed a fully clean
working tree — no untracked files. The `*.attempt1`/`*.bak60`/`*.old60` files from the
attribution-fidelity arc were already cleaned up in that earlier session. Nothing to resolve here.

## Item 2 — `record-cassette.yml` parse error

**Root cause, confirmed via PyYAML, not assumed:** the "Clear cassette and checkpoint" step
embedded a Python heredoc (`python - <<'EOF' ... EOF`) inside a YAML `run: |` block scalar. The
heredoc body was written at column 0 (no indentation), which is *less* indented than the block
scalar's established level (10 spaces, set by the first content line) — this terminates the YAML
block early and the parser then tries to read `import json` as a new top-level mapping key,
failing with `could not find expected ':'` at line 83. This explains the observed symptom
exactly: GitHub creates a failed run entry on every push (event: push) even though the workflow
declares `on: workflow_dispatch:` only, because a YAML parse error prevents GitHub from ever
correctly registering the trigger.

**Fix:** replaced the fragile multi-line heredoc with two single-line `python3 -c "..."` calls —
eliminates the entire class of bug (no embedded multi-line block whose indentation must be
reconciled with YAML's own rules), not just this instance. Verified: `yaml.safe_load` parses
cleanly, `on:` resolves to `{'workflow_dispatch': None}` (confirming no push trigger), job
structure unchanged (12 steps, same names/order), and the two `python3 -c` calls produce
byte-identical JSON content to what the original script wrote.

**"Dry validation," not a live dispatch:** did not call `gh workflow run` — an actual dispatch
would trigger the real ~20-minute job (live Groq API calls, GCP secret access, a real git push
from CI), which is not "zero-cost" and not what "dry" should mean. Validation here means
syntactic (PyYAML) + structural (step count/names/order) confirmation. `actionlint` (a stronger,
GH-Actions-schema-aware validator) is not installed locally and installing it would add a new
dependency outside this hygiene pass's scope — noted as a gap, not silently worked around.

## Item 1 — `pip-audit` chronically red

### Starting state: 25 vulnerabilities across 7 packages

`pip-audit -r requirements.lock --skip-editable` (matching `ci.yml`'s exact invocation) found 25
CVEs: 11 in `aiohttp`, 2 in `idna` (duplicate advisory IDs for the same finding), 6 in
`starlette`, 1 in `pyasn1`, 1 in `pydantic-settings`, 3 in `transformers` (1 already suppressed
as CVE-2026-1839, 2 new), 1 in `torch`.

### Per-CVE triage

**Fixed (22 CVEs) — surgical `requirements.lock` bump, not suppression:**

| Package | Old → New | CVEs fixed |
|---|---|---|
| `aiohttp` | 3.13.5 → 3.14.1 | 11 (PYSEC-2026-237, CVE-2026-34993/47265/54273/54279/54277/50269/54276/54278/54280/54274) |
| `idna` | 3.13 → 3.18 | 1 (PYSEC-2026-215, listed twice) |
| `starlette` | 0.52.1 → 1.3.1 | 6 (PYSEC-2026-161/248/249, CVE-2026-48817/48818, plus the BADHOST-class fix) |
| `pydantic-settings` | 2.14.0 → 2.14.2 | 1 (GHSA-4xgf-cpjx-pc3j) |
| `fastapi` | 0.136.1 → 0.139.0 | (companion bump — starlette's own version is dictated by fastapi's constraint) |
| `prometheus-fastapi-instrumentator` | 7.1.0 → 8.0.2 | (companion bump — this package's own constraint was what pinned starlette back to 0.52.1 even after upgrading fastapi; needed to move in lockstep) |
| `requests` | 2.33.1 → 2.34.2 | (companion bump, transitive consistency) |

**Not a full `pip-compile` regen.** A full regeneration was tried first and rejected: it also
bumped `numpy` 2.4.4→2.5.1, which introduces PEP 695 (`type X = ...`) syntax into numpy's own
type stubs — `pyproject.toml` pins `mypy`'s `python_version = "3.10"` (no PEP 695 support),
turning `mypy`'s 3 pre-existing, unrelated errors into a total stub-parsing crash. Confirmed
directly: ran `mypy` against both the full-regen candidate (crash) and the current pin (3
pre-existing errors, unchanged) before deciding. Used `pip-compile --upgrade-package <name>`
(pip-tools' targeted-upgrade mode) instead — bumps only what's requested, leaves everything else,
including numpy/pandas/torch/scikit-learn/lightgbm/transformers/sentence-transformers, at their
current pins.

**Side-effect cleanup:** `google-auth`, `google-genai`, `pyasn1`, `pyasn1-modules`, `tenacity`,
`types-requests`, and `fastavro` all dropped out of the lock automatically — none of them are
resolvable from the current `requirements.txt` at all (confirmed: a fresh resolve without them
still passes the full test suite; `pyasn1`'s own CVE was reachable only through this dead
weight). This closes a separate, pre-existing lock/requirements.txt drift as a bonus, not a
new risk.

**Verified safe, not assumed:** built two isolated venvs (old lock vs. surgical candidate,
matching `ci.yml`'s exact install sequence — CPU torch first, then the lock). Both `mypy` runs
show the identical 3 pre-existing errors (`resolution.py` argmax/getitem typing, `triage.py`
`retry_key` redefinition — none new, none related to this change). Full `pytest tests/` —
**176/176 passed** on the surgical candidate (vs. the same 176 on the old lock) — this is the
real gate for whether the starlette 0.x→1.x major bump (the single riskiest change here) broke
anything behaviorally; it didn't. `pytest eval/ -v` — **12/12 passed** (one pre-existing,
unrelated `sklearn` `InconsistentVersionWarning` on cached model pickles trained with a newer
sklearn than the pinned 1.6.1 — not introduced by this change, scikit-learn's pin was untouched).

**Suppressed-with-justification (4 CVEs remain, all unreachable) — full entries in
`DEPENDENCIES.md`:**

| CVE | Package | Why unreachable |
|---|---|---|
| CVE-2026-1839 (existing) | `transformers` | `Trainer._load_rng_state()` — TriageIQ never imports `Trainer`. |
| PYSEC-2025-217 (new) | `transformers` | X-CLIP checkpoint-conversion-script RCE — TriageIQ never imports X-CLIP or runs conversion scripts. |
| CVE-2026-4372 (new) | `transformers` | Config-injection RCE requires the `kernels` package (confirmed not installed, not in the lock at all) AND an untrusted model-config injection point (TriageIQ loads exactly one fixed, trusted model, `BAAI/bge-base-en-v1.5`). |
| CVE-2025-3000 (new) | `torch` | `torch.jit.script` memory corruption — confirmed zero `torch.jit`/`torch.compile`/TorchScript usage anywhere in `src/`, `tests/`, `eval/`, `scripts/`. No fix version is even published upstream. |

None of the 4 are reachable, so none required escalation under this ADR's own rule ("escalate
only reachable-and-not-cheaply-patchable CVEs"). `ci.yml`'s security-audit step now runs with
all 4 `--ignore-vuln` flags, each with an inline comment pointing to the full `DEPENDENCIES.md`
justification.

**Final state, verified:** `pip-audit -r requirements.lock --skip-editable --ignore-vuln
CVE-2026-1839 --ignore-vuln PYSEC-2025-217 --ignore-vuln CVE-2026-4372 --ignore-vuln
CVE-2025-3000` → **"No known vulnerabilities found, 4 ignored."** The job goes green, honestly.

## Item 5 — Portfolio Groq key / GPU isolation (investigate + report only, ESCALATED)

Cross-project survey of `GROQ_API_KEY` usage, autonomous execution, and GPU usage across
sibling projects under `C:\Users\gaura\ml-projects\` (`AetherArt` excluded per hard rule,
untouched). `.env` files were never read (only `.env.example` templates and source-code
variable-name references — no secret values were inspected).

| Project | `GROQ_API_KEY` referenced | Autonomous execution | GPU usage |
|---|---|---|---|
| **triage-iq** (baseline) | Yes | No — CI/CD only on push | No — CPU-only torch |
| **agentgauge** | Yes (parked research scripts only) | **Yes** — `AUTONOMY.md` states a Claude Code **cloud scheduled routine** picks a backlog item and opens a draft PR unattended. Also runs local watchdog scripts (`t18_watchdog.ps1`, `ollama_conn_monitor.ps1`) polling every 240–500ms to detect "contamination" from unexpected models appearing in `ollama ps` — **direct, corroborating evidence of a prior real shared-GPU/Ollama contention incident** on this machine. | Indirect, via local Ollama (no CUDA torch declared) |
| **agentic-shopping-assistant** | Yes (production LLM) | No wired scheduler (a flywheel job is explicitly parked, manual-trigger only) | No |
| **multimodal-fashion-recommender** | Yes | No cron found | **Yes** — `pyproject.toml` pins a CUDA-index torch for this Windows platform; multiple `torch.device("cuda" if available else "cpu")` calls in encoder/ingestion/training scripts. **Most likely source of "an unidentified process loaded a model onto the GPU"** if run unattended. |
| **review-iq** | Yes | **Yes — heaviest GitHub Actions cron footprint**: nightly eval (`0 2 * * *`), daily DB backup (`0 18 * * *`), uptime check every 5 minutes (`*/5 * * * *`). The nightly eval cron almost certainly makes live Groq calls on an unattended schedule. | No |

**Finding:** `GROQ_API_KEY` (same variable name, very likely the same underlying account/org
given these are all one person's projects) is shared across at least 4 active projects plus
triage-iq. Two genuinely autonomous, unattended consumers exist: **agentgauge**'s cloud-scheduled
routine and **review-iq**'s nightly-eval cron — either could silently consume shared Groq TPD
budget that triage-iq's own recording/eval work depends on, with no visibility from triage-iq's
side into when they run. **multimodal-fashion-recommender** is the clearest GPU-contention
candidate on this machine; **agentgauge**'s own watchdog scripts are direct evidence this has
already happened once.

### Proposed isolation plan (escalated — not executed, no cross-project changes made)

1. **Separate Groq API keys per project**, all under the same org (Groq supports multiple named
   keys per account) — gives per-project TPD visibility and lets any one project's key be
   revoked/rate-limited independently without affecting the others. Currently indistinguishable
   from usage data alone since the variable name is shared.
2. **Time-box the autonomous consumers** — if `agentgauge`'s cloud routine and `review-iq`'s
   nightly eval cron are both live, consider staggering their schedules away from any time
   triage-iq is actively re-recording cassettes (a ~20-minute, TPD-sensitive live-Groq operation).
3. **GPU reservation discipline** — since `multimodal-fashion-recommender`'s local dev/training
   scripts explicitly target CUDA and `agentgauge`'s watchdog scripts exist specifically because
   of a past incident, a simple convention (e.g. checking `nvidia-smi`/`ollama ps` before
   starting any GPU-touching local script) would have caught the prior incident before it needed
   a dedicated watchdog.
4. **This is a human decision, not a triage-iq change** — none of these projects are in
   triage-iq's scope to modify. Flagging for your call on which (if any) to act on.

## Consequences

- `requirements.lock`, `DEPENDENCIES.md`, `.github/workflows/ci.yml`, and
  `.github/workflows/record-cassette.yml` are the only files this ADR touches. No `src/` changes,
  no schema changes, no eval/model changes.
- The next push to `main` (once this branch is merged) will trigger a deploy — `torch`,
  `transformers`, `sentence-transformers`, `scikit-learn`, `numpy`, `pandas`, `lightgbm` are all
  **unchanged**, so this is a dependency-only refresh with no expected behavior change to
  `/triage` or `/eval`. The `Dockerfile.prod` build will pick up the new `aiohttp`/`starlette`/
  `fastapi`/`pydantic-settings` versions; verified locally via the full test suite, not merely
  asserted.
- Item 5 remains an open, escalated cross-project question — no action taken beyond the
  investigation and proposed plan above.
