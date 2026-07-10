# Dependency Notes

## Suppressed CVEs

### CVE-2026-1839 — transformers 4.x `Trainer._load_rng_state()` unsafe deserialization

- **Suppressed since:** 2026-05-06
- **CVSS:** 7.8 HIGH (NVD) / 6.5 MEDIUM (CNA)
- **NVD link:** https://nvd.nist.gov/vuln/detail/CVE-2026-1839
- **Affected package:** `transformers==4.57.6`
- **Fix version:** `transformers>=5.0.0rc3`

**Why suppressed:**
The vulnerability is in `Trainer._load_rng_state()` (`src/transformers/trainer.py`), which
uses `torch.load()` without `weights_only=True` during training checkpoint restoration.
TriageIQ is an **inference-only** service: it never imports `Trainer`, never loads checkpoint
files, and exposes no model-path or file-upload surface to external callers. The attack
requires local filesystem write access to a checkpoint directory and explicit `Trainer`
invocation — neither condition is reachable through our API.

**Why not fixed immediately:**
The fix requires a double major version bump:
- `sentence-transformers 2.7.0 → 5.4.1` (three skipped majors; embedding behaviour risk)
- `transformers 4.57.6 → 5.x`

No 4.x backport exists (`4.57.6` is the final 4.x release).

**Revisit trigger:**
When a planned dependency refresh is scheduled (not bundled into an unrelated PR).
Prerequisite: `sentence-transformers 5.x` stable for 6+ months with confirmed API
compatibility for `SentenceTransformer.encode()` and `BAAI/bge-base-en-v1.5`.

### PYSEC-2025-217 — transformers X-CLIP checkpoint conversion deserialization RCE

- **Suppressed since:** 2026-07-10
- **Affected package:** `transformers==4.57.6`
- **Fix version:** none published for the 4.x line (same constraint as CVE-2026-1839 below)

**Why suppressed:**
The vulnerability is in the X-CLIP model's checkpoint-conversion script
(`convert_x_clip_original_pytorch_checkpoint_to_pytorch.py`), which requires explicitly
running that conversion utility against an untrusted checkpoint file. TriageIQ never
imports X-CLIP, never runs any checkpoint-conversion script, and loads exactly one
HuggingFace model (`BAAI/bge-base-en-v1.5`, hardcoded in
`src/triage_iq/models/similar_issues.py`) via `SentenceTransformer.encode()` — no
conversion utilities anywhere in the pipeline. No reachable surface.

**Revisit trigger:** Same as CVE-2026-1839 — bundled into the same future
`sentence-transformers`/`transformers` major-version refresh.

### CVE-2026-4372 — transformers config injection RCE via `_attn_implementation_internal`

- **Suppressed since:** 2026-07-10
- **CVSS:** Critical (unauthenticated RCE, widely reported)
- **Affected package:** `transformers==4.57.6` (vulnerable range 4.56.0–5.2.x)
- **Fix version:** `transformers>=5.3.0`

**Why suppressed:**
The vulnerability requires the optional `kernels` package to be installed (the exploitable
kernel-dispatch path) AND a way for an attacker to supply an untrusted `config.json` that
gets loaded. Checked directly: the `kernels` package is not installed and is not pulled in
by anything in `requirements.lock` (`pip show kernels` → not found). TriageIQ also loads
exactly one fixed, project-controlled model (`BAAI/bge-base-en-v1.5`) — there is no code
path where an external caller can supply an arbitrary model config to be loaded. Both
preconditions for exploitation are absent.

**Why not fixed immediately:** Same triple-major-version-bump blocker as CVE-2026-1839
(`transformers>=5.3.0` requires `sentence-transformers>=5.x`).

**Revisit trigger:** Same as CVE-2026-1839 — this is the more urgent of the two open
transformers findings if the dependency refresh becomes reachable sooner; re-check first.

### CVE-2025-3000 — torch `torch.jit.script` memory corruption on scripted classes with list attributes

- **Suppressed since:** 2026-07-10
- **CVSS:** 5.3 MEDIUM
- **Affected package:** `torch==2.11.0`
- **Fix version:** none published (`pip-audit` reports no fix version for this advisory)

**Why suppressed:**
The vulnerability requires calling `torch.jit.script` on a scripted class with a list
attribute. Checked directly: TriageIQ never calls `torch.jit.script`, `torch.jit.trace`,
`torch.compile`, or any TorchScript API anywhere in the codebase (`grep` across
`src/`, `tests/`, `eval/`, `scripts/` — zero matches). `sentence-transformers`'
default `.encode()` inference path used here does not invoke TorchScript compilation
either. No reachable surface.

**Revisit trigger:** No fix version is currently published upstream — nothing to upgrade
to. Re-check when PyTorch ships a patched release, or if this codebase ever adds
TorchScript-based model export/optimization (it does not today).
