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
The **served inference API** never imports `Trainer`, never loads checkpoint files, and
exposes no model-path or file-upload surface to external callers — that's the surface this
suppression covers. `scripts/deberta_train.py` (an offline training script, not part of the
API) does import `Trainer`, but the attack requires local filesystem write access to a
checkpoint directory plus explicit `Trainer` invocation — neither condition is reachable
through the deployed service.

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

### PYSEC-2026-2290 — transformers LightGlue model-loading path RCE via `trust_remote_code` override

- **Suppressed since:** 2026-07-16
- **Affected package:** `transformers==4.57.6`
- **Fix version:** none published for the 4.x line (same constraint as the other open
  transformers findings above)

**Why suppressed:**
The vulnerability is in the LightGlue model's loading path: `LightGlueConfig` reads
`trust_remote_code` from an untrusted `config.json` and propagates it into nested
`AutoConfig.from_pretrained()` calls, overriding a caller's explicit
`trust_remote_code=False`. Checked directly: TriageIQ never loads a LightGlue model, and
the only raw `AutoModel.from_pretrained()` call anywhere in the codebase
(`scripts/w3_t4_train.py:317`) loads a single hardcoded, project-controlled model ID
(`BAAI/bge-base-en-v1.5`, the same constant used by the production retriever in
`src/triage_iq/models/similar_issues.py`) — never a dynamic or attacker-suppliable repo
ID. No reachable surface: no LightGlue model, no untrusted config source.

**Why not fixed immediately:** Same triple-major-version-bump blocker as the other open
transformers findings (`transformers>=5.x` requires `sentence-transformers>=5.x`).

**Revisit trigger:** Same as the other transformers findings above — bundled into the
same future major-version dependency refresh.

### CVE-2026-9856 — transformers `save_pretrained()` path traversal via `chat_template` keys

- **Suppressed since:** 2026-09-05
- **Affected package:** `transformers==4.57.6`
- **Fix version:** `transformers>=5.10.0` (same constraint as the other open transformers
  findings above)

**Why suppressed:**
The vulnerability is in `PreTrainedTokenizerBase.save_pretrained()` / `ProcessorMixin.
save_pretrained()`: dictionary keys from a loaded `chat_template` are used directly as
output filenames without path-traversal validation. A malicious Hugging Face Hub
repository's `tokenizer_config.json` can supply a `chat_template` dict whose keys escape
the intended save directory (e.g. `../../`-style keys), so calling `save_pretrained()` on
a tokenizer/processor loaded from that repo can write attacker-controlled content to an
attacker-chosen path. Checked directly: the only `.save_pretrained()` call sites anywhere
in this codebase are in four offline training scripts (`scripts/d2_train.py`,
`scripts/d3_train.py`, `scripts/deberta_train.py`, `scripts/w3_t4_train.py`), and every one
loads its tokenizer via a hardcoded `BASE_MODEL` constant (`BAAI/bge-base-en-v1.5` or
`microsoft/deberta-v3-base`) — never a dynamic, configurable, or attacker-suppliable model
ID. `src/triage_iq/models/similar_issues.py` loads a tokenizer via `SentenceTransformer`
for inference only and never calls `save_pretrained()`. No reachable surface: this
codebase never saves a tokenizer/processor loaded from an untrusted repository.

**Why not fixed immediately:** Same triple-major-version-bump blocker as the other open
transformers findings (`transformers>=5.x` requires `sentence-transformers>=5.x`).

**Revisit trigger:** Same as the other transformers findings above — bundled into the
same future major-version dependency refresh.

### PYSEC-2026-3447 — setuptools `FileList` MANIFEST.in exclude-pattern Unicode-normalization bypass

- **Suppressed since:** 2026-07-16
- **Affected package:** `setuptools` (version varies by CI runner image — not a pinned
  dependency in `requirements.lock`; TriageIQ never lists setuptools as a runtime or
  build requirement, it's incidental to the Python environment)
- **Fix version:** `setuptools>=83.0.0`

**Why suppressed:**
The vulnerability requires building a source distribution (`sdist`) with a `MANIFEST.in`
containing `exclude`/`global-exclude`/`recursive-exclude`/`prune` directives, on a
filesystem with Unicode NFD/NFC normalization behavior (macOS APFS/HFS+). Checked
directly: this repo has no `MANIFEST.in`, no `setup.py`/`setup.cfg` (packaging config is
`pyproject.toml`, tool-config only per this project's conventions) — it is never built as
a distributable sdist. CI and production both run on Linux (`ubuntu-latest` /
`Dockerfile.prod`'s Linux base image), not macOS, so the filesystem precondition is also
absent. Both preconditions for exploitation are unmet.

**Why not fixed immediately:** Not a pinned dependency — there is no `requirements.lock`
line to bump. The version pip-audit reports varies by whichever setuptools ships with
the runner's Python installation.

**Revisit trigger:** If this repo ever starts building/publishing an sdist (it does not
today), pin `setuptools>=83.0.0` explicitly as a build dependency at that time.

### PYSEC-2026-3716 — datasets folder-based dataset builder path traversal

- **Suppressed since:** 2026-08-27
- **Affected package:** `datasets==4.8.5` (`requirements.lock`; declared directly in
  `requirements.txt` alongside `transformers`/`sentence-transformers`, but never imported
  directly anywhere in this codebase — `grep -rn "^import datasets\|^from datasets"` across
  `src/`, `eval/`, `scripts/`, `tests/` returns zero matches; it is pulled in for the
  HF ecosystem tooling those packages use, not called by TriageIQ's own code)
- **Fix version:** `datasets>=5.0.1` (published; not yet pinned here — see below)

**Why suppressed:**
The vulnerability is in `datasets`' folder-based dataset builders (e.g. the `imagefolder`/
`audiofolder` builders): a crafted `file_name` metadata field is joined to the dataset
directory without path-traversal validation, then read and embedded into output when
`save_to_disk()` or `push_to_hub()` is called on the loaded dataset. Checked directly:
TriageIQ never calls `datasets.load_dataset()` with a folder-based builder, never calls
`save_to_disk()`/`push_to_hub()` anywhere, and never loads a dataset folder from an
untrusted or attacker-suppliable source — all data ingestion in this repo goes through its
own `pandas`/`parquet`-based pipeline (`src/triage_iq/data/`), not `datasets`' loaders. No
reachable surface: the vulnerable call chain is never invoked, directly or transitively.

**Why not fixed immediately:** A fixed release (`datasets==5.0.1`) is published, but bumping
to it is not a safe standalone one-line pin edit right now. `requirements.txt`'s
`sentence-transformers` floor was already raised to `>=5.7.0,<6.0` by Dependabot PR #69
(2026-08-11) without a matching `requirements.lock` regeneration — the committed lock still
has `sentence-transformers==2.7.0`, which no longer satisfies `requirements.txt`'s own
constraint. Any fresh `pip-compile requirements.txt --output-file=requirements.lock` run
(even one scoped with `--upgrade-package datasets`) re-resolves against that already-bumped
floor and silently pulls `sentence-transformers` 2.7.0→5.7.0 and `transformers` 4.57.6→5.16.1
along with it — the exact triple-major-version refresh that CVE-2026-1839, PYSEC-2025-217,
CVE-2026-4372, and PYSEC-2026-2290 above are deliberately deferring pending 6+ months of
`sentence-transformers` 5.x stability and confirmed `.encode()` API compatibility. Bumping
`datasets` alone requires first resolving the `requirements.txt`/`requirements.lock` drift
without triggering that refresh (e.g. temporarily re-pinning `sentence-transformers` back to
its `<3.0` floor for the regen, then restoring the `>=5.7.0` floor once the major refresh is
actually scheduled) — not something to do as a side effect of one CVE bump.

**Revisit trigger:** (1) Resolve the `requirements.txt`/`requirements.lock` drift on
`sentence-transformers` as its own reviewed change; (2) then bump `requirements.lock`'s
`datasets` pin to `>=5.0.1` and drop this suppression — no source code change needed since
the package isn't imported directly.
