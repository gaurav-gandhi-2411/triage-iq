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
