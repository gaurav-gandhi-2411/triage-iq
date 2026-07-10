from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "v1"


class CassetteMissError(RuntimeError):
    """Raised in replay-only mode when a requested LLM call is not in the cassette.

    A miss means the upstream request changed (prompt, model, or parameters).
    To fix: re-run eval/record_cassettes.py and commit the updated cassette + baseline.
    """


class CassettePlayer:
    """JSON-backed cassette for LLM replay in CI.

    In record mode (strict=False): cache miss falls through to the live LLM call,
    then the response is stored. Use for the one-time recording pass.

    In replay mode (strict=True, default): cache miss raises CassetteMissError.
    Use in CI — no live LLM calls permitted.

    The cassette uses the same SHA-256 key computation as LLMCache
    (triage_iq.cache.llm_cache) so recorded keys are compatible.

    Storage format: JSON (human-readable, diff-friendly for PR review).
    """

    def __init__(self, path: Path | str, strict: bool = True) -> None:
        self._path = Path(path)
        self._strict = strict
        self._entries: dict[str, Any] = {}
        if self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Key computation (delegates to LLMCache for consistency)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_key(
        provider: str,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **extra: Any,
    ) -> str:
        from triage_iq.cache import LLMCache
        return LLMCache.compute_key(provider, model, messages, temperature, max_tokens, **extra)

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get(self, key: str) -> dict | None:
        """Return cached response dict or None.

        In strict (replay) mode, raises CassetteMissError on miss instead of returning None.
        """
        entry = self._entries.get(key)
        if entry is None and self._strict:
            raise CassetteMissError(
                f"Cassette miss for key {key[:16]}… — "
                "re-run eval/record_cassettes.py and commit the updated cassette."
            )
        return entry

    def set(
        self,
        key: str,
        provider: str,
        model: str,
        request: Any,
        response: Any,
    ) -> None:
        """Store a cassette entry and persist to disk.

        The response dict is what get() returns — it must have a 'content' key
        (matching LLMCache's storage contract).
        """
        self._entries[key] = response
        self._save()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "entries": len(self._entries),
            "path": str(self._path),
            "strict": self._strict,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self._entries = raw.get("entries", {})

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "description": (
                "Recorded LLM interactions for eval/test_quality_regression.py. "
                "DO NOT EDIT manually. Re-record via eval/record_cassettes.py."
            ),
            "entries": self._entries,
        }
        # newline="\n" pins LF regardless of platform -- eval_cassette.json is declared
        # `text eol=lf` in .gitattributes, and Path.write_text's default newline translation
        # (CRLF on Windows) would otherwise write bytes that differ from what git actually
        # commits, silently invalidating any cassette_hash computed from this file pre-commit.
        self._path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
        )
