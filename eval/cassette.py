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

        Entries written before the request-storage schema addition are the response dict
        directly (no "response" wrapper key); entries written after wrap it as
        {"provider", "model", "request_messages", "response"}. Both are read transparently
        -- callers always get back the response dict either way.
        """
        entry = self._entries.get(key)
        if entry is None:
            if self._strict:
                nearest = self._nearest_entry_diagnostic(key)
                raise CassetteMissError(
                    f"Cassette miss for key {key[:16]}… — "
                    "re-run eval/record_cassettes.py and commit the updated cassette."
                    + (f"\n{nearest}" if nearest else "")
                )
            return None
        return entry["response"] if isinstance(entry, dict) and "response" in entry else entry

    def get_request(self, key: str) -> dict | None:
        """Return the stored request (provider, model, request_messages) for `key`, or None.

        Only available for entries written after the request-storage schema addition --
        entries recorded before it never had a request stored, so this returns None for
        them even though the entry itself exists (a real miss on `get_request`, not a bug).
        """
        entry = self._entries.get(key)
        if isinstance(entry, dict) and "response" in entry:
            return {
                "provider": entry.get("provider"),
                "model": entry.get("model"),
                "request_messages": entry.get("request_messages"),
            }
        return None

    def _nearest_entry_diagnostic(self, key: str, max_candidates: int = 500) -> str | None:
        """Best-effort diagnostic for a miss: find the stored request whose messages are
        textually closest to... nothing, since we don't have the missing request's own text
        here (only its hash). This reports how many entries in the cassette DO have request
        text stored, so a human knows whether request-level diffing is even possible for
        this cassette, rather than guessing from a bare 16-char hash prefix.
        """
        with_requests = sum(
            1 for e in list(self._entries.values())[:max_candidates]
            if isinstance(e, dict) and "response" in e and e.get("request_messages")
        )
        total = len(self._entries)
        if with_requests == 0:
            return (
                f"(0/{total} entries in this cassette have stored request text -- "
                "recorded before the request-storage schema addition. Re-record to get "
                "diagnosable misses.)"
            )
        return f"({with_requests}/{total} entries have stored request text available for diffing.)"

    def diff_against_nearest(self, request_messages: list[dict]) -> str:
        """Find the stored entry whose request_messages are textually closest to
        `request_messages` and report where they diverge. For manual triage after a
        CassetteMissError -- not called automatically from get(), since get() only has the
        already-hashed key, not the original messages that produced it (the caller does).

        "Closest" = longest common prefix over the joined message content strings; good
        enough to point at the right entry and the right divergent field without pulling in
        a real diff library.
        """
        target_text = "\x00".join(m.get("content", "") for m in request_messages)
        best_key: str | None = None
        best_prefix_len = -1
        best_text = ""
        for k, entry in self._entries.items():
            if not (isinstance(entry, dict) and "response" in entry and entry.get("request_messages")):
                continue
            candidate_text = "\x00".join(m.get("content", "") for m in entry["request_messages"])
            prefix_len = 0
            for a, b in zip(target_text, candidate_text):
                if a != b:
                    break
                prefix_len += 1
            if prefix_len > best_prefix_len:
                best_prefix_len = prefix_len
                best_key = k
                best_text = candidate_text
        if best_key is None:
            return "No committed entry has stored request text to diff against (all pre-date the request-storage schema addition)."
        divergence_point = best_prefix_len
        return (
            f"Nearest committed entry: {best_key[:16]}… "
            f"(matches for the first {divergence_point} characters, then diverges)\n"
            f"  requested:  ...{target_text[max(0, divergence_point-40):divergence_point+80]!r}\n"
            f"  committed:  ...{best_text[max(0, divergence_point-40):divergence_point+80]!r}"
        )

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
        (matching LLMCache's storage contract). `request` (the messages list passed to
        compute_key) is stored alongside it so a future miss can be diffed against the
        nearest committed entry instead of just a hash prefix.
        """
        self._entries[key] = {
            "provider": provider,
            "model": model,
            "request_messages": request,
            "response": response,
        }
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
