from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "v1"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key   TEXT PRIMARY KEY,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    request_json  TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hit_count   INTEGER DEFAULT 0,
    last_hit_at TIMESTAMP
)
"""


class LLMCache:
    """SQLite-backed LLM response cache keyed on SHA-256 of canonical request JSON.

    Thread-safe for concurrent FastAPI requests: uses check_same_thread=False plus
    a write lock. Reads are unlocked (SQLite WAL mode serialises internally).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else Path("data/llm_cache.sqlite")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_TABLE)
        self._lock = threading.Lock()
        self._session_hits = 0
        self._session_misses = 0

    # ------------------------------------------------------------------
    # Key computation
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
        """SHA-256 of a canonical JSON payload that uniquely identifies a request."""
        payload: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "provider": provider,
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(extra)
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get(self, key: str) -> dict | None:
        """Return cached response dict or None on cache miss.

        Also increments hit_count and last_hit_at in the DB on a hit.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM llm_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is None:
                self._session_misses += 1
                return None
            self._session_hits += 1
            self._conn.execute(
                "UPDATE llm_cache SET hit_count = hit_count + 1, last_hit_at = CURRENT_TIMESTAMP"
                " WHERE cache_key = ?",
                (key,),
            )
        return json.loads(row[0])

    def set(
        self,
        key: str,
        provider: str,
        model: str,
        request: Any,
        response: Any,
    ) -> None:
        """Insert a cache entry. Silently ignores duplicate keys (INSERT OR IGNORE)."""
        request_json = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        response_json = json.dumps(response, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO llm_cache"
                " (cache_key, provider, model, request_json, response_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (key, provider, model, request_json, response_json),
            )

    # ------------------------------------------------------------------
    # Stats / admin
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return session and DB stats."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM llm_cache").fetchone()
        entries, total_hits = row if row else (0, 0)
        size_bytes = self._path.stat().st_size if self._path.exists() else 0
        total = self._session_hits + self._session_misses
        return {
            "entries": entries,
            "total_hits_ever": int(total_hits),
            "session_hits": self._session_hits,
            "session_misses": self._session_misses,
            "session_hit_rate": self._session_hits / total if total else 0.0,
            "size_bytes": size_bytes,
            "path": str(self._path),
        }

    def clear(self) -> int:
        """Delete all cache entries. Returns number of rows removed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM llm_cache")
            return cur.rowcount

    def clear_provider(self, provider: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM llm_cache WHERE provider = ?", (provider,)
            )
            return cur.rowcount

    def clear_model(self, provider: str, model: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM llm_cache WHERE provider = ? AND model = ?", (provider, model)
            )
            return cur.rowcount

    def close(self) -> None:
        self._conn.close()
