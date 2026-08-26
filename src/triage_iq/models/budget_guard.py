"""Proactive daily token-budget guard for Groq calls.

Part B2 (2026-08-27 diagnostic session): consolidating to one Groq account makes TPD
exhaustion the shared, single point of contention between production and batch (eval/
cassette-recording) work. This guard lets production check "am I about to blow the day's
budget" BEFORE spending a request on a call that Groq would reject anyway -- saving the
latency of a doomed call and giving early, proactive degradation instead of only reacting
to a 429 after the fact.

State must be durable and shared across Cloud Run instances -- Cloud Run can run N
concurrent instances and scale to zero, so an in-process counter would both undercount
(each instance has its own view) and reset on every cold start. Firestore is the cheapest
fit for this project's existing stack: already GCP, serverless, pay-per-operation with a
generous free tier, and a single document is enough -- no VM, no new service to run.

This is NOT "zero new infra", and this module does not provision it: using Firestore for
real requires (1) enabling the Firestore API on the project and (2) granting the Cloud Run
runtime service account `roles/datastore.user`. Both are one-time GCP changes outside this
module's scope -- see the PR body. Until that grant exists, `DailyBudgetGuard` fails OPEN
(never blocks a request because the guard itself can't reach its backend) -- a budget guard
that can 500 the whole service on a permission it doesn't have is a worse bug than the
thing it exists to prevent.
"""

from __future__ import annotations

import datetime
import logging
import threading

logger = logging.getLogger(__name__)

_WARNED_UNAVAILABLE = threading.Event()


class DailyBudgetGuard:
    """Tracks Groq token usage against a per-UTC-day budget in Firestore.

    Usage:
        guard = DailyBudgetGuard(project="triageiq-prod-260812", limit_tokens=150_000)
        if guard.over_budget():
            # degrade without spending a call
        ...
        raw, usage = self._groq_completion(messages)
        guard.record(usage["prompt_tokens"] + usage["completion_tokens"])
    """

    def __init__(
        self,
        project: str,
        limit_tokens: int,
        collection: str = "groq_daily_budget",
        workload: str = "default",
    ) -> None:
        self._project = project
        self._limit = limit_tokens
        self._collection = collection
        self._workload = workload
        self._client = None  # lazy; None means "not yet attempted or unavailable"
        self._client_attempted = False

    def _get_client(self):
        if self._client_attempted:
            return self._client
        self._client_attempted = True
        try:
            from google.cloud import firestore

            self._client = firestore.Client(project=self._project)
        except Exception as exc:  # ImportError, DefaultCredentialsError, PermissionDenied, ...
            if not _WARNED_UNAVAILABLE.is_set():
                _WARNED_UNAVAILABLE.set()
                logger.warning(
                    "DailyBudgetGuard: Firestore unavailable (%s: %s) -- failing OPEN "
                    "(budget checks are a no-op until this is fixed; see budget_guard.py's "
                    "module docstring for the required one-time IAM grant).",
                    type(exc).__name__, exc,
                )
            self._client = None
        return self._client

    def _doc_id(self) -> str:
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        return f"{self._workload}_{today}"

    def over_budget(self) -> bool:
        """True if today's recorded usage has already reached the budget. Fails open
        (returns False) if Firestore is unreachable -- never blocks production because
        the guard itself is broken."""
        client = self._get_client()
        if client is None:
            return False
        try:
            doc = client.collection(self._collection).document(self._doc_id()).get()
            used = doc.get("tokens_used") if doc.exists else 0
            return bool(used) and used >= self._limit
        except Exception as exc:
            logger.warning("DailyBudgetGuard.over_budget() read failed, failing open: %s", exc)
            return False

    def record(self, tokens: int) -> None:
        """Atomically add `tokens` to today's counter. Best-effort -- a failure here must
        never fail the request that already succeeded against Groq."""
        if tokens <= 0:
            return
        client = self._get_client()
        if client is None:
            return
        try:
            from google.cloud import firestore

            ref = client.collection(self._collection).document(self._doc_id())
            ref.set({"tokens_used": firestore.Increment(tokens)}, merge=True)
        except Exception as exc:
            logger.warning("DailyBudgetGuard.record() write failed (non-fatal): %s", exc)


class GroqBudgetExceeded(Exception):
    """Raised in place of a live Groq call when DailyBudgetGuard.over_budget() is True.
    Treated as a degrade trigger by triage_with_metadata, same as a real Groq exception --
    see triage.py's _is_groq_unavailable / GroqBudgetExceeded check."""


class NoOpBudgetGuard:
    """Default guard when no durable backend is configured -- always reports under budget.
    Distinct class (not just `None`) so callers don't need a None-check at every call site."""

    def over_budget(self) -> bool:
        return False

    def record(self, tokens: int) -> None:
        return None
