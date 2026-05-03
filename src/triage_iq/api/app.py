"""FastAPI application for the TriageIQ triage assistant.

Startup: loads all per-repo models once (TF-IDF, BGE+FAISS, LightGBM, TriageAssistant).
POST /triage: accepts issue text, returns a TriagePlan JSON with structured access log.
GET  /health: returns loaded repos and uptime.
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from ..config import get_settings
from .loader import ModelStore
from .schemas import HealthResponse, TriageRequest

logger = logging.getLogger(__name__)

# llama-3.1-8b-instant pricing (per million tokens, as of 2025)
_GROQ_PRICE_PER_MTOK = 0.27


# ---------------------------------------------------------------------------
# JSON access logging — one JSON object per stdout line so Cloud Run's
# logging agent reliably routes entries to jsonPayload, not textPayload.
# Filter with: jsonPayload.log_type="access"
# ---------------------------------------------------------------------------

_LOG_RECORD_BUILTIN_KEYS: frozenset[str] = frozenset(vars(logging.makeLogRecord({})))


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d: dict[str, object] = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
        }
        for key, val in vars(record).items():
            if key not in _LOG_RECORD_BUILTIN_KEYS and not key.startswith("_"):
                d[key] = val
        if record.exc_info:
            d["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


class _StdoutHandler(logging.Handler):
    """Writes to sys.stdout at emit time so pytest capsys captures output."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            import sys
            print(self.format(record), file=sys.stdout, flush=True)
        except Exception:
            self.handleError(record)


_access_handler = _StdoutHandler()
_access_handler.setFormatter(_JsonFormatter())
_access_logger = logging.getLogger("triage_iq.access")
_access_logger.setLevel(logging.INFO)
_access_logger.addHandler(_access_handler)
_access_logger.propagate = False


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    """Return real client IP; Cloud Run proxies via X-Forwarded-For."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


limiter = Limiter(key_func=_client_ip)


def _rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse({"detail": "Too many requests"}, status_code=429)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    limiter.enabled = cfg.rate_limit_enabled
    logger.info("Loading models from %s …", cfg.data_dir)
    app.state.store = ModelStore.load_all(
        data_dir=cfg.data_dir,
        groq_api_key=cfg.groq_api_key.get_secret_value(),
    )
    logger.info("Models ready: %s", app.state.store.repos)
    yield


app = FastAPI(
    title="TriageIQ",
    description="LLM-powered GitHub issue triage assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)


@app.post("/triage")
@limiter.limit("10/hour")
@limiter.limit("30/day")
def triage(body: TriageRequest, request: Request) -> JSONResponse:
    """Triage a GitHub issue and return a structured TriagePlan."""
    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()

    store: ModelStore = request.app.state.store
    try:
        bundle = store.get(body.repo)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    issue = pd.Series({
        "number": body.issue_number,
        "title": body.title,
        "body_clean": body.body,
        "created_at": body.created_at or datetime.now(timezone.utc),
    })

    try:
        plan, meta = bundle.assistant.triage_with_metadata(issue)
    except Exception as exc:
        total_ms = round((time.perf_counter() - t_start) * 1000, 1)
        _log_request(
            request_id=request_id,
            endpoint="/triage",
            repo=body.repo,
            title_length=len(body.title),
            body_length=len(body.body),
            total_latency_ms=total_ms,
            status="error",
            error=str(exc),
        )
        logger.exception("Triage failed for repo=%s", body.repo)
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    _log_request(
        request_id=request_id,
        endpoint="/triage",
        repo=body.repo,
        title_length=len(body.title),
        body_length=len(body.body),
        total_latency_ms=total_ms,
        status="success",
        predicted_component=plan.predicted_component,
        **{k: v for k, v in meta.items() if k != "total_latency_ms"},
    )

    result = plan.model_dump()
    result["_request_id"] = request_id
    result["_llm_status"] = meta.get("llm_status", "ok")
    return JSONResponse(content=result)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    cfg = get_settings()
    store: ModelStore = request.app.state.store
    return HealthResponse(
        status="ok",
        repos_loaded=store.repos,
        groq_key_present=bool(cfg.groq_api_key.get_secret_value()),
        uptime_s=round(time.monotonic() - store.start_time, 1),
    )


# ---------------------------------------------------------------------------
# Structured access log helper
# ---------------------------------------------------------------------------

def _log_request(**fields) -> None:
    _access_logger.info("access", extra={"log_type": "access", **fields})
