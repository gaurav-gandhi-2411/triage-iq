"""FastAPI application for the TriageIQ triage assistant.

Startup: loads all per-repo models once (TF-IDF, BGE+FAISS, LightGBM, TriageAssistant).
POST /triage: accepts issue text, returns a TriagePlan JSON with structured access log.
GET  /health: returns loaded repos and uptime.
"""

import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from ..config import get_settings
from ..models.triage import TriagePlan
from .loader import ModelStore
from .schemas import HealthResponse, ServiceInfoResponse, TriageRequest

logger = logging.getLogger(__name__)

# llama-3.1-8b-instant pricing (per million tokens, as of 2025)
_GROQ_PRICE_PER_MTOK = 0.27

# ---------------------------------------------------------------------------
# Prometheus custom metrics
# ---------------------------------------------------------------------------

_triage_requests_total = Counter(
    "triage_requests_total",
    "Total /triage requests by repo and outcome",
    ["repo", "status"],  # status: success | error | fallback
)
_triage_llm_fallback_total = Counter(
    "triage_llm_fallback_total",
    "Number of triage calls that used the fallback plan (LLM parse failure)",
)
_triage_groq_tokens_total = Counter(
    "triage_groq_tokens_total",
    "Cumulative Groq tokens consumed (prompt + completion)",
)
_triage_latency_seconds = Histogram(
    "triage_latency_seconds",
    "End-to-end /triage request latency in seconds",
    buckets=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0],
)


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
# /metrics auth dependency
# ---------------------------------------------------------------------------

def _verify_metrics_token(authorization: str | None = Header(default=None)) -> None:
    """Fail-closed /metrics auth.

    - Token set (any env): require Authorization: Bearer <token>; 401 otherwise.
    - No token, env=prod: 503 — prevents silent exposure if the secret reference breaks.
    - No token, env=dev/test: open endpoint (local development convenience).

    .strip() on the configured token guards against whitespace baked into a
    Secret Manager value (e.g. trailing \\r\\n from PowerShell-piped creation).
    hmac.compare_digest is timing-safe for token comparison.
    """
    cfg = get_settings()
    token = (cfg.metrics_token.get_secret_value() if cfg.metrics_token else "").strip()
    if token:
        incoming = authorization or ""
        if not hmac.compare_digest(incoming, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="Unauthorized")
        return
    if cfg.environment == "prod":
        raise HTTPException(status_code=503, detail="Metrics endpoint not available")


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
    _token_set = bool(cfg.metrics_token and cfg.metrics_token.get_secret_value())
    _metrics_state = (
        "protected with token" if _token_set
        else "open (dev only)" if cfg.environment != "prod"
        else "disabled (prod, no token)"
    )
    logger.info("metrics endpoint: %s", _metrics_state)
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

# Instrument all routes for automatic HTTP metrics (request count, latency).
# We do NOT call .expose() — we add /metrics manually below with token auth.
Instrumentator().instrument(app)


@app.get("/", response_model=ServiceInfoResponse, include_in_schema=False)
def service_info() -> ServiceInfoResponse:
    """Service discovery endpoint — returns name, version, and links."""
    return ServiceInfoResponse(
        service=app.title,
        version=app.version,
        description=app.description or "",
        docs="/docs",
        health="/health",
        repository="https://github.com/gaurav-gandhi-2411/triage-iq",
        supported_repos=list(app.state.store.repos) if hasattr(app.state, "store") else [],
    )


@app.get("/metrics", include_in_schema=False)
def metrics(_: None = Depends(_verify_metrics_token)) -> Response:
    """Prometheus metrics endpoint. Requires Authorization: Bearer <METRICS_TOKEN> when configured."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/triage", response_model=TriagePlan)
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
        _triage_requests_total.labels(repo=body.repo, status="error").inc()
        _triage_latency_seconds.observe(total_ms / 1000.0)
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
    llm_status = meta.get("llm_status", "ok")
    req_status = "fallback" if llm_status == "parse_failure" else "success"

    _triage_requests_total.labels(repo=body.repo, status=req_status).inc()
    _triage_latency_seconds.observe(total_ms / 1000.0)
    if llm_status != "ok":
        _triage_llm_fallback_total.inc()
    tokens = (meta.get("groq_tokens_prompt") or 0) + (meta.get("groq_tokens_completion") or 0)
    if tokens:
        _triage_groq_tokens_total.inc(tokens)

    _log_request(
        request_id=request_id,
        endpoint="/triage",
        repo=body.repo,
        title_length=len(body.title),
        body_length=len(body.body),
        total_latency_ms=total_ms,
        status="success",
        predicted_component=plan.predicted_component,
        groq_tokens_total=tokens,
        **{k: v for k, v in meta.items() if k != "total_latency_ms"},
    )

    result = plan.model_dump()
    result["_request_id"] = request_id
    result["_llm_status"] = llm_status
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
