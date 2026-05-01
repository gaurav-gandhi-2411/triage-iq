"""FastAPI application for the TriageIQ triage assistant.

Startup: loads all per-repo models once (TF-IDF, BGE+FAISS, LightGBM, TriageAssistant).
POST /triage: accepts issue text, returns a TriagePlan JSON with structured access log.
GET  /health: returns loaded repos and uptime.
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .loader import ModelStore
from .schemas import HealthResponse, TriageRequest

logger = logging.getLogger(__name__)

# llama-3.1-8b-instant pricing (per million tokens, as of 2025)
_GROQ_PRICE_PER_MTOK = 0.27


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = Path(os.environ.get("DATA_DIR", Path(__file__).parent.parent.parent.parent / "data"))
    logger.info("Loading models from %s …", data_dir)
    app.state.store = ModelStore.load_all(data_dir=data_dir)
    logger.info("Models ready: %s", app.state.store.repos)
    yield


app = FastAPI(
    title="TriageIQ",
    description="LLM-powered GitHub issue triage assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/triage")
def triage(body: TriageRequest, request: Request) -> JSONResponse:
    """Triage a GitHub issue and return a structured TriagePlan."""
    request_id = str(uuid.uuid4())
    t_start = time.perf_counter()

    store: ModelStore = request.app.state.store
    try:
        bundle = store.get(body.repo)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    issue = pd.Series({
        "number": body.issue_number,
        "title": body.title,
        "body_clean": body.body,
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
        raise HTTPException(status_code=500, detail=f"Triage failed: {exc}")

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
    return JSONResponse(content=result)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    store: ModelStore = request.app.state.store
    return HealthResponse(
        status="ok",
        repos_loaded=store.repos,
        groq_key_present=bool(os.environ.get("GROQ_API_KEY")),
        uptime_s=round(time.monotonic() - store.start_time, 1),
    )


# ---------------------------------------------------------------------------
# Structured logging helper
# ---------------------------------------------------------------------------

def _log_request(**fields) -> None:
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **fields,
    }
    logger.info("ACCESS %s", json.dumps(record, default=str))
