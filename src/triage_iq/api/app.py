"""FastAPI application for the TriageIQ triage assistant.

Startup: loads all per-repo models once (TF-IDF, BGE+FAISS, LightGBM, TriageAssistant).
POST /triage: accepts issue text, returns a TriagePlan JSON.
GET  /health: returns loaded repos and uptime.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .loader import ModelStore
from .schemas import HealthResponse, TriageRequest

logger = logging.getLogger(__name__)


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
        plan = bundle.assistant.triage(issue)
    except Exception as exc:
        logger.exception("Triage failed for repo=%s", body.repo)
        raise HTTPException(status_code=500, detail=f"Triage failed: {exc}")

    return JSONResponse(content=plan.model_dump())


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    store: ModelStore = request.app.state.store
    return HealthResponse(
        status="ok",
        repos_loaded=store.repos,
        groq_key_present=bool(os.environ.get("GROQ_API_KEY")),
        uptime_s=round(time.monotonic() - store.start_time, 1),
    )
