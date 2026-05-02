"""Request/response schemas for the triage API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SUPPORTED_REPOS = Literal["microsoft/vscode", "kubernetes/kubernetes"]


class TriageRequest(BaseModel):
    repo: SUPPORTED_REPOS
    title: str = Field(min_length=1, max_length=512)
    body: str = Field(default="", max_length=32_000)
    issue_number: int = Field(default=-1)
    created_at: datetime | None = Field(
        default=None,
        description="ISO 8601 timestamp of issue creation. Defaults to request time if omitted.",
    )


class HealthResponse(BaseModel):
    status: str
    repos_loaded: list[str]
    groq_key_present: bool
    uptime_s: float
