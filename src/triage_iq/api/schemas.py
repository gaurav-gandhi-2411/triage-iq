"""Request/response schemas for the triage API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


SUPPORTED_REPOS = Literal["microsoft/vscode", "kubernetes/kubernetes"]


class TriageRequest(BaseModel):
    repo: SUPPORTED_REPOS
    title: str = Field(min_length=1, max_length=512)
    body: str = Field(default="", max_length=32_000)
    issue_number: int = Field(default=-1)


class HealthResponse(BaseModel):
    status: str
    repos_loaded: list[str]
    groq_key_present: bool
    uptime_s: float
