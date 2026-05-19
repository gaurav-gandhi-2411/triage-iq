"""Centralised application configuration via pydantic-settings.

All values are read from environment variables (or .env file).
Call get_settings() anywhere; the result is lru_cached after the first call.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: SecretStr
    data_dir: Path = _PROJECT_ROOT / "data"
    port: int = 8080
    log_level: str = "INFO"
    rate_limit_enabled: bool = True
    rate_limit_per_hour: str = "10/hour"
    rate_limit_per_day: str = "30/day"
    groq_model_triage: str = "llama-3.1-8b-instant"
    environment: Literal["dev", "test", "prod"] = "prod"
    metrics_token: SecretStr | None = None
    llm_cache_enabled: bool = False
    llm_cache_path: Path = _PROJECT_ROOT / "data" / "llm_cache.sqlite"
    reranker_enabled: bool = False
    reranker_model: str = "mixedbread-ai/mxbai-rerank-base-v1"

    @field_validator("groq_api_key", mode="after")
    @classmethod
    def _key_must_not_be_empty(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value():
            raise ValueError("GROQ_API_KEY must not be empty")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # pydantic-settings reads required fields from env
