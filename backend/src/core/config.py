from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    llm_request_timeout: float = Field(default=60.0, alias="LLM_REQUEST_TIMEOUT")
    llm_max_retries: int = Field(default=5, alias="LLM_MAX_RETRIES")
    corpus_dir: Path = Field(default=Path("judgement_pdfs"), alias="CORPUS_DIR")
    sqlite_db_path: Path = Field(default=Path("data/Casey.db"), alias="SQLITE_DB_PATH")
    # Qdrant – set QDRANT_URL (+ optionally QDRANT_API_KEY) for remote/cloud mode.
    # Leave unset to use embedded local storage at QDRANT_PATH (default: data/qdrant).
    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="judgments", alias="QDRANT_COLLECTION")
    qdrant_path: Path = Field(default=Path("data/qdrant"), alias="QDRANT_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    # Redis — leave unset to disable caching (app still works without it)
    redis_url: str | None = Field(default=None, alias="REDIS_URL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
