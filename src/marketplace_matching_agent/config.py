"""Application configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    cohere_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    postgres_url: str = "postgresql://postgres:matchdev@localhost:5433/marketplace"
    prompt_version: str = "v0.1.0"
    model_id: str = "claude-sonnet-4-5-20251022"
    embed_model: str = "voyage-3-large"
    rerank_model: str = "rerank-v3.5"
    embed_dim: int = 256
    rrf_k: int = 60
    eval_use_cache: bool = False
    eval_cost_cap_usd: float = 5.0


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
