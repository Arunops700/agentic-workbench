"""Configuration from the environment / `.env`."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Policy: "anthropic" for real tool-use, "heuristic" for the offline rules-based default.
    policy: str = Field(default="anthropic")
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-opus-4-8")
    max_tokens: int = Field(default=1024)

    # Safety: cap the agent loop so it can never run away.
    max_steps: int = Field(default=8)


def load_settings() -> Settings:
    return Settings()
