from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables / .env. No secrets in code."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    model_dir: str = "models"

    # Online verification providers (empty key => provider reports "not configured")
    google_fact_check_api_key: str = ""
    google_fact_check_language: str = "en"
    gdelt_enabled: bool = True
    gdelt_min_interval_seconds: float = 5.5  # GDELT asks for ~1 request / 5 s per IP

    # Limits
    rate_limit_per_minute: int = 30
    request_timeout_seconds: float = 10.0
    verification_timeout_seconds: float = 40.0
    max_article_chars: int = 20000
    max_fetch_bytes: int = 2_000_000
    max_redirects: int = 3

    # Evidence assessment
    sources_config: str = "configs/sources.yaml"
    max_claims_searched: int = 2
    relevance_threshold: float = 0.5
    factcheck_relevance_threshold: float = 0.4
    support_relevance_threshold: float = 0.7
    min_independent_sources: int = 3   # news-only "Supported" needs this many distinct domains
    min_debunk_sources: int = 2        # news-only "Contradicted" needs this many distinct domains
    news_max_age_days: int = 365

    feedback_path: str = "data/feedback/feedback.jsonl"

    # Monitoring (events hold derived numbers only, never article text)
    monitoring_enabled: bool = True
    monitoring_log_path: str = "data/monitoring/events.jsonl"
    drift_window: int = 2000
    drift_min_samples: int = 100
    min_review_labels: int = 50
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
