"""Configuration management for SearchGPT."""

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # LLM Settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    default_llm_provider: str = "openai"
    default_llm_model: str = "gpt-4o-mini"
    
    # Search Settings
    default_top_k: int = 10
    default_hybrid_alpha: float = 0.5
    enable_reranking: bool = True
    
    # Vector Database Settings
    vector_db_path: str = "data/indices"
    embedding_model: str = "text-embedding-3-small"
    
    # Cache Settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Logging
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
