# src/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    RERANKING_MODEL: str = "llama3.2:3b"
    
    # Search settings
    DEFAULT_TOP_K: int = 10
    DEFAULT_HYBRID_ALPHA: float = 0.5
    RERANKING_TOP_K: int = 50  # Re-rank top 50 from hybrid search
    
    EMBEDDING_BATCH_SIZE: int = 32
    RERANKING_BATCH_SIZE: int = 5
    ENABLE_CACHING: bool = True

    INDEX_DIR : str = "data/indices/"
    
    class Config:
        env_file = ".env"

settings = Settings()