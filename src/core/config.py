# src/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Lightweight sentence-transformers model from Hugging Face I used it in my LinkedIn project
    RERANKING_MODEL: str = "llama3.2:3b"
    
    # Search settings
    DEFAULT_TOP_K: int = 10
    DEFAULT_HYBRID_ALPHA: float = 0.5
    RERANKING_TOP_K: int = 50  # Re-rank top 50 from hybrid search
    
    EMBEDDING_BATCH_SIZE: int = 32
    RERANKING_BATCH_SIZE: int = 5
    ENABLE_CACHING: bool = True

    INDEX_DIR: str = "data/indices/"
    DATASET_PATH: str = "data/datasets/"
    
    # Additional paths for SearchManager
    data_path: str = "data/"
    vector_db_path: str = "data/indices/"
    
    class Config:
        env_file = ".env"

settings = Settings()

def get_settings() -> Settings:
    """Get application settings."""
    return settings