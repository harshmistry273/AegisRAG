"""
AegisRAG Configuration Module
Lightweight, resilient, type-safe settings management using standard pydantic and dotenv.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Automatically locate and load .env file
BASE_DIR = Path(__file__).resolve().parent.parent
env_file = BASE_DIR / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()


class Settings:
    """Centralized configuration object for AegisRAG."""

    def __init__(self):
        # Project metadata
        self.PROJECT_NAME: str = "AegisRAG: Enterprise Corrective RAG System"
        self.VERSION: str = "1.0.0"
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

        # Base directories
        self.BASE_DIR: Path = BASE_DIR
        self.DATA_DIR: Path = BASE_DIR / "data"
        self.PERSIST_DIRECTORY: Path = BASE_DIR / "data" / "vector_db"

        # Groq Inference Settings
        self.GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
        self.GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
        self.GROQ_MAX_TOKENS: int = int(os.getenv("GROQ_MAX_TOKENS", "2048"))

        # Embedding Settings
        self.EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
        self.EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
        self.EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

        # Ingestion & Chunking
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

        # Retrieval Settings
        self.BM25_K1: float = float(os.getenv("BM25_K1", "1.5"))
        self.BM25_B: float = float(os.getenv("BM25_B", "0.75"))
        self.RRF_K: int = int(os.getenv("RRF_K", "60"))
        self.RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
        self.RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))

        # Agentic Corrective RAG (CRAG) Parameters
        self.RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.60"))
        self.ENABLE_QUERY_REWRITE: bool = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() == "true"
        self.ENABLE_WEB_SEARCH_FALLBACK: bool = os.getenv("ENABLE_WEB_SEARCH_FALLBACK", "true").lower() == "true"
        self.ENABLE_HALLUCINATION_CHECK: bool = os.getenv("ENABLE_HALLUCINATION_CHECK", "true").lower() == "true"
        self.MAX_REWRITE_ATTEMPTS: int = int(os.getenv("MAX_REWRITE_ATTEMPTS", "2"))

        # Server Settings
        self.HOST: str = os.getenv("HOST", "0.0.0.0")
        self.PORT: int = int(os.getenv("PORT", "8000"))

        # Ensure runtime directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
