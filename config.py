"""
Centralized configuration for the RAG system.

All tunable parameters are defined here as dataclass fields,
making them easy to override and document.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""

    # ── Chunking (Parent-Child) ──────────────────────────────
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 50

    # ── Embedding Model ──────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Retrieval ────────────────────────────────────────────
    top_k: int = 5
    confidence_threshold: float = 0.3
    enable_hybrid_search: bool = False
    enable_reranking: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── LLM Providers ────────────────────────────────────────
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").lower())
    
    gemini_model: str = "gemini-2.0-flash"
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = "gpt-4o-mini"
    
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = "llama3"

    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ── Cost Estimation ($ per 1M tokens) ────────────────────
    cost_per_1m_input: float = 0.15   # Approx Gemini 2.0 Flash pricing
    cost_per_1m_output: float = 0.60  # Approx Gemini 2.0 Flash pricing

    # ── Paths ────────────────────────────────────────────────
    vector_db_path: str = "vector_db"
    data_path: str = "data"
    log_path: str = "logs"
    monitoring_path: str = "monitoring"

    # ── Logging ──────────────────────────────────────────────
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Create required directories on initialization."""
        for dir_path in [self.vector_db_path, self.data_path, self.log_path, self.monitoring_path]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

# Singleton configuration instance
config = RAGConfig()
