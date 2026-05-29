"""
Observability and Monitoring module.

Provides a unified logging interface to track queries, retrieval performance,
LLM generation metrics, token usage, and errors. Logs are stored as structured JSON.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler

from src.utils import setup_logger

logger = setup_logger(__name__)

class RAGLogger:
    """
    Structured logger for the RAG pipeline.
    Writes JSON logs to a rotating file in the monitoring directory.
    """
    
    def __init__(self, log_dir: str = "monitoring"):
        """Initialize the RAG logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_logger = logging.getLogger("rag_metrics")
        self.metrics_logger.setLevel(logging.INFO)
        
        if not self.metrics_logger.handlers:
            log_file = self.log_dir / "rag_metrics.jsonl"
            handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
            )
            # Just log the raw JSON message without standard formatter
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.metrics_logger.addHandler(handler)

    def _log_event(self, event_type: str, data: dict):
        """Write a structured JSON log entry."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            **data
        }
        self.metrics_logger.info(json.dumps(log_entry))

    def log_query(self, query_id: str, original_query: str, rewritten_query: str = None, category: str = "RAG Query"):
        """Log incoming user query."""
        self._log_event("query", {
            "query_id": query_id,
            "original_query": original_query,
            "rewritten_query": rewritten_query or original_query,
            "category": category
        })

    def log_retrieval(self, query_id: str, num_chunks: int, avg_score: float, latency_ms: float):
        """Log retrieval metrics."""
        self._log_event("retrieval", {
            "query_id": query_id,
            "num_chunks_retrieved": num_chunks,
            "avg_retrieval_score": avg_score,
            "retrieval_latency_ms": latency_ms
        })

    def log_generation(
        self, 
        query_id: str, 
        prompt_tokens: int, 
        completion_tokens: int, 
        total_tokens: int, 
        estimated_cost: float, 
        latency_ms: float,
        provider: str,
        model: str
    ):
        """Log LLM generation metrics including cost and token usage."""
        self._log_event("generation", {
            "query_id": query_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "generation_latency_ms": latency_ms,
            "provider": provider,
            "model": model
        })

    def log_error(self, query_id: str, error_message: str, component: str):
        """Log system errors."""
        self._log_event("error", {
            "query_id": query_id,
            "error_message": error_message,
            "component": component
        })

# Singleton instance
rag_logger = RAGLogger()
