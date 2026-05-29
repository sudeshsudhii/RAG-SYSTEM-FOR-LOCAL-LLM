"""
Tests for the Retriever module.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.vector_store import FAISSVectorStore, SearchResult
from src.retriever import Retriever
from src.chunking import Chunk


class TestRetriever:
    """Tests for the Retriever class."""

    def setup_method(self):
        """Set up test fixtures with a mock embedding model."""
        self.dimension = 384
        self.store = FAISSVectorStore(dimension=self.dimension)

        # Create a mock embedding model
        self.mock_embedding = MagicMock()
        self.mock_embedding.dimension = self.dimension

        self.retriever = Retriever(
            vector_store=self.store,
            embedding_model=self.mock_embedding,
            confidence_threshold=0.3,
        )

    def _random_embeddings(self, n: int) -> np.ndarray:
        """Generate random normalized embeddings."""
        embeddings = np.random.randn(n, self.dimension).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def _populate_store(self, n: int = 10):
        """Add test data to the vector store."""
        embeddings = self._random_embeddings(n)
        contents = [f"Test content for chunk {i}." for i in range(n)]
        metadata = [
            {
                "document": f"doc{i % 3}.pdf",
                "page": i % 5 + 1,
                "chunk_id": f"chunk_{i}",
            }
            for i in range(n)
        ]
        self.store.add_documents(embeddings, contents, metadata)
        return embeddings

    def test_retrieve_empty_index(self):
        """Test retrieval on empty index returns empty list."""
        self.mock_embedding.embed_query.return_value = self._random_embeddings(1)
        results = self.retriever.retrieve("test query")
        assert results == []

    def test_retrieve_basic(self):
        """Test basic vector retrieval."""
        embeddings = self._populate_store(10)

        # Mock embed_query to return the first embedding
        self.mock_embedding.embed_query.return_value = embeddings[0:1]

        results = self.retriever.retrieve("test query", top_k=3)

        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_retrieve_with_threshold_filtering(self):
        """Test that low-confidence results are filtered."""
        embeddings = self._populate_store(10)

        # Use a random query that might have low similarity
        random_query = self._random_embeddings(1)
        self.mock_embedding.embed_query.return_value = random_query

        # Set high threshold
        self.retriever.confidence_threshold = 0.99

        results = self.retriever.retrieve("test query", top_k=5, filter_threshold=True)

        for result in results:
            assert result.score >= 0.99

    def test_retrieve_without_threshold_filtering(self):
        """Test retrieval without threshold filtering."""
        embeddings = self._populate_store(10)
        self.mock_embedding.embed_query.return_value = self._random_embeddings(1)

        results = self.retriever.retrieve(
            "test query", top_k=5, filter_threshold=False
        )

        assert len(results) == 5

    def test_hybrid_retrieve_fallback(self):
        """Test hybrid search falls back when rank_bm25 unavailable."""
        embeddings = self._populate_store(5)
        self.mock_embedding.embed_query.return_value = embeddings[0:1]

        chunks = [
            Chunk(
                content=f"Chunk {i}",
                metadata={"chunk_id": f"chunk_{i}"},
            )
            for i in range(5)
        ]

        # Should work even without rank_bm25 (falls back to vector-only)
        results = self.retriever.hybrid_retrieve(
            "test query", chunks, top_k=3
        )
        assert isinstance(results, list)

    def test_rerank_empty(self):
        """Test re-ranking empty results returns empty list."""
        results = self.retriever.rerank("test query", [], top_k=3)
        assert results == []

    def test_rerank_without_cross_encoder(self):
        """Test re-ranking falls back when cross-encoder unavailable."""
        results = [
            SearchResult(content="A", metadata={}, score=0.8),
            SearchResult(content="B", metadata={}, score=0.6),
        ]

        # Should gracefully handle missing cross-encoder
        reranked = self.retriever.rerank("test", results, top_k=2)
        assert len(reranked) <= 2
