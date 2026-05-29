"""
Tests for the FAISS vector store module.
"""

import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from src.vector_store import FAISSVectorStore, SearchResult


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dimension = 384
        self.store = FAISSVectorStore(dimension=self.dimension)

    def _random_embeddings(self, n: int) -> np.ndarray:
        """Generate random normalized embeddings."""
        embeddings = np.random.randn(n, self.dimension).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    def test_initialization(self):
        """Test vector store initializes correctly."""
        assert len(self.store) == 0
        assert self.store.dimension == self.dimension

    def test_add_documents(self):
        """Test adding documents to the store."""
        embeddings = self._random_embeddings(5)
        contents = [f"Document {i}" for i in range(5)]
        metadata = [{"id": i} for i in range(5)]

        self.store.add_documents(embeddings, contents, metadata)

        assert len(self.store) == 5

    def test_add_documents_mismatched_lengths(self):
        """Test ValueError on mismatched input lengths."""
        embeddings = self._random_embeddings(5)
        contents = [f"Doc {i}" for i in range(3)]
        metadata = [{"id": i} for i in range(5)]

        with pytest.raises(ValueError, match="Mismatched lengths"):
            self.store.add_documents(embeddings, contents, metadata)

    def test_add_documents_wrong_dimension(self):
        """Test ValueError on wrong embedding dimension."""
        embeddings = np.random.randn(5, 128).astype(np.float32)
        contents = [f"Doc {i}" for i in range(5)]
        metadata = [{"id": i} for i in range(5)]

        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            self.store.add_documents(embeddings, contents, metadata)

    def test_search_basic(self):
        """Test basic similarity search."""
        embeddings = self._random_embeddings(10)
        contents = [f"Document {i}" for i in range(10)]
        metadata = [{"id": i} for i in range(10)]

        self.store.add_documents(embeddings, contents, metadata)

        # Search with the first embedding (should find itself)
        results = self.store.search(embeddings[0:1], top_k=3)

        assert len(results) == 3
        assert isinstance(results[0], SearchResult)
        assert results[0].score >= results[1].score  # Sorted by score
        assert results[0].content == "Document 0"  # Exact match should be first

    def test_search_empty_index(self):
        """Test searching an empty index returns empty list."""
        query = self._random_embeddings(1)
        results = self.store.search(query, top_k=5)
        assert results == []

    def test_search_top_k_larger_than_index(self):
        """Test top_k larger than index size returns all results."""
        embeddings = self._random_embeddings(3)
        contents = [f"Doc {i}" for i in range(3)]
        metadata = [{"id": i} for i in range(3)]

        self.store.add_documents(embeddings, contents, metadata)
        results = self.store.search(embeddings[0:1], top_k=10)

        assert len(results) == 3

    def test_save_and_load(self, tmp_path):
        """Test index persistence roundtrip."""
        # Add data
        embeddings = self._random_embeddings(5)
        contents = [f"Document {i}" for i in range(5)]
        metadata = [{"id": i, "document": f"doc{i}.txt"} for i in range(5)]
        self.store.add_documents(embeddings, contents, metadata)

        # Save
        save_path = str(tmp_path / "test_index")
        self.store.save_index(save_path)

        # Load into new store
        new_store = FAISSVectorStore(dimension=self.dimension)
        loaded = new_store.load_index(save_path)

        assert loaded is True
        assert len(new_store) == 5

        # Verify search still works
        results = new_store.search(embeddings[0:1], top_k=1)
        assert len(results) == 1
        assert results[0].content == "Document 0"

    def test_load_nonexistent(self, tmp_path):
        """Test loading from nonexistent path returns False."""
        result = self.store.load_index(str(tmp_path / "nonexistent"))
        assert result is False

    def test_clear(self):
        """Test clearing the vector store."""
        embeddings = self._random_embeddings(5)
        contents = [f"Doc {i}" for i in range(5)]
        metadata = [{"id": i} for i in range(5)]

        self.store.add_documents(embeddings, contents, metadata)
        assert len(self.store) == 5

        self.store.clear()
        assert len(self.store) == 0

    def test_get_indexed_documents(self):
        """Test listing indexed documents."""
        embeddings = self._random_embeddings(3)
        contents = ["A", "B", "C"]
        metadata = [
            {"document": "doc1.pdf"},
            {"document": "doc2.txt"},
            {"document": "doc1.pdf"},  # Duplicate
        ]

        self.store.add_documents(embeddings, contents, metadata)
        docs = self.store.get_indexed_documents()

        assert docs == ["doc1.pdf", "doc2.txt"]  # Sorted, unique

    def test_1d_query_embedding(self):
        """Test that 1D query embeddings are reshaped correctly."""
        embeddings = self._random_embeddings(5)
        contents = [f"Doc {i}" for i in range(5)]
        metadata = [{"id": i} for i in range(5)]
        self.store.add_documents(embeddings, contents, metadata)

        # Pass 1D query
        query = embeddings[0].flatten()
        results = self.store.search(query, top_k=1)
        assert len(results) == 1
