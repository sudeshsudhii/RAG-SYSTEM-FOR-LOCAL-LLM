"""
Tests for the TextChunker module.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.chunking import TextChunker, Chunk
from src.ingestion import Document


class TestTextChunker:
    """Tests for TextChunker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    def test_initialization(self):
        """Test chunker initializes with correct parameters."""
        assert self.chunker.chunk_size == 100
        assert self.chunker.chunk_overlap == 20

    def test_invalid_overlap(self):
        """Test that overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="must be less than chunk size"):
            TextChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="must be less than chunk size"):
            TextChunker(chunk_size=100, chunk_overlap=150)

    def test_empty_text(self):
        """Test chunking empty text returns empty list."""
        assert self.chunker.chunk_text("") == []
        assert self.chunker.chunk_text("   ") == []
        assert self.chunker.chunk_text(None) == []

    def test_short_text(self):
        """Test text shorter than chunk_size produces one chunk."""
        text = "This is a short text."
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_size_respected(self):
        """Test that chunks do not exceed chunk_size (with sentence boundary allowance)."""
        text = " ".join([f"Sentence number {i}." for i in range(50)])
        chunker = TextChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.chunk_text(text)

        # Most chunks should be under or near chunk_size
        for chunk in chunks:
            # Allow some tolerance for sentence boundaries
            assert len(chunk) <= 250, f"Chunk too long: {len(chunk)} chars"

    def test_overlap_exists(self):
        """Test that consecutive chunks share overlapping content."""
        sentences = [f"This is sentence number {i} with some content." for i in range(20)]
        text = " ".join(sentences)
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_text(text)

        if len(chunks) >= 2:
            # Check that consecutive chunks share some text
            for i in range(len(chunks) - 1):
                current_words = set(chunks[i].split())
                next_words = set(chunks[i + 1].split())
                overlap = current_words & next_words
                # There should be some overlap
                assert len(overlap) > 0, (
                    f"No overlap between chunk {i} and {i + 1}"
                )

    def test_sentence_boundary_preservation(self):
        """Test that chunks tend to end at sentence boundaries."""
        text = (
            "First sentence here. Second sentence here. "
            "Third sentence here. Fourth sentence here. "
            "Fifth sentence here."
        )
        chunker = TextChunker(chunk_size=80, chunk_overlap=15)
        chunks = chunker.chunk_text(text)

        # Chunks should end at or near sentence boundaries (periods)
        for chunk in chunks:
            stripped = chunk.strip()
            # Most chunks should end with a period
            assert stripped[-1] in ".!?", (
                f"Chunk does not end at sentence boundary: '{stripped[-20:]}'"
            )

    def test_chunk_documents_metadata(self):
        """Test that chunk_documents preserves and enriches metadata."""
        documents = [
            Document(
                content="This is a test document with enough content to chunk. "
                        "It has multiple sentences. Each one is important.",
                metadata={"document": "test.txt", "page": 1, "source_type": "txt"},
            )
        ]

        chunks = self.chunker.chunk_documents(documents)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["document"] == "test.txt"
            assert chunk.metadata["page"] == 1
            assert chunk.metadata["source_type"] == "txt"
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_documents_multiple_docs(self):
        """Test chunking multiple documents produces sequential chunk indices."""
        documents = [
            Document(
                content="Document one content here. It is the first document.",
                metadata={"document": "doc1.txt", "page": 1, "source_type": "txt"},
            ),
            Document(
                content="Document two content here. It is the second document.",
                metadata={"document": "doc2.txt", "page": 1, "source_type": "txt"},
            ),
        ]

        chunks = self.chunker.chunk_documents(documents)

        # Verify sequential chunk indices
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_documents_empty_list(self):
        """Test chunking empty document list returns empty."""
        assert self.chunker.chunk_documents([]) == []

    def test_chunk_id_uniqueness(self):
        """Test that chunk IDs are unique across chunks."""
        text = " ".join([f"Sentence {i} with some content here." for i in range(30)])
        documents = [
            Document(
                content=text,
                metadata={"document": "test.txt", "page": 1, "source_type": "txt"},
            )
        ]

        chunks = self.chunker.chunk_documents(documents)
        chunk_ids = [c.metadata["chunk_id"] for c in chunks]

        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found"

    def test_default_config(self):
        """Test default chunker configuration."""
        chunker = TextChunker()
        assert chunker.chunk_size == 800
        assert chunker.chunk_overlap == 150
