"""
Tests for the document ingestion module.
"""

import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.ingestion import (
    load_document,
    get_supported_extensions,
    TextLoader,
    MarkdownLoader,
    Document,
)


class TestTextLoader:
    """Tests for TextLoader."""

    def test_load_text_file(self, tmp_path):
        """Test loading a basic text file."""
        file = tmp_path / "test.txt"
        file.write_text("Hello, world! This is a test document.", encoding="utf-8")

        loader = TextLoader()
        docs = loader.load(str(file))

        assert len(docs) == 1
        assert "Hello, world!" in docs[0].content
        assert docs[0].metadata["document"] == "test.txt"
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["source_type"] == "txt"

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty text file returns empty list."""
        file = tmp_path / "empty.txt"
        file.write_text("", encoding="utf-8")

        loader = TextLoader()
        docs = loader.load(str(file))

        assert len(docs) == 0

    def test_file_not_found(self):
        """Test FileNotFoundError for missing files."""
        loader = TextLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.txt")

    def test_load_multiline(self, tmp_path):
        """Test loading a multi-line text file."""
        content = "Line 1.\nLine 2.\nLine 3."
        file = tmp_path / "multi.txt"
        file.write_text(content, encoding="utf-8")

        loader = TextLoader()
        docs = loader.load(str(file))

        assert len(docs) == 1
        assert "Line 1." in docs[0].content


class TestMarkdownLoader:
    """Tests for MarkdownLoader."""

    def test_load_markdown(self, tmp_path):
        """Test loading a basic markdown file."""
        file = tmp_path / "test.md"
        file.write_text("# Hello\n\nThis is **markdown** content.", encoding="utf-8")

        loader = MarkdownLoader()
        docs = loader.load(str(file))

        assert len(docs) == 1
        assert "Hello" in docs[0].content
        assert docs[0].metadata["source_type"] == "md"

    def test_strip_frontmatter(self, tmp_path):
        """Test that YAML frontmatter is stripped."""
        content = """---
title: Test
date: 2024-01-01
---

# Actual Content

This is the body."""
        file = tmp_path / "front.md"
        file.write_text(content, encoding="utf-8")

        loader = MarkdownLoader()
        docs = loader.load(str(file))

        assert len(docs) == 1
        assert "title:" not in docs[0].content
        assert "Actual Content" in docs[0].content

    def test_load_markdown_no_frontmatter(self, tmp_path):
        """Test loading markdown without frontmatter."""
        file = tmp_path / "nofm.md"
        file.write_text("# Simple\n\nJust content.", encoding="utf-8")

        loader = MarkdownLoader()
        docs = loader.load(str(file))

        assert len(docs) == 1
        assert "Simple" in docs[0].content


class TestLoadDocument:
    """Tests for the load_document factory function."""

    def test_dispatch_txt(self, tmp_path):
        """Test factory dispatches to TextLoader for .txt."""
        file = tmp_path / "doc.txt"
        file.write_text("Test content.", encoding="utf-8")

        docs = load_document(str(file))
        assert len(docs) == 1
        assert docs[0].metadata["source_type"] == "txt"

    def test_dispatch_md(self, tmp_path):
        """Test factory dispatches to MarkdownLoader for .md."""
        file = tmp_path / "doc.md"
        file.write_text("# Test\n\nContent.", encoding="utf-8")

        docs = load_document(str(file))
        assert len(docs) == 1
        assert docs[0].metadata["source_type"] == "md"

    def test_unsupported_format(self, tmp_path):
        """Test ValueError for unsupported file formats."""
        file = tmp_path / "doc.docx"
        file.write_text("test", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(str(file))

    def test_get_supported_extensions(self):
        """Test supported extensions list."""
        exts = get_supported_extensions()
        assert ".pdf" in exts
        assert ".txt" in exts
        assert ".md" in exts
