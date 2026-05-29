"""
Document ingestion module.

Provides loaders for PDF, TXT, and Markdown files using the Strategy pattern.
Each loader extracts text and preserves page-level metadata.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils import clean_text, setup_logger

logger = setup_logger(__name__)


@dataclass
class Document:
    """
    Represents a loaded document page/section.

    Attributes:
        content: Extracted text content.
        metadata: Source metadata (document name, page, source type).
    """

    content: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """
        Load a document and return a list of Document objects.

        Args:
            file_path: Path to the document file.

        Returns:
            List of Document objects, one per page/section.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be parsed.
        """
        ...

    def _validate_file(self, file_path: str) -> Path:
        """Validate that the file exists and return a Path object."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        return path


class PDFLoader(DocumentLoader):
    """Loader for PDF documents using pypdf."""

    def load(self, file_path: str) -> list[Document]:
        """
        Extract text from each page of a PDF.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Document objects, one per page.
        """
        path = self._validate_file(file_path)

        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF loading. Install it with: pip install pypdf"
            )

        documents = []
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)
            logger.info(f"Loading PDF: {path.name} ({total_pages} pages)")

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = clean_text(text)

                if text.strip():
                    documents.append(
                        Document(
                            content=text,
                            metadata={
                                "document": path.name,
                                "page": page_num,
                                "total_pages": total_pages,
                                "source_type": "pdf",
                            },
                        )
                    )

            logger.info(
                f"Extracted {len(documents)} non-empty pages from {path.name}"
            )

        except Exception as e:
            logger.error(f"Failed to load PDF {path.name}: {e}")
            raise ValueError(f"Failed to parse PDF: {e}") from e

        return documents


class TextLoader(DocumentLoader):
    """Loader for plain text files."""

    def load(self, file_path: str) -> list[Document]:
        """
        Load a plain text file as a single document.

        Args:
            file_path: Path to the text file.

        Returns:
            List with a single Document object.
        """
        path = self._validate_file(file_path)
        logger.info(f"Loading text file: {path.name}")

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback to latin-1 for non-UTF8 files
            text = path.read_text(encoding="latin-1")
            logger.warning(f"File {path.name} is not UTF-8, using latin-1 fallback")

        text = clean_text(text)

        if not text.strip():
            logger.warning(f"File {path.name} is empty after cleaning")
            return []

        return [
            Document(
                content=text,
                metadata={
                    "document": path.name,
                    "page": 1,
                    "total_pages": 1,
                    "source_type": "txt",
                },
            )
        ]


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files with optional frontmatter stripping."""

    # YAML frontmatter pattern: --- ... ---
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

    def load(self, file_path: str) -> list[Document]:
        """
        Load a Markdown file, stripping frontmatter.

        Args:
            file_path: Path to the Markdown file.

        Returns:
            List with a single Document object.
        """
        path = self._validate_file(file_path)
        logger.info(f"Loading Markdown file: {path.name}")

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
            logger.warning(f"File {path.name} is not UTF-8, using latin-1 fallback")

        # Strip YAML frontmatter
        text = self.FRONTMATTER_PATTERN.sub("", text)
        text = clean_text(text)

        if not text.strip():
            logger.warning(f"File {path.name} is empty after cleaning")
            return []

        return [
            Document(
                content=text,
                metadata={
                    "document": path.name,
                    "page": 1,
                    "total_pages": 1,
                    "source_type": "md",
                },
            )
        ]


# ── Loader Registry ─────────────────────────────────────────────────────────

LOADER_REGISTRY: dict[str, type[DocumentLoader]] = {
    ".pdf": PDFLoader,
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
}


def load_document(file_path: str) -> list[Document]:
    """
    Factory function to load a document using the appropriate loader.

    Dispatches to the correct loader based on file extension.

    Args:
        file_path: Path to the document.

    Returns:
        List of Document objects.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    loader_class = LOADER_REGISTRY.get(extension)
    if loader_class is None:
        supported = ", ".join(LOADER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported file type: '{extension}'. Supported types: {supported}"
        )

    loader = loader_class()
    return loader.load(file_path)


def get_supported_extensions() -> list[str]:
    """Return a list of supported file extensions."""
    return list(LOADER_REGISTRY.keys())
