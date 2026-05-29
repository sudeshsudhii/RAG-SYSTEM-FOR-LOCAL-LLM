"""
Utility functions for the RAG system.

Provides text cleaning, logging setup, and helper functions
used across the pipeline.
"""

import hashlib
import logging
import re
import sys
import unicodedata
from pathlib import Path


def setup_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """
    Create a configured logger with console and file handlers.

    Args:
        name: Logger name (typically module name).
        log_dir: Directory for log files.
        level: Logging level string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / "rag_system.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.

    Operations:
        1. Normalize Unicode (NFKD → NFKC)
        2. Remove control characters (except newlines/tabs)
        3. Fix common encoding artifacts
        4. Normalize whitespace
        5. Strip leading/trailing whitespace

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters (keep newlines and tabs)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Cc" or char in ("\n", "\t", "\r")
    )

    # Fix common encoding artifacts
    replacements = {
        "\u2018": "'",   # Left single quote
        "\u2019": "'",   # Right single quote
        "\u201c": '"',   # Left double quote
        "\u201d": '"',   # Right double quote
        "\u2013": "-",   # En dash
        "\u2014": "--",  # Em dash
        "\u2026": "...", # Ellipsis
        "\u00a0": " ",   # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalize whitespace: collapse multiple spaces/tabs (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


def generate_chunk_id(document_name: str, page: int, chunk_index: int) -> str:
    """
    Generate a deterministic, unique chunk identifier.

    Args:
        document_name: Name of the source document.
        page: Page number (0 for non-paginated documents).
        chunk_index: Sequential index of the chunk within the page.

    Returns:
        A short hash string like 'abc12def'.
    """
    raw = f"{document_name}::page_{page}::chunk_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:10]


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    Truncate text to a maximum length with ellipsis.

    Args:
        text: Text to truncate.
        max_length: Maximum character length.

    Returns:
        Truncated text with '...' if shortened.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes to a human-readable string.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Formatted string (e.g., '1.5 MB').
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
