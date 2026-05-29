"""
Text chunking module.

Implements Parent-Child chunking for advanced retrieval.
Documents are split into larger Parent Chunks (for LLM context), 
which are subdivided into smaller Child Chunks (for dense vector search).
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.ingestion import Document
from src.utils import generate_chunk_id, setup_logger

logger = setup_logger(__name__)


@dataclass
class Chunk:
    """Base chunk dataclass."""
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None

@dataclass
class ParentChunk(Chunk):
    """
    A larger text chunk meant to provide broad context to the LLM.
    """
    pass

@dataclass
class ChildChunk(Chunk):
    """
    A smaller text chunk meant for highly specific vector retrieval.
    Contains a reference to its ParentChunk.
    """
    parent_id: str = ""


class ParentChildChunker:
    """
    Sentence-boundary-aware chunker that creates parent and child chunks.
    """

    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(
        self, 
        parent_chunk_size: int = 2000, 
        parent_chunk_overlap: int = 200,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 50
    ) -> None:
        """
        Initialize the chunker.

        Args:
            parent_chunk_size: Max characters per parent chunk.
            parent_chunk_overlap: Character overlap between parent chunks.
            child_chunk_size: Max characters per child chunk.
            child_chunk_overlap: Character overlap between child chunks.
        """
        if parent_chunk_overlap >= parent_chunk_size:
            raise ValueError("Parent overlap must be less than parent chunk size")
        if child_chunk_size > parent_chunk_size:
            raise ValueError("Child chunk size cannot exceed parent chunk size")
            
        self.parent_size = parent_chunk_size
        self.parent_overlap = parent_chunk_overlap
        self.child_size = child_chunk_size
        self.child_overlap = child_chunk_overlap
        
        logger.info(
            f"ParentChildChunker init: Parent({self.parent_size}, {self.parent_overlap}) "
            f"Child({self.child_size}, {self.child_overlap})"
        )

    def _split_into_sentences(self, text: str) -> list[str]:
        sentences = self.SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            sentences = [s.strip() for s in text.split("\n") if s.strip()]
        return sentences

    def _merge_into_blocks(self, sentences: list[str], max_size: int, overlap: int) -> list[str]:
        if not sentences:
            return []

        blocks: list[str] = []
        current_block: list[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if sentence_length > max_size:
                if current_block:
                    blocks.append(" ".join(current_block))
                    current_block = []
                    current_length = 0

                for i in range(0, sentence_length, max_size - overlap):
                    sub = sentence[i : i + max_size]
                    if sub.strip():
                        blocks.append(sub.strip())
                continue

            new_length = current_length + sentence_length + (1 if current_block else 0)

            if new_length > max_size and current_block:
                blocks.append(" ".join(current_block))

                overlap_block: list[str] = []
                overlap_length = 0

                for prev_sentence in reversed(current_block):
                    test_length = overlap_length + len(prev_sentence) + (1 if overlap_block else 0)
                    if test_length <= overlap:
                        overlap_block.insert(0, prev_sentence)
                        overlap_length = test_length
                    else:
                        break

                current_block = overlap_block
                current_length = overlap_length

            current_block.append(sentence)
            current_length += sentence_length + (1 if len(current_block) > 1 else 0)

        if current_block:
            blocks.append(" ".join(current_block))

        return blocks

    def chunk_documents(self, documents: list[Document]) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        Chunk documents into a hierarchy of Parents and Children.

        Returns:
            Tuple of (list[ParentChunk], list[ChildChunk])
        """
        all_parents: list[ParentChunk] = []
        all_children: list[ChildChunk] = []
        
        parent_index = 0
        child_index = 0

        for doc in documents:
            doc_name = doc.metadata.get("document", "unknown")
            page = doc.metadata.get("page", 1)

            # 1. Create Parent Chunks
            sentences = self._split_into_sentences(doc.content)
            parent_texts = self._merge_into_blocks(sentences, self.parent_size, self.parent_overlap)
            
            for p_text in parent_texts:
                parent_id = generate_chunk_id(doc_name, page, parent_index)
                parent_chunk = ParentChunk(
                    content=p_text,
                    metadata={
                        "document": doc_name,
                        "page": page,
                        "chunk_id": parent_id,
                        "type": "parent"
                    }
                )
                all_parents.append(parent_chunk)
                parent_index += 1
                
                # 2. Create Child Chunks derived directly from this Parent Chunk
                child_sentences = self._split_into_sentences(p_text)
                child_texts = self._merge_into_blocks(child_sentences, self.child_size, self.child_overlap)
                
                for c_text in child_texts:
                    child_id = generate_chunk_id(f"{doc_name}_child", page, child_index)
                    child_chunk = ChildChunk(
                        content=c_text,
                        parent_id=parent_id,
                        metadata={
                            "document": doc_name,
                            "page": page,
                            "chunk_id": child_id,
                            "parent_id": parent_id,
                            "type": "child"
                        }
                    )
                    all_children.append(child_chunk)
                    child_index += 1

        logger.info(
            f"Chunked {len(documents)} document(s) into {len(all_parents)} parents "
            f"and {len(all_children)} children."
        )
        return all_parents, all_children
