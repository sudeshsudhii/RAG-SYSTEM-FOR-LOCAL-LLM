"""
FAISS vector store module.

Provides a persistent vector store backed by FAISS with metadata management.
Supports indexing child chunks and resolving to parent chunks.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class SearchResult:
    """
    A single search result from the vector store.
    """
    content: str
    metadata: dict
    score: float


class FAISSVectorStore:
    """
    FAISS-backed vector store with metadata and parent-chunk persistence.
    """

    INDEX_FILENAME = "faiss_index.bin"
    METADATA_FILENAME = "metadata.pkl"

    def __init__(self, dimension: int = 384) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install it with: pip install faiss-cpu")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store: list[dict] = []
        self.content_store: list[str] = []
        
        # Maps parent_id to parent chunk content
        self.parent_store: dict[str, str] = {}

        logger.info(f"FAISS vector store initialized (dimension={dimension})")

    def add_documents(
        self,
        embeddings: np.ndarray,
        contents: list[str],
        metadata: list[dict],
        parents: list = None
    ) -> None:
        """
        Add child document embeddings to the index and optionally save parents.

        Args:
            embeddings: Numpy array of shape (n, dimension).
            contents: List of child text contents.
            metadata: List of metadata dicts for children.
            parents: Optional list of ParentChunk objects to store.
        """
        if len(embeddings) != len(contents) or len(embeddings) != len(metadata):
            raise ValueError("Mismatched lengths for embeddings, contents, and metadata")

        if len(embeddings) == 0:
            return

        embeddings = embeddings.astype(np.float32)

        self.index.add(embeddings)
        self.content_store.extend(contents)
        self.metadata_store.extend(metadata)

        if parents:
            for p in parents:
                self.parent_store[p.metadata["chunk_id"]] = p.content

        logger.info(
            f"Added {len(embeddings)} vectors. Total index size: {self.index.ntotal}. "
            f"Parent store size: {len(self.parent_store)}"
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        resolve_parents: bool = True
    ) -> list[SearchResult]:
        """
        Search the index. Optionally resolve to parent chunks.
        """
        if self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)
        effective_k = min(top_k * 3, self.index.ntotal) # Fetch more to deduplicate parents

        scores, indices = self.index.search(query_embedding, effective_k)

        results: list[SearchResult] = []
        seen_parents = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            child_metadata = self.metadata_store[idx]
            parent_id = child_metadata.get("parent_id")

            if resolve_parents and parent_id and parent_id in self.parent_store:
                if parent_id in seen_parents:
                    continue  # Skip duplicate parents
                
                seen_parents.add(parent_id)
                
                # Create a merged metadata prioritizing parent reference
                meta = child_metadata.copy()
                meta["type"] = "parent"
                meta["child_match_score"] = float(score)
                
                results.append(
                    SearchResult(
                        content=self.parent_store[parent_id],
                        metadata=meta,
                        score=float(score),
                    )
                )
            else:
                # Return raw child
                results.append(
                    SearchResult(
                        content=self.content_store[idx],
                        metadata=child_metadata,
                        score=float(score),
                    )
                )
                
            if len(results) >= top_k:
                break

        return results

    def save_index(self, path: str) -> None:
        import faiss
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(save_dir / self.INDEX_FILENAME))
        
        with open(save_dir / self.METADATA_FILENAME, "wb") as f:
            pickle.dump({
                "metadata_store": self.metadata_store,
                "content_store": self.content_store,
                "parent_store": self.parent_store,
                "dimension": self.dimension,
            }, f)

    def load_index(self, path: str) -> bool:
        import faiss
        save_dir = Path(path)
        index_path = save_dir / self.INDEX_FILENAME
        metadata_path = save_dir / self.METADATA_FILENAME

        if not index_path.exists() or not metadata_path.exists():
            return False

        try:
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.metadata_store = data["metadata_store"]
                self.content_store = data["content_store"]
                self.parent_store = data.get("parent_store", {})
                self.dimension = data["dimension"]
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def clear(self) -> None:
        import faiss
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = []
        self.content_store = []
        self.parent_store = {}

    def delete_saved_index(self, path: str) -> None:
        save_dir = Path(path)
        for filename in [self.INDEX_FILENAME, self.METADATA_FILENAME]:
            filepath = save_dir / filename
            if filepath.exists():
                filepath.unlink()

    def __len__(self) -> int:
        return self.index.ntotal

    def get_indexed_documents(self) -> list[str]:
        doc_names = {meta.get("document", "unknown") for meta in self.metadata_store}
        return sorted(doc_names)
