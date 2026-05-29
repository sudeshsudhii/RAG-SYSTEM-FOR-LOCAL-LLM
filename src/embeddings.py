"""
Embedding model wrapper.

Provides a clean interface around SentenceTransformers for generating
document and query embeddings with L2 normalization for cosine similarity.
"""

from typing import Optional

import numpy as np

from src.utils import setup_logger

logger = setup_logger(__name__)


class EmbeddingModel:
    """
    Wrapper around SentenceTransformers for text embedding.

    Uses lazy loading to defer model initialization until first use,
    and normalizes embeddings to unit length for cosine similarity
    via inner product.
    """

    _instance: Optional["EmbeddingModel"] = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initialize the embedding model (lazy-loaded).

        Args:
            model_name: HuggingFace model identifier.
        """
        self.model_name = model_name
        self._model = None
        self._dimension: Optional[int] = None

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2") -> "EmbeddingModel":
        """
        Get or create a singleton instance of the embedding model.

        Args:
            model_name: HuggingFace model identifier.

        Returns:
            Singleton EmbeddingModel instance.
        """
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def _load_model(self) -> None:
        """Load the SentenceTransformer model if not already loaded."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded. Dimension: {self._dimension}"
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @property
    def dimension(self) -> int:
        """Get the embedding dimension, loading the model if necessary."""
        self._load_model()
        assert self._dimension is not None
        return self._dimension

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show a progress bar.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            Numpy array of shape (len(texts), dimension).
        """
        self._load_model()

        if not texts:
            return np.array([]).reshape(0, self.dimension)

        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        if normalize:
            embeddings = self._normalize(embeddings)

        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generate an embedding for a single query.

        Args:
            query: Query string to embed.
            normalize: Whether to L2-normalize the embedding.

        Returns:
            Numpy array of shape (1, dimension).
        """
        self._load_model()

        embedding = self._model.encode(
            [query], convert_to_numpy=True
        )

        if normalize:
            embedding = self._normalize(embedding)

        return embedding

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize embeddings to unit length.

        This enables cosine similarity via inner product (dot product).

        Args:
            embeddings: Embedding array to normalize.

        Returns:
            Normalized embedding array.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms
