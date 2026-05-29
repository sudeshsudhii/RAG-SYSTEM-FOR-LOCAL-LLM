"""
Retrieval module.

Implements vector retrieval, BM25 hybrid search with Reciprocal Rank Fusion,
and cross-encoder re-ranking for precision improvement.
Supports resolving to Parent Chunks.
"""

from typing import Optional

import numpy as np

from src.chunking import Chunk
from src.embeddings import EmbeddingModel
from src.vector_store import FAISSVectorStore, SearchResult
from src.utils import setup_logger

logger = setup_logger(__name__)


class Retriever:
    """
    Retrieval pipeline supporting vector search, hybrid search, and re-ranking.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_model: EmbeddingModel,
        confidence_threshold: float = 0.3,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.confidence_threshold = confidence_threshold
        self._cross_encoder = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_threshold: bool = True,
    ) -> list[SearchResult]:
        """Basic vector similarity retrieval, resolving to Parent chunks."""
        if len(self.vector_store) == 0:
            return []

        query_embedding = self.embedding_model.embed_query(query)
        # Vector store handles parent resolution natively now
        results = self.vector_store.search(query_embedding, top_k=top_k, resolve_parents=True)

        if filter_threshold:
            results = [r for r in results if r.score >= self.confidence_threshold]

        logger.info(f"Vector retrieval: results={len(results)}/{top_k}")
        return results

    def hybrid_retrieve(
        self,
        query: str,
        child_chunks: list[Chunk],
        top_k: int = 5,
        rrf_k: int = 60,
        vector_weight: float = 0.6,
    ) -> list[SearchResult]:
        """Hybrid retrieval combining BM25 and vector search via RRF."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return self.retrieve(query, top_k=top_k)

        if not child_chunks:
            return self.retrieve(query, top_k=top_k)

        # ── BM25 retrieval on children ──
        tokenized_chunks = [chunk.content.lower().split() for chunk in child_chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_rankings = np.argsort(bm25_scores)[::-1][:top_k * 4] # higher recall for RRF

        # ── Vector retrieval on children (without parent resolution for now) ──
        query_embedding = self.embedding_model.embed_query(query)
        vector_results_raw = self.vector_store.search(query_embedding, top_k=top_k * 4, resolve_parents=False)

        # ── Reciprocal Rank Fusion ──
        fused_scores: dict[int, float] = {}
        bm25_weight = 1.0 - vector_weight

        # BM25 contribution
        for rank, chunk_idx in enumerate(bm25_rankings):
            chunk_idx = int(chunk_idx)
            rrf_score = bm25_weight / (rrf_k + rank + 1)
            fused_scores[chunk_idx] = fused_scores.get(chunk_idx, 0) + rrf_score

        # Vector contribution
        chunk_id_to_idx = {c.metadata.get("chunk_id"): i for i, c in enumerate(child_chunks)}
        for rank, result in enumerate(vector_results_raw):
            chunk_id = result.metadata.get("chunk_id")
            chunk_idx = chunk_id_to_idx.get(chunk_id)
            if chunk_idx is not None:
                rrf_score = vector_weight / (rrf_k + rank + 1)
                fused_scores[chunk_idx] = fused_scores.get(chunk_idx, 0) + rrf_score

        # Sort fused scores
        sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        # Resolve top children back to parents
        results = []
        seen_parents = set()
        
        for idx in sorted_indices:
            if idx < len(child_chunks):
                child = child_chunks[idx]
                parent_id = child.metadata.get("parent_id")
                
                if parent_id and parent_id in self.vector_store.parent_store:
                    if parent_id in seen_parents:
                        continue
                    seen_parents.add(parent_id)
                    
                    meta = child.metadata.copy()
                    meta["type"] = "parent"
                    
                    results.append(SearchResult(
                        content=self.vector_store.parent_store[parent_id],
                        metadata=meta,
                        score=float(fused_scores[idx])
                    ))
                else:
                    results.append(SearchResult(
                        content=child.content,
                        metadata=child.metadata,
                        score=float(fused_scores[idx])
                    ))
            
            if len(results) >= top_k:
                break

        return results

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> list[SearchResult]:
        """Re-rank results using a cross-encoder model."""
        if not results:
            return []

        try:
            from sentence_transformers import CrossEncoder

            if self._cross_encoder is None:
                self._cross_encoder = CrossEncoder(model_name)

            pairs = [(query, r.content) for r in results]
            scores = self._cross_encoder.predict(pairs)

            reranked = []
            for result, score in zip(results, scores):
                reranked.append(
                    SearchResult(
                        content=result.content,
                        metadata=result.metadata,
                        score=float(score),
                    )
                )

            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked[:top_k]

        except ImportError:
            return results[:top_k]
        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            return results[:top_k]
