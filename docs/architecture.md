# System Architecture

## Overview
This RAG system implements a modular architecture supporting multiple LLM backends, hybrid retrieval, and real-time observability.

## Components
1. **Ingestion Layer**: Supports PDF, TXT, MD via Strategy pattern.
2. **Chunking Layer**: Implements Parent-Child chunking. Sentences are grouped into Parent Chunks (2000 chars) for LLM context, which are subdivided into Child Chunks (400 chars) for embedding.
3. **Vector Store**: FAISS `IndexFlatIP`. Child embeddings are indexed; metadata maps them to Parent chunks stored alongside the index.
4. **Retrieval Layer**: Supports Dense Vector search and Hybrid search (BM25) fused with Reciprocal Rank Fusion (RRF), followed by Cross-Encoder re-ranking.
5. **Generation Layer**: `LLMFactory` abstracts Gemini, OpenAI, and Ollama.
6. **Observability Layer**: `RAGLogger` writes JSONL metrics for token tracking, latency, and cost estimation.
7. **UI**: Streamlit multi-page interface (Chat & Analytics).

## Routing
Queries are pre-processed by a `QueryClassifier` to determine if they are Greetings, Small Talk, Unsupported, or valid RAG queries, reducing unnecessary retrieval latency and cost.
