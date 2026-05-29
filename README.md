# Local LLM RAG Pipeline

## Overview

This repository contains a local Retrieval-Augmented Generation (RAG) pipeline designed to ground Large Language Model (LLM) responses in custom document stores. The primary objective is to implement a robust information retrieval layer that mitigates hallucination issues inherent in generic LLMs while allowing for easy swapping of the underlying generation models (e.g., local inference vs. cloud APIs).

## Architecture

The system is decoupled into four primary stages:

1. **Document Ingestion & Chunking**: 
   Standard static chunking often leads to context fragmentation. This pipeline implements a Parent-Child chunking strategy. Documents are parsed into larger Parent chunks (~2000 chars) for LLM context, which are then subdivided into Child chunks (~400 chars) for precise vector matching.

2. **Vector Indexing & Retrieval**:
   - **Dense Retrieval**: Child chunks are embedded using `all-MiniLM-L6-v2` and indexed in a local FAISS `IndexFlatIP` database.
   - **Sparse Retrieval**: BM25 keyword search runs in parallel to catch exact-match terms that dense embeddings occasionally miss.
   - **Rank Fusion**: The results are merged using Reciprocal Rank Fusion (RRF).

3. **Re-ranking**:
   An optional MS-MARCO Cross-Encoder evaluates the top-k results from the RRF step to calculate exact query-document relevance, discarding low-confidence matches.

4. **Generation & Telemetry**:
   The `LLMFactory` abstracts the generation layer, currently supporting Gemini, OpenAI, and Ollama. A telemetry module intercepts generation calls to log token usage, latency, and estimated costs to a local JSONL file.

## Project Structure

```text
├── app.py                  # Streamlit entry point
├── config.py               # Centralized configuration variables
├── docker-compose.yml      # Docker deployment config
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── docs/                   # Markdown documentation for sub-systems
├── evaluation/             # Scripts for IR metrics (Precision/Recall)
├── src/                    
│   ├── analytics_page.py   # Plotly-based telemetry dashboard
│   ├── chunking.py         # Parent-child chunking logic
│   ├── classifier.py       # Query intent router
│   ├── embeddings.py       # Embedding model wrapper
│   ├── generator.py        # LLM factory and orchestration
│   ├── ingestion.py        # Document parsing
│   ├── observability.py    # JSONL structured logging
│   ├── prompts.py          # System prompts and templates
│   ├── retriever.py        # Hybrid search and RRF logic
│   ├── utils.py            # Helpers
│   └── vector_store.py     # FAISS interactions
└── vector_db/              # Persistent index storage
```

## Setup & Installation

### Local Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/sudeshsudhii/RAG-SYSTEM-FOR-LOCAL-LLM.git
   cd RAG-SYSTEM-FOR-LOCAL-LLM
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`:
   ```env
   GEMINI_API_KEY=your_api_key
   OPENAI_API_KEY=your_api_key
   OLLAMA_BASE_URL=http://localhost:11434
   LLM_PROVIDER=gemini
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Docker Deployment

To spin up the containerized stack:
```bash
docker compose up -d --build
```
The Streamlit interface will be available at `http://localhost:8501`.

## Trade-offs and Design Decisions

- **FAISS vs. Cloud Vector DBs**: FAISS `IndexFlatIP` was chosen for local development simplicity and zero-cost operation. However, because it operates entirely in-memory, it is not suitable for horizontal scaling. For a production deployment, this layer should be swapped for a distributed database like Qdrant or Milvus.
- **Synchronous Ingestion**: Document parsing and indexing currently block the main Streamlit thread. While acceptable for a prototype or small document sets, heavy ingestion workloads require decoupling via a task queue (e.g., Celery/Redis).
- **Cross-Encoder Latency**: The MS-MARCO cross-encoder significantly improves retrieval precision but adds ~1-2 seconds of latency depending on CPU constraints. It is heavily recommended to disable this flag in `config.py` if running on severely resource-constrained hardware.

## Evaluation

The `evaluation/` directory contains automated benchmarking scripts to validate system performance against a static dataset (`benchmark_dataset.json`).

- `retrieval_eval.py`: Measures Information Retrieval metrics including Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
- `answer_eval.py`: Implements an LLM-as-a-judge pattern to score response Groundedness and Completeness against expected ground truths.

## License

This project is provided as an open-source reference architecture. See `LICENSE` for details.
