# Docker Deployment

## Prerequisites
- Docker
- Docker Compose

## Quickstart
1. Rename `.env.example` to `.env` and fill in your API keys:
   ```
   GEMINI_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```
2. Start the container:
   ```bash
   docker compose up -d --build
   ```
3. Access the Streamlit application at `http://localhost:8501`.

## Volumes
The `docker-compose.yml` file persists:
- `/vector_db`: The FAISS index and metadata.
- `/logs`: Standard application logs.
- `/monitoring`: JSON metrics for the Analytics dashboard.
- `/data`: Unprocessed data files.
