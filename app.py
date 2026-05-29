"""
RAG Document Q&A System — Streamlit Application.

A production-grade, modular Retrieval-Augmented Generation system
for document question-answering with source citations and analytics.
"""

import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st

from config import RAGConfig
from src.ingestion import load_document, get_supported_extensions
from src.chunking import ParentChildChunker, Chunk
from src.embeddings import EmbeddingModel
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from src.generator import RAGGenerator, LLMFactory, RAGResponse
from src.utils import setup_logger, format_file_size, truncate_text
from src.analytics_page import render_analytics

logger = setup_logger(__name__)

# Max file size: 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024 


# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    /* Global Theme & typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #e0e0ff, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 1rem; color: #c4b5fd; }

    /* Source Cards & Chunks (same as before) */
    .source-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #2a2a4a 100%);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .source-card .doc-name { color: #a78bfa; font-weight: 600; font-size: 0.9rem; }
    .source-card .page-info { color: #94a3b8; font-size: 0.8rem; }
    
    .chunk-card {
        background: rgba(30, 30, 63, 0.6);
        border: 1px solid rgba(100, 100, 180, 0.15);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    
    .chunk-score {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .score-high { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .score-medium { background: rgba(250, 204, 21, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); }
    .score-low { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .conf-high { background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.1)); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .conf-medium { background: linear-gradient(135deg, rgba(250, 204, 21, 0.2), rgba(250, 204, 21, 0.1)); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); }
    .conf-low { background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1)); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    .stat-card {
        background: linear-gradient(135deg, #1e1e3f 0%, #2a2a4a 100%);
        border: 1px solid rgba(167, 139, 250, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    .stat-card .stat-value { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
    .stat-card .stat-label { font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem; }

    .file-item {
        background: rgba(30, 30, 63, 0.4);
        border: 1px solid rgba(100, 100, 180, 0.1);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
    }
    .file-item .file-name { color: #c4b5fd; font-weight: 500; }
    .file-item .file-ext { color: #64748b; font-size: 0.75rem; }

    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.3), transparent);
        margin: 1.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State ────────────────────────────────────────────────────────────

def init_session_state() -> None:
    defaults = {
        "config": RAGConfig(),
        "messages": [],
        "child_chunks": [],
        "indexed_files": [],
        "vector_store_initialized": False,
        "page": "Chat",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ── Component Initialization ─────────────────────────────────────────────────

@st.cache_resource
def load_embedding_model(model_name: str) -> EmbeddingModel:
    model = EmbeddingModel(model_name)
    _ = model.dimension
    return model

def get_vector_store() -> FAISSVectorStore:
    if "vector_store" not in st.session_state:
        config = st.session_state.config
        store = FAISSVectorStore(dimension=config.embedding_dimension)
        if store.load_index(config.vector_db_path):
            st.session_state.indexed_files = store.get_indexed_documents()
            st.session_state.vector_store_initialized = True
        st.session_state.vector_store = store
    return st.session_state.vector_store


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    config = st.session_state.config

    with st.sidebar:
        st.title("Navigation")
        st.session_state.page = st.radio("Go to", ["Chat", "Analytics"], label_visibility="collapsed")
        
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        
        st.markdown("## 📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents (Max 10MB)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            valid_files = []
            for f in uploaded_files:
                if f.size > MAX_FILE_SIZE:
                    st.error(f"{f.name} exceeds 10MB limit.")
                else:
                    valid_files.append(f)
                    
            if valid_files and st.button("📥 Index Documents", type="primary", use_container_width=True):
                index_documents(valid_files, config)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        st.markdown("## 📋 Indexed Files")
        vector_store = get_vector_store()
        indexed_docs = vector_store.get_indexed_documents()

        if indexed_docs:
            for doc_name in indexed_docs:
                ext = Path(doc_name).suffix.upper()
                st.markdown(
                    f"""<div class="file-item">
                        <span class="file-name">📄 {doc_name}</span>
                        <span class="file-ext">{ext}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No documents indexed yet.")

        if st.button("🗑️ Clear Index", use_container_width=True):
            clear_index(config)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        with st.expander("⚙️ Advanced Settings", expanded=False):
            config.llm_provider = st.selectbox("LLM Provider", ["gemini", "openai", "ollama"], 
                index=["gemini", "openai", "ollama"].index(config.llm_provider.lower()))
            
            if config.llm_provider == "gemini":
                config.gemini_api_key = st.text_input("Gemini API Key", type="password", value=config.gemini_api_key)
            elif config.llm_provider == "openai":
                config.openai_api_key = st.text_input("OpenAI API Key", type="password", value=config.openai_api_key)
            elif config.llm_provider == "ollama":
                config.ollama_base_url = st.text_input("Ollama Base URL", value=config.ollama_base_url)

            config.top_k = st.slider("Top-K Results", 1, 20, config.top_k)
            config.confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, config.confidence_threshold, 0.05)
            config.enable_hybrid_search = st.toggle("Hybrid Search (BM25 + Vector)", config.enable_hybrid_search)
            config.enable_reranking = st.toggle("Cross-Encoder Re-ranking", config.enable_reranking)
            st.session_state["enable_query_rewrite"] = st.toggle("Query Rewriting", False)


# ── Indexing Logic ───────────────────────────────────────────────────────────

def index_documents(uploaded_files: list, config: RAGConfig) -> None:
    embedding_model = load_embedding_model(config.embedding_model)
    vector_store = get_vector_store()
    
    chunker = ParentChildChunker(
        parent_chunk_size=config.parent_chunk_size,
        parent_chunk_overlap=config.parent_chunk_overlap,
        child_chunk_size=config.child_chunk_size,
        child_chunk_overlap=config.child_chunk_overlap
    )

    all_parents = []
    all_children = []
    
    progress_bar = st.sidebar.progress(0, "Processing documents...")

    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((i) / len(uploaded_files), f"Processing: {uploaded_file.name}...")

        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            documents = load_document(tmp_path)
            for doc in documents:
                doc.metadata["document"] = uploaded_file.name

            parents, children = chunker.chunk_documents(documents)
            all_parents.extend(parents)
            all_children.extend(children)
        finally:
            os.unlink(tmp_path)

    if not all_children:
        st.sidebar.error("No text content could be extracted.")
        progress_bar.empty()
        return

    progress_bar.progress(0.8, "Generating embeddings (Child chunks)...")
    contents = [c.content for c in all_children]
    embeddings = embedding_model.embed_texts(contents, show_progress=False)

    progress_bar.progress(0.9, "Indexing vectors...")
    metadata = [c.metadata for c in all_children]
    vector_store.add_documents(embeddings, contents, metadata, parents=all_parents)

    vector_store.save_index(config.vector_db_path)

    st.session_state.child_chunks = all_children
    st.session_state.indexed_files = vector_store.get_indexed_documents()
    st.session_state.vector_store_initialized = True

    progress_bar.progress(1.0, "✅ Indexing complete!")
    st.sidebar.success(f"Indexed {len(all_children)} chunks from {len(uploaded_files)} document(s).")


def clear_index(config: RAGConfig) -> None:
    vector_store = get_vector_store()
    vector_store.clear()
    vector_store.delete_saved_index(config.vector_db_path)

    st.session_state.child_chunks = []
    st.session_state.indexed_files = []
    st.session_state.messages = []
    st.session_state.vector_store_initialized = False

    if "vector_store" in st.session_state:
        del st.session_state.vector_store

    st.rerun()


# ── Chat Interface ───────────────────────────────────────────────────────────

def render_chat() -> None:
    st.markdown(
        """<div class="main-header">
            <h1>🔍 RAG Document Q&A</h1>
            <p>Advanced Parent-Child retrieval with multi-provider LLM support.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    config = st.session_state.config

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "response" in message:
                render_metadata(message["response"])

    if not st.session_state.messages and not st.session_state.vector_store_initialized:
        st.info("👈 Upload and index documents from the sidebar to start asking questions.")
        return

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                response = generate_answer(prompt, config)

            st.markdown(response.answer)
            render_metadata(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "response": response,
            "original_query": prompt,
        })


def render_metadata(response: RAGResponse):
    if response.confidence > 0:
        conf_class = "conf-high" if response.confidence > 0.7 else "conf-medium" if response.confidence > 0.4 else "conf-low"
        st.markdown(f"""<span class="confidence-badge {conf_class}">Confidence: {response.confidence:.2f}</span>""", unsafe_allow_html=True)

    if response.query_used:
        st.caption(f"🔄 Used query: *{response.query_used}*")
        
    if response.tokens.get("total_tokens", 0) > 0:
        st.caption(f"🪙 Tokens: {response.tokens['total_tokens']} (${response.cost:.4f})")

    if response.sources:
        st.markdown("**📚 Sources:**")
        for source in response.sources:
            score_class = "score-high" if source['score'] > 0.7 else "score-medium" if source['score'] > 0.4 else "score-low"
            st.markdown(
                f"""<div class="source-card">
                    <span class="doc-name">📄 {source['document']}</span>
                    <span class="page-info"> — Page {source['page']}</span>
                    <span class="chunk-score {score_class}" style="margin-left: 0.5rem;">{source['score']:.3f}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    if response.retrieved_chunks:
        with st.expander(f"🔎 Context Passed to LLM ({len(response.retrieved_chunks)} blocks)"):
            for i, chunk in enumerate(response.retrieved_chunks, 1):
                ctype = chunk.metadata.get("type", "chunk")
                st.markdown(
                    f"""<div class="chunk-card">
                        <strong>Block {i} [{ctype}]</strong> — <span style="color: #a78bfa;">{chunk.metadata.get('document')}</span> (Score: {chunk.score:.3f})
                        <div style="color: #cbd5e1; margin-top:0.5rem;">{truncate_text(chunk.content, 400)}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )


def generate_answer(query: str, config: RAGConfig) -> RAGResponse:
    try:
        embedding_model = load_embedding_model(config.embedding_model)
        vector_store = get_vector_store()
        retriever = Retriever(vector_store, embedding_model, config.confidence_threshold)
        
        llm = LLMFactory.create(config)
        generator = RAGGenerator(llm, retriever, config)

        query_id = str(uuid.uuid4())
        
        return generator.answer(
            query=query,
            query_id=query_id,
            top_k=config.top_k,
            enable_hybrid=config.enable_hybrid_search,
            enable_reranking=config.enable_reranking,
            enable_query_rewrite=st.session_state.get("enable_query_rewrite", False),
            chunks=st.session_state.get("child_chunks", []),
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        return RAGResponse(answer=f"❌ Error: {e}")


# ── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    render_sidebar()
    if st.session_state.page == "Chat":
        render_chat()
    else:
        render_analytics(st.session_state.config)

if __name__ == "__main__":
    main()
