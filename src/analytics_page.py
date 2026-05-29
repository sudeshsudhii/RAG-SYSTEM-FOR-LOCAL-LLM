"""
Analytics Dashboard module.

Reads JSON logs and displays key metrics and Plotly charts.
"""

import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

def load_metrics(log_dir: str) -> pd.DataFrame:
    """Load JSONL metrics into a pandas DataFrame."""
    log_file = Path(log_dir) / "rag_metrics.jsonl"
    if not log_file.exists():
        return pd.DataFrame()
        
    data = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except:
                pass
                
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def render_analytics(config) -> None:
    """Render the Analytics Streamlit page."""
    st.markdown("## 📊 RAG Analytics Dashboard")
    
    df = load_metrics(config.monitoring_path)
    if df.empty:
        st.info("No analytics data available yet. Ask some questions in the Chat tab first.")
        return

    # Split data by event type
    queries_df = df[df['event_type'] == 'query'].copy()
    retrieval_df = df[df['event_type'] == 'retrieval'].copy()
    gen_df = df[df['event_type'] == 'generation'].copy()
    error_df = df[df['event_type'] == 'error'].copy()

    # Top Level Metrics
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Queries", len(queries_df))
    with cols[1]:
        avg_lat = gen_df['generation_latency_ms'].mean() if not gen_df.empty else 0
        st.metric("Avg Gen Latency", f"{avg_lat/1000:.2f}s")
    with cols[2]:
        avg_score = retrieval_df['avg_retrieval_score'].mean() if not retrieval_df.empty else 0
        st.metric("Avg Retrieval Score", f"{avg_score:.2f}")
    with cols[3]:
        total_cost = gen_df['estimated_cost_usd'].sum() if not gen_df.empty else 0
        st.metric("Total Est. Cost", f"${total_cost:.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Query Volume")
        if not queries_df.empty:
            queries_df['hour'] = queries_df['timestamp'].dt.floor('H')
            vol = queries_df.groupby('hour').size().reset_index(name='count')
            fig = px.line(vol, x='hour', y='count', markers=True, title="Queries over Time")
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.markdown("### Generation Latency")
        if not gen_df.empty:
            fig2 = px.histogram(gen_df, x='generation_latency_ms', nbins=20, title="Latency Distribution (ms)")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Retrieval Scores")
        if not retrieval_df.empty:
            fig3 = px.box(retrieval_df, y='avg_retrieval_score', points="all", title="Retrieval Score Spread")
            st.plotly_chart(fig3, use_container_width=True)
            
    with col4:
        st.markdown("### Query Categories")
        if not queries_df.empty and 'category' in queries_df.columns:
            cat_counts = queries_df['category'].value_counts().reset_index()
            cat_counts.columns = ['category', 'count']
            fig4 = px.pie(cat_counts, values='count', names='category', title="Query Intents")
            st.plotly_chart(fig4, use_container_width=True)
            
    if not error_df.empty:
        st.markdown("### ⚠️ Errors")
        st.dataframe(error_df[['timestamp', 'component', 'error_message']])
