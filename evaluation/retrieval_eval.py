"""
Script to evaluate retrieval performance against the benchmark dataset.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.embeddings import EmbeddingModel
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from evaluation.metrics import precision_at_k, recall_at_k, mean_reciprocal_rank, hit_rate

def run_evaluation():
    print("Loading vector store...")
    vector_store = FAISSVectorStore(dimension=config.embedding_dimension)
    if not vector_store.load_index(config.vector_db_path):
        print("❌ Could not load vector index. Have you indexed documents?")
        return

    print("Loading embedding model...")
    embedding_model = EmbeddingModel(config.embedding_model)
    retriever = Retriever(vector_store, embedding_model, confidence_threshold=0.0)

    dataset_path = Path("evaluation/benchmark_dataset.json")
    if not dataset_path.exists():
        print("❌ Benchmark dataset not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    k_value = 5

    print(f"Running evaluation on {len(dataset)} queries...")
    for item in dataset:
        query = item["query"]
        expected_docs = item["relevant_docs"]
        
        retrieved = retriever.retrieve(query, top_k=k_value)
        retrieved_docs = [r.metadata.get("document", "") for r in retrieved]
        
        p_k = precision_at_k(expected_docs, retrieved_docs, k_value)
        r_k = recall_at_k(expected_docs, retrieved_docs, k_value)
        mrr = mean_reciprocal_rank(expected_docs, retrieved_docs)
        hr = hit_rate(expected_docs, retrieved_docs)
        
        results.append({
            "query": query,
            "precision@k": p_k,
            "recall@k": r_k,
            "mrr": mrr,
            "hit_rate": hr
        })

    # Aggregate
    avg_p = sum(r["precision@k"] for r in results) / len(results)
    avg_r = sum(r["recall@k"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_hr = sum(r["hit_rate"] for r in results) / len(results)

    final_metrics = {
        f"precision@{k_value}": avg_p,
        f"recall@{k_value}": avg_r,
        "mrr": avg_mrr,
        "hit_rate": avg_hr
    }

    print("\n✅ Evaluation Complete!")
    print(json.dumps(final_metrics, indent=2))

    with open("evaluation/eval_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

if __name__ == "__main__":
    run_evaluation()
