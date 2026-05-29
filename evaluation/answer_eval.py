"""
Script to evaluate answer quality (Groundedness and Completeness) using LLM-as-a-judge.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from src.generator import LLMFactory

EVAL_PROMPT = """You are an impartial evaluator for a RAG system.
Evaluate the given generated answer based on the provided expected answer.

Score the following two metrics from 0.0 to 1.0:
1. Groundedness (Faithfulness): Does the answer stick to the facts in the expected answer without hallucinating?
2. Completeness: Does the generated answer fully address the core concepts in the expected answer?

Expected Answer: {expected}
Generated Answer: {generated}

Output ONLY a JSON dict in this exact format:
{{"groundedness": 0.0, "completeness": 0.0}}
"""

def run_evaluation():
    dataset_path = Path("evaluation/benchmark_dataset.json")
    if not dataset_path.exists():
        print("❌ Benchmark dataset not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Initialize LLM for judging (assumes Gemini is configured)
    llm = LLMFactory.create(config)
    
    # We will simulate the "Generated Answer" by generating it now 
    # (in a real system, you'd run the full RAG pipeline per query, but for simplicity we'll just evaluate)
    from src.embeddings import EmbeddingModel
    from src.vector_store import FAISSVectorStore
    from src.retriever import Retriever
    from src.generator import RAGGenerator

    vector_store = FAISSVectorStore(dimension=config.embedding_dimension)
    vector_store.load_index(config.vector_db_path)
    embedding_model = EmbeddingModel(config.embedding_model)
    retriever = Retriever(vector_store, embedding_model, confidence_threshold=0.0)
    generator = RAGGenerator(llm, retriever, config)

    results = []
    print(f"Evaluating answers for {len(dataset)} queries...")
    
    for item in dataset:
        query = item["query"]
        expected = item["expected_answer"]
        
        response = generator.answer(query, top_k=3)
        generated = response.answer
        
        try:
            eval_response = llm.generate(EVAL_PROMPT.format(expected=expected, generated=generated))
            # Clean Markdown formatting if LLM added it
            eval_response = eval_response.replace("```json", "").replace("```", "").strip()
            scores = json.loads(eval_response)
        except Exception as e:
            print(f"Eval parsing failed: {e}")
            scores = {"groundedness": 0.0, "completeness": 0.0}
            
        results.append({
            "query": query,
            "groundedness": scores.get("groundedness", 0.0),
            "completeness": scores.get("completeness", 0.0)
        })

    avg_g = sum(r["groundedness"] for r in results) / len(results)
    avg_c = sum(r["completeness"] for r in results) / len(results)

    final_metrics = {
        "groundedness": avg_g,
        "completeness": avg_c
    }

    print("\n✅ Answer Evaluation Complete!")
    print(json.dumps(final_metrics, indent=2))
    
    with open("evaluation/answer_eval_results.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

if __name__ == "__main__":
    run_evaluation()
