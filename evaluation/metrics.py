"""
Metrics calculation for RAG evaluation.
"""

def precision_at_k(relevant_docs: list[str], retrieved_docs: list[str], k: int) -> float:
    retrieved_k = retrieved_docs[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for doc in retrieved_k if doc in relevant_docs)
    return hits / k

def recall_at_k(relevant_docs: list[str], retrieved_docs: list[str], k: int) -> float:
    retrieved_k = retrieved_docs[:k]
    if not relevant_docs:
        return 0.0
    hits = sum(1 for doc in retrieved_k if doc in relevant_docs)
    return hits / len(relevant_docs)

def mean_reciprocal_rank(relevant_docs: list[str], retrieved_docs: list[str]) -> float:
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1.0 / rank
    return 0.0

def hit_rate(relevant_docs: list[str], retrieved_docs: list[str]) -> float:
    return 1.0 if any(doc in relevant_docs for doc in retrieved_docs) else 0.0
