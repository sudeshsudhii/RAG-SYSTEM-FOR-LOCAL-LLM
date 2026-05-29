# Evaluation Framework

The RAG system includes an automated evaluation suite to measure both Retrieval performance and LLM Generation quality.

## Setup
Ensure you have indexed the `data/sample.txt` document into the vector store before running evaluations, as the `benchmark_dataset.json` queries it.

## 1. Retrieval Evaluation
Measures how well the system finds the correct context documents.

**Metrics Calculated:**
- **Precision@K**: Proportion of retrieved documents that are relevant.
- **Recall@K**: Proportion of relevant documents that were retrieved.
- **MRR (Mean Reciprocal Rank)**: Position of the first relevant document.
- **Hit Rate**: % of queries where at least one relevant document was found.

**Run:**
```bash
python evaluation/retrieval_eval.py
```

## 2. Answer Evaluation (LLM-as-a-judge)
Uses the LLM to score the generated answers against ground-truth expected answers.

**Metrics Calculated:**
- **Groundedness**: Does the answer stick strictly to the facts? (0 to 1)
- **Completeness**: Does the answer address all parts of the question? (0 to 1)

**Run:**
```bash
python evaluation/answer_eval.py
```
