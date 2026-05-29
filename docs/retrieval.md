# Advanced Retrieval Strategy

## Parent-Child Chunking
To balance the trade-off between search precision and LLM context window quality, this system uses Parent-Child Chunking:
1. **Parent Chunks**: Large sections of text (~2000 chars) that capture full context.
2. **Child Chunks**: Small sentences (~400 chars) that are tightly focused.
*How it works*: Only Child Chunks are converted to embeddings and searched. When a match is found, the system traces the Child back to its Parent and feeds the *entire Parent Chunk* to the LLM.

## Hybrid Search (RRF)
We combine traditional keyword search (BM25) with Dense Vector Search. Because the scores are on different scales, we use **Reciprocal Rank Fusion (RRF)** to combine the rankings seamlessly.

## Cross-Encoder Re-ranking
After retrieving the top-K chunks, an optional MS-MARCO Cross-Encoder evaluates the exact pair `(Query, Document)` to provide a highly accurate relevance score, overriding the initial cosine-similarity ranking.
