# ADR-002: Hybrid Retrieval Strategy (BM25 + FAISS with RRF)

**Status:** Accepted  
**Date:** 2026-03-01  
**Decision Maker:** Paul Sebastian Kradyel  
**Supersedes:** Dense-only FAISS retrieval

## Context

Ragas evaluation of the FAISS-only retrieval pipeline showed a context_precision score of 0.50 — meaning half of retrieved chunks were irrelevant. Analysis revealed two failure modes:

1. **Keyword queries** (e.g., "401k match policy") — FAISS retrieved semantically similar but wrong sections because "401k" doesn't have strong semantic neighbors
2. **Exact term queries** (e.g., "maternity leave eligibility") — FAISS prioritized paraphrases over exact matches

## Decision Drivers

- Must improve context_precision from 0.50 baseline
- Must maintain sub-1-second query latency
- Must work within existing 512MB RAM constraint
- Must not require re-building the FAISS index
- Implementation should be maintainable and testable

## Options Considered

### Option A: Hybrid BM25 + FAISS with Reciprocal Rank Fusion (Chosen)
- Combine sparse (BM25) and dense (FAISS) retrieval
- Merge results using RRF: score(doc) = Σ weight_i / (k + rank_i)
- Weights tunable (default: BM25=0.4, FAISS=0.6)

### Option B: Re-rank with Cross-Encoder
- Use FAISS top-20, then re-rank with a cross-encoder model
- Higher quality but adds 200-500ms latency and ~400MB memory

### Option C: Increase top_k from 3 to 5
- Simple change, more context for LLM
- Risk: more irrelevant chunks dilute context quality

### Option D: Fine-tune Embeddings on HR Domain
- Train adapter on HR-specific question-passage pairs
- Highest potential quality but requires training data and compute

### Option E: Use LangChain EnsembleRetriever
- Built-in hybrid retrieval
- Removed in LangChain 1.2+ — not viable

## Decision

**Chose Option A: Manual RRF implementation with BM25 + FAISS**

## Rationale

1. **Data-driven:** Ragas baseline identified retrieval as the bottleneck (context_precision=0.50), not generation (faithfulness=1.0). Hybrid search directly targets retrieval quality.

2. **Complementary strengths confirmed by testing:**

   | Query | BM25 #1 | FAISS #1 | Winner |
   |-------|---------|----------|--------|
   | "401k match policy" | ❌ Vision Insurance | ✅ 401K plan | FAISS |
   | "maternity leave" | ✅ Secondary Caregiver | ❌ Other Benefits | BM25 |
   | "report harassment" | ✅ Exact match | ✅ Related section | Both |

3. **RRF over score fusion:** BM25 scores are unbounded while FAISS cosine similarity is 0-1. RRF uses ranks instead of scores, eliminating the normalization problem.

4. **Manual RRF over EnsembleRetriever:** LangChain removed EnsembleRetriever in v1.2. Our 20-line manual implementation has zero external dependencies and gives full control over the fusion constant (k=60).

5. **Resource efficient:** BM25 index adds ~2MB memory and ~50ms latency — negligible compared to LLM generation (~700ms).

## Implementation

```
hybrid_retriever.py
├── reciprocal_rank_fusion()    # Core RRF algorithm
├── HybridRetriever(BaseRetriever)  # LangChain-compatible interface  
├── load_and_chunk_documents()  # Shared chunking for BM25
└── create_hybrid_retriever()   # Factory function
```

Integration in app.py: single line change from `vector_db.similarity_search()` to `hybrid_retriever.invoke()`.

## Consequences

### Positive
- Directly addresses the identified retrieval bottleneck
- BM25 catches keyword-heavy queries that FAISS misses
- Drop-in replacement — same `.invoke()` interface
- Weights are tunable for domain-specific optimization

### Negative
- BM25 requires loading raw documents at startup (adds ~1 second to init)
- Documents must be chunked twice (FAISS index + BM25 index) — minor duplication
- BM25 index is not persistent (rebuilt on each startup, unlike FAISS)

### Future Optimization
- Tune BM25/FAISS weights using Ragas A/B testing
- Add BM25 index persistence if startup time becomes an issue
- Consider cross-encoder re-ranking as a later enhancement (Option B)

## Metrics

| Metric | Before (FAISS-only) | After (Hybrid) | Target |
|--------|-------------------|----------------|--------|
| context_precision | 0.50 | TBD | > 0.65 |
| faithfulness | 1.00 | 1.00 | > 0.90 |
| answer_relevancy | 0.40 | TBD | > 0.55 |
| Query latency (p50) | 0.7s | 0.8s | < 2.0s |
