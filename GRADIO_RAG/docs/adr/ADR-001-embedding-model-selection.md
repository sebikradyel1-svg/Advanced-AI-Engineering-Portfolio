# ADR-001: Embedding Model Selection

**Status:** Accepted  
**Date:** 2025-12-15  
**Decision Maker:** Paul Sebastian Kradyel

## Context

The HR RAG System needs an embedding model to convert document chunks and queries into dense vectors for semantic similarity search. The model choice affects retrieval quality, latency, memory usage, and deployment cost.

## Decision Drivers

- Must run on consumer GPU (RTX 3060, 12GB VRAM) and CPU-only environments (Render free tier, 512MB RAM)
- Retrieval quality must be sufficient for HR policy Q&A (domain-specific but English-only)
- Latency target: < 100ms per embedding call
- Must be compatible with FAISS index and LangChain ecosystem

## Options Considered

### Option A: sentence-transformers/all-MiniLM-L6-v2 (Chosen)
- **Dimensions:** 384
- **Parameters:** 22M
- **MTEB score:** 63.05
- **Memory:** ~90MB
- **Latency:** ~5ms per query on CPU

### Option B: sentence-transformers/all-mpnet-base-v2
- **Dimensions:** 768
- **Parameters:** 109M
- **Memory:** ~420MB
- **MTEB score:** 65.07
- **Latency:** ~15ms per query on CPU

### Option C: BAAI/bge-large-en-v1.5
- **Dimensions:** 1024
- **Parameters:** 335M
- **Memory:** ~1.3GB
- **MTEB score:** 64.23
- **Latency:** ~30ms per query on CPU

### Option D: OpenAI text-embedding-3-small (API)
- **Dimensions:** 1536
- **Cost:** $0.02/1M tokens
- **Latency:** 200-500ms (network dependent)
- **Dependency:** Requires API key and internet

## Decision

**Chose Option A: all-MiniLM-L6-v2**

## Rationale

1. **Memory constraint is binding:** Render free tier has 512MB total. At ~90MB, MiniLM leaves headroom for FAISS index, LLM inference, and FastAPI overhead. Options B and C would consume 80-250% of available RAM.

2. **Quality is sufficient:** The 2-point MTEB gap between MiniLM (63.05) and mpnet (65.07) is marginal for our use case. HR policy documents use consistent terminology, reducing the need for nuanced semantic understanding.

3. **Latency advantage:** 5ms vs 15-30ms matters when embedding multiple chunks during index building and when users expect sub-second responses.

4. **No external dependency:** Unlike OpenAI embeddings, local models work offline and have zero marginal cost — critical for a portfolio project that runs on free tier.

## Consequences

### Positive
- Fits within 512MB RAM budget with room to spare
- Zero API cost for embeddings
- Works offline (no internet dependency for embeddings)
- Fast index building (~5 seconds for 64 chunks)

### Negative
- Lower semantic quality than larger models (may miss some subtle paraphrases)
- 384-dimensional vectors have less expressive capacity than 768 or 1024
- Mitigated by: hybrid search (BM25 catches what dense retrieval misses)

### Risks
- If document corpus grows significantly (10K+ chunks), retrieval quality may degrade — would need to re-evaluate with larger model
- Monitoring via Ragas context_precision metric will detect quality degradation early

## Metrics to Monitor

| Metric | Current | Threshold | Action if Breached |
|--------|---------|-----------|-------------------|
| context_precision (Ragas) | 0.50 | < 0.30 | Upgrade to mpnet-base-v2 |
| Embedding latency (p99) | 5ms | > 50ms | Profile and optimize |
| Memory usage | ~90MB | > 200MB | Investigate memory leak |
