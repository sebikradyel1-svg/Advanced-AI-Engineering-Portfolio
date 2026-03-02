# ADR-003: LLM Provider Selection — Groq API with Llama 3.3 70B

**Status:** Accepted  
**Date:** 2025-12-20  
**Decision Maker:** Paul Sebastian Kradyel  
**Supersedes:** Local FLAN-T5 Small inference

## Context

The RAG system initially used FLAN-T5 Small (80M parameters) running locally. While this fit within the 512MB RAM constraint, response quality was poor — answers were often incomplete, generic, or failed to synthesize information from multiple retrieved chunks.

## Decision Drivers

- Response quality must be sufficient for realistic HR Q&A
- Must be free or near-zero cost (portfolio project)
- Latency target: < 3 seconds per response
- Must not exceed 512MB RAM on Render free tier

## Options Considered

### Option A: Groq API — Llama 3.3 70B Versatile (Chosen)
- Free tier: 30 req/min, 100K tokens/day
- Latency: 200-800ms (fastest inference API available)
- Quality: 70B parameter model — strong reasoning and instruction following
- RAM: ~0MB (cloud inference)

### Option B: Local Llama 3.2 3B via llama.cpp
- Free, no API dependency
- RAM: ~2-3GB (quantized Q4_K_M)
- Exceeds Render free tier RAM budget

### Option C: OpenAI GPT-3.5 Turbo API
- High quality, well-documented
- Cost: $0.50/1M input tokens
- Latency: 500-2000ms

### Option D: Local FLAN-T5 Small (Previous)
- 80M parameters, fits in 512MB
- Poor quality for complex HR queries
- No API dependency

## Decision

**Chose Option A: Groq API with Llama 3.3 70B**

## Rationale

1. **Quality leap:** 70B parameters vs 80M — qualitative improvement in answer coherence, completeness, and ability to synthesize multiple context chunks.

2. **Free tier is sufficient:** 100K tokens/day supports ~200-400 queries, well beyond portfolio demo needs. 30 req/min is fine for single-user demo.

3. **Fastest inference available:** Groq's LPU hardware delivers 200-800ms latency — faster than OpenAI, faster than any local model on consumer hardware.

4. **Zero RAM impact:** Cloud inference means the 512MB Render budget is fully available for embeddings, FAISS, and FastAPI.

5. **No vendor lock-in:** Using LangChain's ChatGroq wrapper — swapping to OpenAI, Anthropic, or local models requires changing one class instantiation.

## Consequences

### Positive
- Dramatic quality improvement (faithfulness = 1.0 on Ragas evaluation)
- Sub-1-second query processing in production
- Zero memory overhead for LLM

### Negative
- Internet dependency for generation (embeddings and retrieval are still local)
- Rate limits constrain automated testing (Ragas evaluation hits 30 req/min limit)
- Daily token limit (100K) constrains bulk evaluation

### Mitigations
- Graceful fallback message if API is unavailable
- Rate limiting with configurable delay in evaluation scripts
- Test questions cached to JSON for re-evaluation without regeneration

## Cost Analysis

| Scenario | Daily Tokens | Monthly Cost |
|----------|-------------|-------------|
| Demo usage (20 queries/day) | ~10K | $0 (free tier) |
| Heavy testing (100 queries/day) | ~50K | $0 (free tier) |
| Ragas eval (50 questions) | ~80K | $0 (free tier) |
| Production (1000 queries/day) | ~500K | Requires paid tier |
