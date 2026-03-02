# Architecture Decision Records — HR RAG System

This directory contains the Architecture Decision Records (ADRs) for the HR RAG Knowledge Assistant.

ADRs document the key technical decisions made during development, including the context, options considered, rationale, and consequences. They serve as a living record of *why* the system is built the way it is.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](ADR-001-embedding-model-selection.md) | Embedding Model Selection | Accepted | 2025-12-15 |
| [ADR-002](ADR-002-hybrid-retrieval-strategy.md) | Hybrid Retrieval Strategy (BM25 + FAISS) | Accepted | 2026-03-01 |
| [ADR-003](ADR-003-llm-provider-selection.md) | LLM Provider Selection (Groq API) | Accepted | 2025-12-20 |

## ADR Format

Each ADR follows this structure:

- **Context:** Why is this decision needed?
- **Decision Drivers:** What constraints and requirements shaped the decision?
- **Options Considered:** What alternatives were evaluated?
- **Decision:** What was chosen?
- **Rationale:** Why was this option selected over alternatives?
- **Consequences:** What are the positive and negative outcomes?
- **Metrics:** How do we measure success?
