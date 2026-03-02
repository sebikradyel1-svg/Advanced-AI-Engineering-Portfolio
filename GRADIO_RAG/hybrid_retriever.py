"""
Hybrid Retriever: BM25 (sparse) + FAISS (dense) with manual RRF
================================================================
No dependency on EnsembleRetriever (removed in LangChain 1.2+).
Uses custom Reciprocal Rank Fusion implementation.

RRF formula: score(doc) = sum( weight_i / (k + rank_i) )
k=60 (standard constant from Cormack et al. 2009)
"""

import logging
import os
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
RRF_K = 60
BM25_WEIGHT = 0.4
FAISS_WEIGHT = 0.6


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float],
    k: int = RRF_K,
    top_n: int = TOP_K,
) -> list[Document]:
    """Merge ranked lists using RRF."""
    doc_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for retriever_idx, doc_list in enumerate(ranked_lists):
        weight = weights[retriever_idx]
        for rank, doc in enumerate(doc_list, start=1):
            key = doc.page_content
            rrf_score = weight / (k + rank)
            doc_scores[key] = doc_scores.get(key, 0.0) + rrf_score
            doc_map[key] = doc

    sorted_keys = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


class HybridRetriever(BaseRetriever):
    """BM25 + FAISS retriever with RRF fusion."""

    bm25_retriever: Any = None
    faiss_retriever: Any = None
    bm25_weight: float = BM25_WEIGHT
    faiss_weight: float = FAISS_WEIGHT
    top_k: int = TOP_K

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        bm25_docs = self.bm25_retriever.invoke(query)
        faiss_docs = self.faiss_retriever.invoke(query)
        return reciprocal_rank_fusion(
            ranked_lists=[bm25_docs, faiss_docs],
            weights=[self.bm25_weight, self.faiss_weight],
            top_n=self.top_k,
        )


def load_and_chunk_documents(docs_path: str) -> list[Document]:
    """Load and chunk source documents."""
    with open(docs_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    texts = splitter.split_text(raw_text)
    documents = [Document(page_content=t) for t in texts]
    logger.info("Loaded %d chunks from %s", len(documents), docs_path)
    return documents


def create_hybrid_retriever(
    docs_path: str = "company_policies.txt",
    faiss_path: str = "faiss_index",
    top_k: int = TOP_K,
    bm25_weight: float = BM25_WEIGHT,
    faiss_weight: float = FAISS_WEIGHT,
) -> HybridRetriever:
    """Create hybrid BM25 + FAISS retriever with RRF."""
    documents = load_and_chunk_documents(docs_path)
    bm25_retriever = BM25Retriever.from_documents(documents, k=top_k)
    logger.info("BM25 retriever created with k=%d", top_k)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        faiss_path, embeddings, allow_dangerous_deserialization=True,
    )
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    logger.info("FAISS retriever loaded from %s with k=%d", faiss_path, top_k)

    hybrid = HybridRetriever(
        bm25_retriever=bm25_retriever,
        faiss_retriever=faiss_retriever,
        bm25_weight=bm25_weight,
        faiss_weight=faiss_weight,
        top_k=top_k,
    )
    logger.info("Hybrid retriever ready: BM25(%.1f) + FAISS(%.1f)", bm25_weight, faiss_weight)
    return hybrid


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("  HYBRID RETRIEVER — COMPARISON TEST")
    print("=" * 60)

    docs_path = "company_policies.txt"
    faiss_path = "faiss_index"

    if not os.path.exists(docs_path) or not os.path.exists(faiss_path):
        print("ERROR: Ensure company_policies.txt and faiss_index/ exist.")
        exit(1)

    documents = load_and_chunk_documents(docs_path)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    bm25_ret = BM25Retriever.from_documents(documents, k=TOP_K)
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    faiss_ret = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    hybrid_ret = create_hybrid_retriever(docs_path, faiss_path)

    test_queries = [
        "What is the 401k match policy?",
        "How much vacation time do employees get?",
        "What happens if I'm late to work?",
        "maternity leave eligibility",
        "How do I report harassment?",
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"  Query: {query}")
        print(f"{'─' * 60}")

        bm25_docs = bm25_ret.invoke(query)
        faiss_docs = faiss_ret.invoke(query)
        hybrid_docs = hybrid_ret.invoke(query)

        print(f"\n  BM25 ({len(bm25_docs)} docs):")
        for i, d in enumerate(bm25_docs):
            print(f"    [{i+1}] {d.page_content[:80]}...")

        print(f"\n  FAISS ({len(faiss_docs)} docs):")
        for i, d in enumerate(faiss_docs):
            print(f"    [{i+1}] {d.page_content[:80]}...")

        print(f"\n  HYBRID ({len(hybrid_docs)} docs):")
        for i, d in enumerate(hybrid_docs):
            print(f"    [{i+1}] {d.page_content[:80]}...")

    print(f"\n{'=' * 60}")
    print("  Done. Hybrid combines best of both retrievers.")
    print(f"{'=' * 60}")
