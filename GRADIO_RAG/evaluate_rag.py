"""
HR RAG System - Evaluation Pipeline with Ragas
================================================
Evaluates RAG quality using three core metrics:
- Context Relevance (context_precision): Are retrieved chunks relevant?
- Answer Faithfulness: Is the answer grounded in the context?
- Answer Relevance: Does the answer address the question?

Usage:
    python evaluate_rag.py                          # Full eval (50 questions)
    python evaluate_rag.py --num-questions 10       # Quick eval
    python evaluate_rag.py --questions-file qa.json  # Use pre-generated questions
    python evaluate_rag.py --export-csv              # Also export CSV

Trade-off decisions:
- Synthetic Q&A generation via LLM vs manual curation:
  Chose LLM generation for scalability; manual ground trfuths would score higher
  on context_recall but don't scale for CI/CD.
- Ragas vs custom metrics: Ragas provides standardized, paper-backed metrics
  (ES, ARES) that interviewers recognize. Custom metrics could be more specific
  to HR domain but lack credibility.
- 50-100 questions vs fewer: 50 gives statistically meaningful averages while
  keeping Groq API costs near zero (free tier).
"""

import json
import os
import sys
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — fail fast with helpful messages
# ---------------------------------------------------------------------------
def _check_dependencies():
    """Verify all required packages are installed."""
    missing = []
    for pkg, install in [
        ("ragas", "ragas"),
        ("datasets", "datasets"),
        ("langchain_community", "langchain-community"),
        ("langchain_huggingface", "langchain-huggingface"),
        ("groq", "groq"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(install)
    if missing:
        logger.error(
            "Missing dependencies. Install with:\n"
            f"  pip install {' '.join(missing)}"
        )
        sys.exit(1)

_check_dependencies()

from datasets import Dataset
from ragas import evaluate

# Ragas v1.0+ moved metrics to ragas.metrics.collections
try:
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
except ImportError:
    from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# langchain_text_splitters is a separate package since langchain 0.2+
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
DOCUMENTS_PATH = "company_policies.txt"
TOP_K = 3
DEFAULT_NUM_QUESTIONS = 50
REPORT_DIR = "evaluation_reports"

# Rate-limit guard for Groq free tier (30 req/min)
GROQ_DELAY_SECONDS = 4


# ============================================================================
# 1. LOAD RAG COMPONENTS
# ============================================================================
class RAGEvaluator:
    """Orchestrates test-set generation, RAG querying, and Ragas evaluation."""

    def __init__(self, faiss_path: str = FAISS_INDEX_PATH, docs_path: str = DOCUMENTS_PATH, use_hybrid: bool = False):
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        logger.info("Loading FAISS index from %s...", faiss_path)
        self.vectorstore = FAISS.load_local(
            faiss_path,
            self.embeddings,
            allow_dangerous_deserialization=True,  # required for pickle-based index
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        if use_hybrid:
            logger.info("Using HYBRID retriever (BM25 + FAISS)")
            from hybrid_retriever import create_hybrid_retriever
            self.retriever = create_hybrid_retriever(docs_path=docs_path, faiss_path=faiss_path, top_k=TOP_K)
        self.groq = Groq(api_key=GROQ_API_KEY)
        self.docs_path = docs_path

        # Load raw text for question generation
        with open(docs_path, "r", encoding="utf-8") as f:
            self.raw_text = f.read()

        logger.info("RAG evaluator initialised ✓")

    # ------------------------------------------------------------------
    # 2. GENERATE TEST QUESTIONS
    # ------------------------------------------------------------------
    def generate_test_questions(self, num_questions: int = DEFAULT_NUM_QUESTIONS) -> list[dict]:
        """
        Generate synthetic QA pairs from the HR documents using Groq.

        Each pair has: question, ground_truth (expected answer derived from docs).
        We chunk the document and generate questions per chunk to ensure coverage.
        """
        logger.info("Generating %d test questions from %s...", num_questions, self.docs_path)

        # Split document into chunks for targeted question generation
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(self.raw_text)
        logger.info("Document split into %d chunks for question generation.", len(chunks))

        # Distribute questions across chunks (round-robin)
        questions_per_chunk = max(1, num_questions // len(chunks))
        remaining = num_questions - (questions_per_chunk * len(chunks))

        all_qa_pairs: list[dict] = []

        for i, chunk in enumerate(chunks):
            n = questions_per_chunk + (1 if i < remaining else 0)
            if n <= 0:
                continue
            if len(all_qa_pairs) >= num_questions:
                break

            prompt = f"""Based ONLY on the following HR document excerpt, generate exactly {n} question-answer pair(s).

Requirements:
- Questions should be realistic employee queries about HR policies
- Answers must be directly supported by the text — no assumptions
- Mix question types: factual, procedural, eligibility, edge-case
- Return ONLY valid JSON array, no markdown fences

Format:
[
  {{"question": "...", "ground_truth": "..."}}
]

--- DOCUMENT EXCERPT ---
{chunk}
--- END ---"""

            try:
                response = self.groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a QA dataset generator. Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    max_tokens=2000,
                )
                raw = response.choices[0].message.content.strip()
                # Clean potential markdown fences
                raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                pairs = json.loads(raw)
                all_qa_pairs.extend(pairs)
                logger.info("  Chunk %d/%d → %d questions generated", i + 1, len(chunks), len(pairs))
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("  Chunk %d/%d failed: %s — skipping", i + 1, len(chunks), e)

            time.sleep(GROQ_DELAY_SECONDS)  # respect rate limits

        all_qa_pairs = all_qa_pairs[:num_questions]
        logger.info("Total test questions generated: %d", len(all_qa_pairs))
        return all_qa_pairs

    # ------------------------------------------------------------------
    # 3. RUN RAG PIPELINE ON TEST SET
    # ------------------------------------------------------------------
    def run_rag_pipeline(self, qa_pairs: list[dict]) -> dict:
        """
        Query the RAG system for each question and collect:
        - question, answer, contexts (retrieved chunks), ground_truth

        Returns a dict ready for Ragas Dataset creation.
        """
        logger.info("Running RAG pipeline on %d questions...", len(qa_pairs))

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for i, pair in enumerate(qa_pairs):
            question = pair["question"]
            ground_truth = pair.get("ground_truth", "")

            # Retrieve documents
            retrieved_docs = self.retriever.invoke(question)
            context_texts = [doc.page_content for doc in retrieved_docs]

            # Generate answer via Groq (same prompt pattern as app.py)
            context_block = "\n\n".join(context_texts)
            answer_prompt = f"""Based on the following context, answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context_block}

Question: {question}

Answer:"""

            try:
                response = self.groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an HR assistant. Answer questions based only on "
                                "the provided context. Be precise and helpful."
                            ),
                        },
                        {"role": "user", "content": answer_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning("  Question %d failed generation: %s", i + 1, e)
                answer = "Error generating answer."

            questions.append(question)
            answers.append(answer)
            contexts.append(context_texts)
            ground_truths.append(ground_truth)

            if (i + 1) % 10 == 0:
                logger.info("  Processed %d/%d questions", i + 1, len(qa_pairs))

            time.sleep(GROQ_DELAY_SECONDS)

        return {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

    # ------------------------------------------------------------------
    # 4. EVALUATE WITH RAGAS
    # ------------------------------------------------------------------
    def evaluate(self, rag_results: dict) -> dict:
        """
        Run Ragas evaluation on the collected RAG results.

        Returns dict with per-question scores and aggregate metrics.
        """
        logger.info("Running Ragas evaluation on %d samples...", len(rag_results["question"]))

        dataset = Dataset.from_dict(rag_results)

        metrics = [
            context_precision,   # Are retrieved chunks relevant?
            faithfulness,        # Is the answer grounded in context?
            answer_relevancy,    # Does the answer address the question?
        ]

        # Ragas uses OpenAI by default for its LLM judge.
        # We configure it to use Groq-compatible endpoint via env vars.
        # Alternative: pass llm= and embeddings= explicitly.
        #
        # If OPENAI_API_KEY is not set, Ragas will fall back to
        # whatever LLM wrapper we provide.
        try:
            from ragas.llms import LangchainLLMWrapper
            from langchain_groq import ChatGroq

            ragas_llm = LangchainLLMWrapper(
                ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0.0)
            )
            from ragas.embeddings import LangchainEmbeddingsWrapper
            ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )
        except ImportError:
            logger.warning(
                "langchain-groq not installed. Trying default Ragas LLM "
                "(requires OPENAI_API_KEY). Install with: pip install langchain-groq"
            )
            result = evaluate(dataset=dataset, metrics=metrics)

        return result

    # ------------------------------------------------------------------
    # 5. GENERATE REPORT
    # ------------------------------------------------------------------
    @staticmethod
    def generate_report(
        ragas_result,
        qa_pairs: list[dict],
        rag_results: dict,
        output_dir: str = REPORT_DIR,
        export_csv: bool = False,
    ) -> Path:
        """
        Save evaluation results as JSON (and optionally CSV).
        Print summary to console.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- Aggregate scores ---
        # Handle both old dict-like and new Ragas result formats
        agg_scores = {}
        try:
            # New Ragas: result might be a dict or have a scores attribute
            if hasattr(ragas_result, 'scores'):
                raw_scores = ragas_result.scores
            elif hasattr(ragas_result, '__getitem__'):
                raw_scores = dict(ragas_result)
            else:
                raw_scores = {}

            for k, v in raw_scores.items():
                if isinstance(v, (int, float)):
                    agg_scores[str(k)] = round(float(v), 4)
        except Exception as e:
            logger.warning("Could not extract aggregate scores: %s", e)

        # --- Per-question detail ---
        per_question = []
        try:
            result_df = ragas_result.to_pandas()
        except AttributeError:
            # Fallback: try converting via dataset
            try:
                result_df = ragas_result.dataset.to_pandas() if hasattr(ragas_result, 'dataset') else None
            except Exception:
                result_df = None

        if result_df is not None:
            metric_cols = ["context_precision", "faithfulness", "answer_relevancy"]
            for idx, row in result_df.iterrows():
                entry = {
                    "question": str(row.get("question", "")),
                    "answer": str(row.get("answer", "")),
                    "ground_truth": str(row.get("ground_truth", "")),
                }
                # Safely extract contexts count
                ctx = row.get("contexts", [])
                entry["num_contexts"] = len(ctx) if isinstance(ctx, (list, tuple)) else 0

                for col in metric_cols:
                    val = row.get(col, None)
                    if val is not None and not (isinstance(val, float) and (val != val)):  # NaN check
                        entry[col] = round(float(val), 4)
                    else:
                        entry[col] = None
                per_question.append(entry)

            # Compute aggregate from per-question if not already available
            if not agg_scores:
                for col in metric_cols:
                    vals = [q[col] for q in per_question if q.get(col) is not None]
                    if vals:
                        agg_scores[col] = round(sum(vals) / len(vals), 4)
        else:
            logger.warning("Could not extract per-question results to DataFrame.")

        report = {
            "metadata": {
                "timestamp": timestamp,
                "model": GROQ_MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "top_k": TOP_K,
                "num_questions": len(qa_pairs),
                "documents": DOCUMENTS_PATH,
            },
            "aggregate_scores": agg_scores,
            "per_question_results": per_question,
        }

        # Save JSON
        json_path = Path(output_dir) / f"ragas_eval_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("JSON report saved → %s", json_path)

        # Save CSV (optional)
        if export_csv and result_df is not None:
            csv_path = Path(output_dir) / f"ragas_eval_{timestamp}.csv"
            result_df.to_csv(csv_path, index=False)
            logger.info("CSV report saved → %s", csv_path)
        elif export_csv:
            logger.warning("CSV export skipped — no DataFrame available.")

        # --- Console summary ---
        print("\n" + "=" * 60)
        print("  HR RAG SYSTEM — RAGAS EVALUATION REPORT")
        print("=" * 60)
        print(f"  Timestamp     : {timestamp}")
        print(f"  Model         : {GROQ_MODEL}")
        print(f"  Questions     : {len(qa_pairs)}")
        print(f"  Top-K chunks  : {TOP_K}")
        print("-" * 60)

        for metric, score in agg_scores.items():
            bar_len = int(score * 30) if score else 0
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {metric:<25} {bar} {score:.4f}")

        print("-" * 60)

        # Flag weak spots
        weak = [q for q in per_question if (q.get("faithfulness") or 0) < 0.5]
        if weak:
            print(f"\n  ⚠  {len(weak)} questions with low faithfulness (<0.5):")
            for w in weak[:5]:
                print(f"     → {w['question'][:70]}...")

        low_precision = [q for q in per_question if (q.get("context_precision") or 0) < 0.5]
        if low_precision:
            print(f"\n  ⚠  {len(low_precision)} questions with low context precision (<0.5):")
            for lp in low_precision[:5]:
                print(f"     → {lp['question'][:70]}...")

        print("\n" + "=" * 60)
        print(f"  Full report → {json_path}")
        print("=" * 60 + "\n")

        return json_path


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HR RAG System with Ragas metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_rag.py                         # Full eval, 50 questions
  python evaluate_rag.py --num-questions 10      # Quick sanity check
  python evaluate_rag.py --questions-file qa.json # Reuse existing test set
  python evaluate_rag.py --export-csv            # Also save CSV
  python evaluate_rag.py --save-questions         # Save generated Q&A for reuse
        """,
    )
    parser.add_argument(
        "--num-questions", "-n", type=int, default=DEFAULT_NUM_QUESTIONS,
        help=f"Number of test questions to generate (default: {DEFAULT_NUM_QUESTIONS})",
    )
    parser.add_argument(
        "--questions-file", "-q", type=str, default=None,
        help="Path to pre-generated questions JSON (skips generation step)",
    )
    parser.add_argument(
        "--save-questions", "-s", action="store_true",
        help="Save generated questions to evaluation_reports/test_questions.json",
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Also export per-question results as CSV",
    )
    parser.add_argument(
        "--faiss-path", type=str, default=FAISS_INDEX_PATH,
        help=f"Path to FAISS index directory (default: {FAISS_INDEX_PATH})",
    )
    parser.add_argument(
        "--docs-path", type=str, default=DOCUMENTS_PATH,
        help=f"Path to source documents (default: {DOCUMENTS_PATH})",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Use hybrid retriever (BM25 + FAISS) instead of FAISS-only",
    )
    args = parser.parse_args()

    # Validate API key
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    evaluator = RAGEvaluator(faiss_path=args.faiss_path, docs_path=args.docs_path, use_hybrid=getattr(args, 'hybrid', False))

    # Step 1: Get test questions
    if args.questions_file:
        logger.info("Loading pre-generated questions from %s", args.questions_file)
        with open(args.questions_file, "r", encoding="utf-8") as f:
            qa_pairs = json.loads(f.read())
        logger.info("Loaded %d questions.", len(qa_pairs))
    else:
        qa_pairs = evaluator.generate_test_questions(args.num_questions)

        if args.save_questions:
            os.makedirs(REPORT_DIR, exist_ok=True)
            q_path = Path(REPORT_DIR) / "test_questions.json"
            with open(q_path, "w", encoding="utf-8") as f:
                json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
            logger.info("Test questions saved → %s", q_path)

    if not qa_pairs:
        logger.error("No test questions available. Exiting.")
        sys.exit(1)

    # Step 2: Run RAG pipeline
    rag_results = evaluator.run_rag_pipeline(qa_pairs)

    # Step 3: Ragas evaluation
    ragas_result = evaluator.evaluate(rag_results)

    # Step 4: Report
    evaluator.generate_report(
        ragas_result,
        qa_pairs,
        rag_results,
        export_csv=args.export_csv,
    )


if __name__ == "__main__":
    main()
