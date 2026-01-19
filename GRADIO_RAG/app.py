# HR RAG Knowledge Assistant - Gradio Web Interface
# Production-ready deployment with document upload and source citations
# Compatible with LangChain 0.3+

import os
import tempfile
import gradio as gr
import torch
from typing import List, Tuple
from dataclasses import dataclass

# LangChain components (minimal imports for compatibility)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# Transformers for LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"
    top_k_retrieval: int = 3


class DocumentProcessor:
    """Processes and chunks documents for the RAG system."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " "]
        )

    def process_file(self, file_path: str) -> List[Document]:
        """Load and split a document into chunks."""
        logger.info(f"Processing document: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks


class HRKnowledgeRAGSystem:
    """Main RAG system for HR knowledge management - Simplified version."""
    
    def __init__(self, config: RAGConfig = RAGConfig()):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vector_db = None
        self.embeddings = None
        self.llm_pipeline = None
        self.chat_history: List[Tuple[str, str]] = []
        self.is_initialized = False
        logger.info(f"RAG System initialized on device: {self.device}")

    def setup_embeddings(self):
        """Initialize the embeddings model."""
        if self.embeddings is None:
            logger.info(f"Loading embeddings model: {self.config.embeddings_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embeddings_model
            )
        return self.embeddings

    def setup_llm(self):
        """Initialize the LLM pipeline."""
        if self.llm_pipeline is None:
            logger.info(f"Loading LLM: {self.config.llm_model}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.config.llm_model)
            
            # Move to GPU if available
            if self.device == "cuda":
                model = model.to("cuda")
            
            self.llm_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                device=0 if self.device == "cuda" else -1
            )
        return self.llm_pipeline

    def load_documents(self, file_path: str) -> str:
        """Load documents and initialize the system."""
        try:
            # Process documents
            processor = DocumentProcessor(self.config)
            docs = processor.process_file(file_path)
            
            # Setup embeddings and vector store
            self.setup_embeddings()
            logger.info("Building vector database...")
            self.vector_db = FAISS.from_documents(docs, self.embeddings)
            
            # Setup LLM
            self.setup_llm()
            
            self.is_initialized = True
            self.chat_history = []  # Reset history on new document
            
            return f"✅ Successfully loaded {len(docs)} document chunks. System ready!"
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return f"❌ Error loading documents: {str(e)}"

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for the LLM."""
        # Include recent chat history for context
        history_text = ""
        if self.chat_history:
            recent_history = self.chat_history[-3:]  # Last 3 exchanges
            history_parts = [f"Q: {q}\nA: {a}" for q, a in recent_history]
            history_text = "\n".join(history_parts)
        
        prompt = f"""Answer the question based ONLY on the following context. 
If the answer is not in the context, say "I don't have that information in the documents."
Be concise and professional.

Context:
{context}

{f"Previous conversation:{chr(10)}{history_text}{chr(10)}{chr(10)}" if history_text else ""}Question: {question}

Answer:"""
        return prompt

    def chat(self, question: str) -> Tuple[str, str]:
        """
        Process a question and return answer with sources.
        Returns: (answer, sources_text)
        """
        if not self.is_initialized:
            return "⚠️ System not initialized. Please upload a document first.", ""
        
        try:
            # Retrieve relevant documents
            docs = self.vector_db.similarity_search(
                question, 
                k=self.config.top_k_retrieval
            )
            
            # Build context from retrieved docs
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Build prompt and generate answer
            prompt = self._build_prompt(question, context)
            result = self.llm_pipeline(prompt, max_length=256, do_sample=False)
            answer = result[0]['generated_text'].strip()
            
            # Store in chat history
            self.chat_history.append((question, answer))
            
            # Format source citations
            sources = []
            for i, doc in enumerate(docs, 1):
                source_text = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                sources.append(f"**Source {i}:**\n{source_text}")
            
            sources_text = "\n\n".join(sources) if sources else "No sources retrieved."
            
            logger.info(f"Query processed: {question[:50]}...")
            return answer, sources_text
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"❌ Error processing query: {str(e)}", ""

    def clear_memory(self):
        """Clear conversation memory."""
        self.chat_history = []
        logger.info("Conversation memory cleared")
        return "🧹 Conversation history cleared."


# Global RAG system instance
rag_system = HRKnowledgeRAGSystem()


def process_upload(file) -> str:
    """Handle file upload and initialize the RAG system."""
    if file is None:
        return "⚠️ No file uploaded."
    
    try:
        file_path = file.name if hasattr(file, 'name') else file
        return rag_system.load_documents(file_path)
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return f"❌ Error processing upload: {str(e)}"


def chat_response(message: str, history: List[List[str]]) -> Tuple[str, str]:
    """Process chat message and return response with sources."""
    if not message.strip():
        return "", ""
    
    answer, sources = rag_system.chat(message)
    return answer, sources


def clear_chat():
    """Clear chat history and memory."""
    rag_system.clear_memory()
    return [], "", ""


# Sample policies for demo
SAMPLE_POLICIES = """
CHAPTER 1: WORKING HOURS AND ATTENDANCE
1.1. Standard working hours are 9:00 AM to 6:00 PM, Monday through Friday.
1.2. Employees are entitled to a 60-minute lunch break.
1.3. Remote work is allowed up to 2 days per week with prior manager approval.
1.4. Overtime must be pre-approved by department manager.

CHAPTER 2: LEAVE AND TIME OFF
2.1. Employees are entitled to 21 days of paid annual leave per year.
2.2. Sick leave is paid according to national legislation (up to 183 days).
2.3. Maternity leave consists of 126 calendar days with full salary.
2.4. Paternity leave is 10 working days.
2.5. Public holidays are non-working paid days.

CHAPTER 3: EMPLOYEE BENEFITS
3.1. Private medical insurance is provided for all full-time employees after probation.
3.2. Gym membership or fitness activity allowance of 150 RON/month.
3.3. Meal vouchers worth 30 RON per working day.
3.4. Annual budget of 2000 RON for training and professional development.
3.5. Work from home equipment allowance of 1500 RON (one-time).

CHAPTER 4: PERFORMANCE AND DEVELOPMENT
4.1. Performance reviews are conducted bi-annually (June and December).
4.2. Salary reviews are conducted annually in January.
4.3. Promotion eligibility requires minimum 12 months in current role.
4.4. Internal job postings are available before external recruitment.

CHAPTER 5: CODE OF CONDUCT
5.1. Professional dress code is business casual.
5.2. Confidentiality agreements must be signed by all employees.
5.3. Conflicts of interest must be disclosed to HR immediately.
5.4. Company equipment is for professional use only.
"""


def load_sample_data() -> str:
    """Load sample HR policies for demonstration."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(SAMPLE_POLICIES)
            temp_path = f.name
        
        result = rag_system.load_documents(temp_path)
        os.unlink(temp_path)
        return result
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return f"❌ Error loading sample data: {str(e)}"


def health_check():
    """Return system health status."""
    return {
        "status": "healthy",
        "device": rag_system.device,
        "initialized": rag_system.is_initialized
    }


# Build Gradio Interface
def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(
        title="HR Knowledge Assistant",
    ) as demo:
        
        gr.Markdown(
            """
            # 🏢 HR Knowledge Assistant
            ### AI-Powered Company Policy Q&A System
            
            Upload your company policies document or load sample data to get started.
            Ask questions in natural language and get instant answers with source citations.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400,
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g., How many vacation days do I get per year?",
                        scale=4,
                        show_label=False,
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                clear_btn = gr.Button("🗑️ Clear Chat")
                    
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Management")
                
                file_upload = gr.File(
                    label="Upload Policy Document (.txt)",
                    file_types=[".txt"],
                    type="filepath",
                )
                upload_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )
                
                sample_btn = gr.Button("📋 Load Sample Policies", variant="secondary")
                
                gr.Markdown("### 📚 Source Citations")
                sources_display = gr.Markdown(
                    value="*Sources will appear here after asking a question.*",
                )
        
        gr.Markdown("### 💡 Example Questions")
        gr.Examples(
            examples=[
                "How many vacation days do I have per year?",
                "What are the standard working hours?",
                "Is remote work allowed?",
                "What medical benefits are provided?",
                "What is the budget for professional development?",
            ],
            inputs=msg_input,
        )
        
        gr.Markdown(
            """
            ---
            **Tech Stack:** LangChain • FAISS • HuggingFace Transformers • Gradio  
            **Model:** FLAN-T5 Base | **Embeddings:** all-MiniLM-L6-v2
            """
        )
        
        # Event Handlers
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, "", ""
            
            answer, sources = chat_response(message, chat_history)
            # Gradio 6.0 format: list of dicts with 'role' and 'content'
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": answer})
            return chat_history, "", sources
        
        submit_btn.click(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, sources_display],
        )
        
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, sources_display],
        )
        
        clear_btn.click(
            lambda: ([], "", "*Sources will appear here after asking a question.*"),
            outputs=[chatbot, msg_input, sources_display],
        )
        
        file_upload.change(
            process_upload,
            inputs=[file_upload],
            outputs=[upload_status],
        )
        
        sample_btn.click(
            load_sample_data,
            outputs=[upload_status],
        )
    
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Starting HR Knowledge Assistant on port {port}")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    )
