# HR RAG Knowledge Assistant - Gradio Web Interface
# ULTRA-OPTIMIZED for Render free tier (512MB RAM)

import os
import gc
import tempfile
import gradio as gr
import torch
from typing import List, Tuple
from dataclasses import dataclass
import threading

# Memory optimizations - set BEFORE importing heavy libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Limit torch threads
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# LangChain components
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


def clear_memory():
    """Force garbage collection to free RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "facebook/bart-tiny"  # Small model for 512MB RAM
    top_k_retrieval: int = 3
    faiss_index_path: str = "faiss_index"


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
    
    def process_text(self, text: str) -> List[Document]:
        """Process raw text into document chunks."""
        docs = [Document(page_content=text, metadata={"source": "uploaded"})]
        chunks = self.text_splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks


class HRKnowledgeRAGSystem:
    """Main RAG system - ULTRA-OPTIMIZED for low RAM."""
    
    def __init__(self, config: RAGConfig = RAGConfig()):
        self.config = config
        self.device = "cpu"  # Force CPU for Render free tier
        self.vector_db = None
        self.embeddings = None
        self.llm_pipeline = None
        self.tokenizer = None
        self.model = None
        self.chat_history: List[Tuple[str, str]] = []
        self.is_initialized = False
        self.models_ready = False
        self._loading_lock = threading.Lock()
        logger.info(f"RAG System created on device: {self.device}")

    def _prewarm_models(self):
        """Pre-load models in background thread."""
        try:
            logger.info("🔥 Pre-warming models in background...")
            self.setup_embeddings()
            clear_memory()  # Clean up after embeddings
            self.setup_llm()
            clear_memory()  # Clean up after LLM
            self.models_ready = True
            logger.info("✅ Models pre-warmed and ready!")
        except Exception as e:
            logger.error(f"Pre-warm failed: {e}")

    def start_prewarm(self):
        """Start background pre-warming of models."""
        thread = threading.Thread(target=self._prewarm_models, daemon=True)
        thread.start()
        logger.info("🚀 Started background model pre-warming")

    def setup_embeddings(self):
        """Initialize the embeddings model."""
        with self._loading_lock:
            if self.embeddings is None:
                logger.info(f"Loading embeddings model: {self.config.embeddings_model}")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.embeddings_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                clear_memory()
        return self.embeddings

    def setup_llm(self):
        """Initialize the LLM pipeline with memory optimizations."""
        with self._loading_lock:
            if self.llm_pipeline is None:
                logger.info(f"Loading LLM: {self.config.llm_model}")
                
                # Load with memory optimizations
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.llm_model,
                    use_fast=True
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.llm_model,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,
                    offload_folder="offload" # Aceasta va folosi discul în loc de RAM dacă e nevoie
                )
                
                # Set to evaluation mode (saves memory)
                self.model.eval()
                
                self.llm_pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=256,  # Reduced for memory
                    device=-1  # CPU
                )
                
                clear_memory()
                logger.info("✅ LLM loaded successfully")
        return self.llm_pipeline

    def load_prebuilt_index(self) -> str:
        """Load pre-built FAISS index - uses pre-warmed models if ready."""
        try:
            index_path = self.config.faiss_index_path
            
            if not os.path.exists(index_path):
                logger.warning(f"Pre-built index not found at {index_path}")
                return "⚠️ Pre-built index not found. Please upload a document."
            
            logger.info(f"Loading pre-built FAISS index from {index_path}...")
            
            # Setup embeddings
            self.setup_embeddings()
            clear_memory()
            
            # Load pre-built index
            self.vector_db = FAISS.load_local(
                index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            clear_memory()
            
            # Setup LLM
            self.setup_llm()
            clear_memory()
            
            self.is_initialized = True
            self.chat_history = []
            
            logger.info("✅ Pre-built index loaded successfully!")
            return "✅ Sample policies loaded! System ready - ask me anything!"
            
        except Exception as e:
            logger.error(f"Error loading pre-built index: {e}")
            return f"❌ Error loading index: {str(e)}"

    def load_documents(self, file_path: str) -> str:
        """Load NEW documents (for custom uploads)."""
        try:
            processor = DocumentProcessor(self.config)
            docs = processor.process_file(file_path)
            clear_memory()
            
            self.setup_embeddings()
            logger.info("Building vector database from uploaded document...")
            self.vector_db = FAISS.from_documents(docs, self.embeddings)
            clear_memory()
            
            self.setup_llm()
            clear_memory()
            
            self.is_initialized = True
            self.chat_history = []
            
            return f"✅ Successfully loaded {len(docs)} document chunks. System ready!"
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return f"❌ Error loading documents: {str(e)}"

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the prompt for the LLM."""
        history_text = ""
        if self.chat_history:
            recent_history = self.chat_history[-2:]  # Keep only 2 for memory
            history_parts = [f"Q: {q}\nA: {a}" for q, a in recent_history]
            history_text = "\n".join(history_parts)
        
        # Shorter, more efficient prompt
        prompt = f"""Answer based on this context only. Be concise.

Context:
{context[:1500]}

{f"History:{chr(10)}{history_text}{chr(10)}" if history_text else ""}Question: {question}

Answer:"""
        return prompt

    def chat(self, question: str) -> Tuple[str, str]:
        """Process a question and return answer with sources."""
        if not self.is_initialized:
            return "⚠️ System not initialized. Please click 'Load Sample Policies' or upload a document.", ""
        
        try:
            # Retrieve relevant documents
            docs = self.vector_db.similarity_search(
                question, 
                k=self.config.top_k_retrieval
            )
            
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate answer
            prompt = self._build_prompt(question, context)
            
            with torch.no_grad():  # Disable gradient computation for inference
                result = self.llm_pipeline(
                    prompt, 
                    max_length=128,  # Shorter output for memory
                    do_sample=False,
                    num_beams=1  # Greedy decoding (less memory)
                )
            
            answer = result[0]['generated_text'].strip()
            
            # Keep limited history
            self.chat_history.append((question, answer))
            if len(self.chat_history) > 5:
                self.chat_history = self.chat_history[-5:]
            
            # Format sources
            sources = []
            for i, doc in enumerate(docs, 1):
                source_text = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                sources.append(f"**Source {i}:**\n{source_text}")
            
            sources_text = "\n\n".join(sources) if sources else "No sources retrieved."
            
            # Clean up memory after each query
            clear_memory()
            
            logger.info(f"Query processed: {question[:50]}...")
            return answer, sources_text
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            clear_memory()
            return f"❌ Error processing query: {str(e)}", ""

    def clear_chat_memory(self):
        """Clear conversation memory."""
        self.chat_history = []
        clear_memory()
        logger.info("Conversation memory cleared")
        return "🧹 Conversation history cleared."


# Global RAG system instance
rag_system = HRKnowledgeRAGSystem()

# Pre-warm models at startup (uncomment if you have enough RAM)
# rag_system.start_prewarm()


def process_upload(file) -> str:
    """Handle file upload."""
    if file is None:
        return "⚠️ No file uploaded."
    
    try:
        file_path = file.name if hasattr(file, 'name') else file
        return rag_system.load_documents(file_path)
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return f"❌ Error processing upload: {str(e)}"


def chat_response(message: str, history: List[List[str]]) -> Tuple[str, str]:
    """Process chat message."""
    if not message.strip():
        return "", ""
    
    answer, sources = rag_system.chat(message)
    return answer, sources


def load_sample_data() -> str:
    """Load pre-built sample policies."""
    if rag_system.models_ready:
        logger.info("⚡ Models already pre-warmed - loading will be instant!")
    else:
        logger.info("⏳ Loading models on demand...")
    return rag_system.load_prebuilt_index()


def get_status() -> str:
    """Return current system status."""
    if rag_system.is_initialized:
        return "🟢 System ready! Ask a question."
    elif rag_system.models_ready:
        return "🟢 Models ready! Click 'Load Sample Policies'."
    else:
        return "🟡 Click 'Load Sample Policies' to start."


def health_check():
    """Return system health status."""
    return {
        "status": "healthy",
        "device": rag_system.device,
        "initialized": rag_system.is_initialized,
        "models_ready": rag_system.models_ready,
        "prebuilt_index_exists": os.path.exists("faiss_index")
    }


# Build Gradio Interface
def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="HR Knowledge Assistant") as demo:
        
        gr.Markdown(
            """
            # 🏢 HR Knowledge Assistant
            ### AI-Powered Company Policy Q&A System
            
            Click **"Load Sample Policies"** to get started, or upload your own document.
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
                
                # Status indicator
                status_display = gr.Textbox(
                    label="System Status",
                    value=get_status(),
                    interactive=False,
                    lines=1,
                )
                
                # Refresh status button
                refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
                
                # PROMINENT SAMPLE BUTTON
                sample_btn = gr.Button(
                    "📋 Load Sample Policies", 
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("*Or upload your own:*")
                
                file_upload = gr.File(
                    label="Upload Policy Document (.txt)",
                    file_types=[".txt"],
                    type="filepath",
                )
                
                upload_status = gr.Textbox(
                    label="Load Status",
                    interactive=False,
                    lines=2,
                    value="👆 Click 'Load Sample Policies' to start!"
                )
                
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
                "How do I request time off?",
                "What is the 401k matching policy?",
            ],
            inputs=msg_input,
        )
        
        gr.Markdown(
            """
            ---
            **Tech Stack:** LangChain • FAISS • HuggingFace Transformers • Gradio  
            **Model:** FLAN-T5 Small | **Embeddings:** all-MiniLM-L6-v2  
            **Optimized for:** Low-memory deployment (512MB RAM)
            """
        )
        
        # Event Handlers
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, "", ""
            
            answer, sources = chat_response(message, chat_history)
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
        
        refresh_btn.click(
            get_status,
            outputs=[status_display],
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
    )
