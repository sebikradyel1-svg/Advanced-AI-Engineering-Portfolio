# HR Knowledge RAG System - Complete Documentation

## Overview

An intelligent **Retrieval-Augmented Generation (RAG)** system designed for HR knowledge management. This system allows employees to ask questions about company policies in natural language (Romanian) and receive accurate, contextual answers backed by official documentation.

## What is RAG?

**Retrieval-Augmented Generation** combines:

1. **Information Retrieval**: Finding relevant documents from a knowledge base
2. **Language Generation**: Using an LLM to generate natural answers
3. **Grounding**: Ensuring answers are based on actual documents, not hallucinations

```
User Question → Retrieve Relevant Docs → Generate Answer → Return with Sources
```

## Key Features

- ✅ **Multilingual Support**: Optimized for Romanian language
- ✅ **Document Chunking**: Intelligent text splitting for better retrieval
- ✅ **Vector Search**: Fast semantic search using embeddings
- ✅ **Conversational Memory**: Maintains chat history for context
- ✅ **Source Attribution**: Shows which documents support each answer
- ✅ **Multiple Vector Stores**: Support for FAISS and Chroma
- ✅ **No Hallucinations**: Only answers from provided documents
- ✅ **Easy Setup**: Automatic sample data generation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  HR RAG System                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐      ┌────────────────────┐     │
│  │ HR Documents     │─────▶│ Document Processor │     │
│  │ (.txt, .pdf)     │      │ (Chunking)         │     │
│  └──────────────────┘      └────────┬───────────┘     │
│                                     │                   │
│                            ┌────────▼───────────┐      │
│                            │ Embedding Model    │      │
│                            │ (Multilingual)     │      │
│                            └────────┬───────────┘      │
│                                     │                   │
│                            ┌────────▼───────────┐      │
│                            │  Vector Store      │      │
│                            │  (FAISS/Chroma)    │      │
│                            └────────┬───────────┘      │
│                                     │                   │
│  ┌──────────────────┐      ┌───────▼────────────┐     │
│  │ User Question    │─────▶│ Dense Retriever    │     │
│  │ (Romanian)       │      │ (Top-K Search)     │     │
│  └──────────────────┘      └────────┬───────────┘     │
│                                     │                   │
│                            ┌────────▼───────────┐      │
│                            │ Conversational     │      │
│                            │ Agent + LLM        │      │
│                            │ (FLAN-T5)          │      │
│                            └────────┬───────────┘      │
│                                     │                   │
│                            ┌────────▼───────────┐      │
│                            │ Answer + Sources   │      │
│                            └────────────────────┘      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

```bash
# Core dependencies
pip install torch transformers
pip install langchain langchain-community
pip install sentence-transformers
pip install faiss-cpu  # or faiss-gpu for GPU support
pip install chromadb   # Alternative vector store

# Optional for PDF support
pip install pypdf
```

### Full Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install langchain>=0.1.0
pip install langchain-community>=0.0.10
pip install sentence-transformers>=2.2.0
pip install faiss-cpu>=1.7.4
pip install chromadb>=0.4.0
pip install numpy>=1.24.0
```

### Quick Start Verification

```python
python hr_rag_system_updated.py
```

This will:
1. Create a sample `companyPolicies.txt` file
2. Index the documents
3. Answer 3 demo questions
4. Start interactive mode

## Configuration

### RAGConfig Class

```python
@dataclass
class RAGConfig:
    # Document processing
    chunk_size: int = 500           # Characters per chunk
    chunk_overlap: int = 100        # Overlap between chunks
    
    # Models
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "google/flan-t5-base"
    
    # Retrieval
    top_k_retrieval: int = 5        # Number of docs to retrieve
    
    # Paths
    docs_path: str = "./company_docs"
    vector_store_path: str = "./vector_store"
    vector_db_type: str = "faiss"   # "faiss" or "chroma"
```

### Custom Configuration

```python
from hr_rag_system_updated import RAGConfig, HRKnowledgeRAGSystem

# Create custom config
config = RAGConfig(
    chunk_size=300,                 # Smaller chunks
    top_k_retrieval=3,              # Retrieve fewer docs
    llm_model="google/flan-t5-large",  # Larger model
    vector_db_type="chroma"         # Use Chroma instead of FAISS
)

# Initialize with custom config
rag_system = HRKnowledgeRAGSystem(config=config)
rag_system.setup("your_policies.txt")
```

## Core Components

### 1. DocumentProcessor

Handles document loading and intelligent chunking.

```python
processor = DocumentProcessor(config)

# Load documents
documents = processor.load_documents("policies.txt")

# Split into chunks
chunks = processor.split_documents(documents)
# Each chunk has:
# - page_content: The text
# - metadata: {"chunk_id": 0, "source_type": "hr_policy"}
```

**Chunking Strategy:**
- Uses `RecursiveCharacterTextSplitter`
- Splits on: `\n\n` → `\n` → `.` → `!` → `?` → `,` → ` ` → `""`
- Maintains context with overlapping chunks

### 2. DensePassageRetriever

Creates and manages the vector database.

```python
retriever = DensePassageRetriever(config)

# Initialize embeddings
retriever.initialize_embeddings()

# Create vector store (first time)
retriever.create_vector_store(chunks)

# Load existing vector store
retriever.load_vector_store()
```

**Embedding Model:**
- Default: `paraphrase-multilingual-MiniLM-L12-v2`
- Supports 50+ languages including Romanian
- 384-dimensional dense vectors
- Normalized for cosine similarity

**Vector Stores:**

| Feature | FAISS | Chroma |
|---------|-------|--------|
| Speed | ⚡ Very Fast | 🚀 Fast |
| Memory | 💾 Efficient | 💾 Moderate |
| Persistence | File-based | Database |
| Filtering | Limited | Advanced |
| Best For | Production | Development |

### 3. ConversationalHRAgent

The intelligent agent that answers questions.

```python
agent = ConversationalHRAgent(config, retriever)

# Initialize LLM
agent.initialize_llm()

# Create QA chain
agent.create_qa_chain()

# Ask questions
response = agent.query("Câte zile de concediu am?")
print(response['answer'])
print(response['source_documents'])
```

**Features:**
- **Conversational Memory**: Remembers last 10 messages
- **Source Attribution**: Returns documents used
- **Prompt Engineering**: Romanian-optimized prompts
- **No Hallucinations**: Explicitly instructed to only use context

## Usage Examples

### Basic Usage

```python
from hr_rag_system_updated import HRKnowledgeRAGSystem

# Initialize system
rag_system = HRKnowledgeRAGSystem()

# Setup with your documents
rag_system.setup("company_policies.txt")

# Ask questions
answer = rag_system.chat("Care este programul de lucru?")
# Returns: "Programul standard de lucru este de luni până vineri, 
#           între orele 09:00 și 18:00."
```

### Interactive Mode

```python
# Start interactive chat
print("\n💬 MOD INTERACTIV (scrie 'exit' pentru ieșire)")
while True:
    question = input("\nÎntrebare: ")
    if question.lower().strip() == "exit":
        break
    rag_system.chat(question)
```

### Batch Processing

```python
questions = [
    "Câte zile de concediu am pe an?",
    "Care este programul de lucru?",
    "Ce beneficii medicale oferă compania?",
    "Câte zile pe săptămână pot lucra de acasă?",
    "Cât este bonusul anual?"
]

for question in questions:
    answer = rag_system.chat(question)
```

### Custom Document Loading

```python
# Multiple files
for file_path in ["policies.txt", "benefits.txt", "procedures.txt"]:
    documents = rag_system.document_processor.load_documents(file_path)
    chunks = rag_system.document_processor.split_documents(documents)
    # Add to vector store
    rag_system.retriever.vector_store.add_documents(chunks)
```

## Document Format

### Supported Formats

Currently supports `.txt` files. Can be extended for:
- PDF documents
- Word documents (.docx)
- Markdown files (.md)
- HTML pages

### Recommended Structure

```text
MANUAL DE POLITICI INTERNE - COMPANIA XYZ

CAPITOLUL 1: TITLU CAPITOL
1.1. Prima politică cu detalii clare.
1.2. A doua politică cu informații specifice.
1.3. A treia politică cu date concrete.

CAPITOLUL 2: ALT CAPITOL
2.1. Politică cu cifre exacte: 21 de zile.
2.2. Politică cu termene: între 09:00 și 18:00.
2.3. Politică cu valori: 30 RON per zi.
```

**Best Practices:**
- ✅ Use clear headings and sections
- ✅ Include specific numbers and dates
- ✅ Be explicit and unambiguous
- ✅ Use consistent formatting
- ✅ Avoid vague language

### Sample Policy File

```python
def create_sample_policies_file():
    sample_content = """MANUAL DE POLITICI INTERNE

CAPITOLUL 1: PROGRAMUL DE LUCRU
1.1. Program: luni-vineri, 09:00-18:00
1.2. Pauza de prânz: 13:00-14:00
1.3. Munca de acasă: max 2 zile/săptămână cu aprobare

CAPITOLUL 2: CONCEDII
2.1. Concediu anual: 21 zile lucrătoare
2.2. Concediu medical: conform legislației
2.3. Concediu maternitate: 126 zile

CAPITOLUL 3: BENEFICII
3.1. Asigurare medicală privată (normă întreagă)
3.2. Abonament fitness
3.3. Bonuri de masă: 30 RON/zi
3.4. Budget training: 2000 RON/an
3.5. Bonus anual: 10-25% din salariu
"""
    with open("companyPolicies.txt", "w", encoding="utf-8") as f:
        f.write(sample_content)
```

## Advanced Features

### 1. Conversation Memory

```python
# Get conversation history
history = rag_system.agent.get_conversation_history()
for message in history:
    print(f"{message['type']}: {message['content']}")

# Clear memory for new conversation
rag_system.agent.clear_memory()
```

### 2. Custom Prompts

```python
# Modify the prompt template
custom_prompt = """Ești un expert HR. 
Context: {context}
Istoric: {chat_history}
Întrebare: {question}

Răspunde profesional și detaliat:"""

agent = ConversationalHRAgent(config, retriever)
agent.prompt_template = custom_prompt
```

### 3. Retrieval Parameters

```python
# Adjust number of retrieved documents
config.top_k_retrieval = 10  # More context

# Or modify at runtime
retriever_config = {"k": 3, "fetch_k": 20}
base_retriever = retriever.vector_store.as_retriever(
    search_kwargs=retriever_config
)
```

### 4. Document Metadata Filtering

```python
# Add custom metadata
for chunk in chunks:
    chunk.metadata["department"] = "HR"
    chunk.metadata["version"] = "2025-Q1"
    chunk.metadata["confidential"] = False

# Filter during retrieval (with Chroma)
results = retriever.vector_store.similarity_search(
    query="concediu",
    k=5,
    filter={"department": "HR", "confidential": False}
)
```

## LLM Models

### Supported Models

| Model | Size | Language | Speed | Quality |
|-------|------|----------|-------|---------|
| `flan-t5-base` | 250M | Multi | 🚀🚀🚀 | ⭐⭐⭐ |
| `flan-t5-large` | 780M | Multi | 🚀🚀 | ⭐⭐⭐⭐ |
| `flan-t5-xl` | 3B | Multi | 🚀 | ⭐⭐⭐⭐⭐ |
| `mt5-base` | 580M | Multi | 🚀🚀 | ⭐⭐⭐ |

### Changing Models

```python
# Smaller, faster model
config.llm_model = "google/flan-t5-small"

# Larger, better quality
config.llm_model = "google/flan-t5-large"

# Romanian-specific (if available)
config.llm_model = "dumitrescustefan/bert-base-romanian-uncased-v1"
```

### Generation Parameters

```python
# In ConversationalHRAgent.initialize_llm()
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,      # Max answer length
    temperature=0.7,         # Randomness (0=deterministic, 1=creative)
    top_k=50,               # Consider top 50 tokens
    top_p=0.95,             # Nucleus sampling
    do_sample=True          # Enable sampling
)
```

## Performance Optimization

### 1. GPU Acceleration

```python
# Automatic GPU detection
model_kwargs = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Force specific device
model_kwargs = {'device': 'cuda:0'}  # Use first GPU
model_kwargs = {'device': 'cpu'}     # Force CPU
```

### 2. Batch Processing

```python
# Process multiple questions efficiently
def batch_query(questions: List[str]):
    answers = []
    for q in questions:
        result = rag_system.agent.query(q)
        answers.append(result['answer'])
    return answers
```

### 3. Caching

```python
# FAISS saves to disk automatically
rag_system.retriever.vector_store.save_local("./vector_store")

# Chroma persists automatically
# No action needed

# Load from cache
if os.path.exists("./vector_store"):
    rag_system.retriever.load_vector_store()
else:
    rag_system.setup("policies.txt")
```

### 4. Memory Management

```python
# Clear memory after long sessions
if len(rag_system.agent.memory.chat_memory.messages) > 50:
    rag_system.agent.clear_memory()

# Limit retrieval for faster responses
config.top_k_retrieval = 3  # Instead of 5
```

## Monitoring & Debugging

### Enable Verbose Mode

```python
# ConversationalRetrievalChain with verbose=True shows:
# - Retrieved documents
# - LLM input/output
# - Chain steps

self.chain = ConversationalRetrievalChain.from_llm(
    llm=self.llm,
    retriever=base_retriever,
    verbose=True  # Enable detailed logging
)
```

### Check Retrieved Documents

```python
# See what documents were retrieved
response = rag_system.agent.query("Câte zile de concediu?")
print("\n📄 DOCUMENTE RECUPERATE:")
for i, doc in enumerate(response['source_documents'], 1):
    print(f"\n{i}. Score: {doc.get('score', 'N/A')}")
    print(f"   Content: {doc['content']}")
    print(f"   Metadata: {doc['metadata']}")
```

### Test Retrieval Quality

```python
# Direct similarity search
query = "concediu anual"
docs = rag_system.retriever.vector_store.similarity_search_with_score(
    query, k=5
)

for doc, score in docs:
    print(f"\nScore: {score:.4f}")
    print(f"Content: {doc.page_content[:100]}...")
```

## Error Handling

### Common Issues

**1. Model Download Errors**
```python
# Pre-download models
from transformers import AutoModel
AutoModel.from_pretrained("google/flan-t5-base", cache_dir="./models")
```

**2. Encoding Errors**
```python
# Always use UTF-8
with open("policies.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

**3. Out of Memory**
```python
# Use smaller model
config.llm_model = "google/flan-t5-small"

# Reduce chunk size
config.chunk_size = 300

# Retrieve fewer documents
config.top_k_retrieval = 3
```

**4. FAISS Deserialization Warning**
```python
# Safe to ignore or suppress
self.vector_store = FAISS.load_local(
    path,
    embeddings,
    allow_dangerous_deserialization=True  # Explicitly allow
)
```

## Production Deployment

### 1. API Server

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
rag_system = HRKnowledgeRAGSystem()
rag_system.setup("company_policies.txt")

class Question(BaseModel):
    text: str

@app.post("/ask")
async def ask_question(question: Question):
    response = rag_system.agent.query(question.text)
    return {
        "answer": response['answer'],
        "sources": response['source_documents']
    }

# Run: uvicorn api:app --host 0.0.0.0 --port 8000
```

### 2. Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Pre-download models
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
               AutoTokenizer.from_pretrained('google/flan-t5-base'); \
               AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')"

CMD ["python", "hr_rag_system_updated.py"]
```

### 3. Environment Variables

```python
import os

config = RAGConfig(
    embeddings_model=os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/..."),
    llm_model=os.getenv("LLM_MODEL", "google/flan-t5-base"),
    vector_store_path=os.getenv("VECTOR_STORE_PATH", "./vector_store")
)
```

## Best Practices

### 1. Document Preparation
- ✅ Clean and format documents consistently
- ✅ Remove duplicates and outdated information
- ✅ Use clear section headers
- ✅ Include dates and version numbers

### 2. System Configuration
- ✅ Start with default settings
- ✅ Tune chunk_size based on document structure
- ✅ Adjust top_k based on corpus size
- ✅ Monitor memory usage

### 3. Prompt Engineering
- ✅ Be explicit about language (Romanian)
- ✅ Instruct to cite sources
- ✅ Tell it to say "I don't know" when uncertain
- ✅ Request concise answers

### 4. Quality Assurance
- ✅ Test with common questions
- ✅ Verify source attribution
- ✅ Check for hallucinations
- ✅ Update documents regularly

## Troubleshooting

### Low Quality Answers

```python
# 1. Increase retrieved documents
config.top_k_retrieval = 10

# 2. Reduce chunk size for more precise retrieval
config.chunk_size = 300
config.chunk_overlap = 50

# 3. Use larger LLM
config.llm_model = "google/flan-t5-large"

# 4. Improve prompt
agent.prompt_template = """Detailed instructions..."""
```

### Slow Performance

```python
# 1. Use smaller model
config.llm_model = "google/flan-t5-small"

# 2. Reduce max tokens
max_new_tokens=100  # Instead of 150

# 3. Use GPU
# Install: pip install faiss-gpu
# Automatic detection in code

# 4. Reduce retrieval
config.top_k_retrieval = 3
```

### Incorrect Answers

```python
# 1. Check document quality
print(chunks[0].page_content)

# 2. Test retrieval directly
docs = retriever.vector_store.similarity_search("question", k=5)
for doc in docs:
    print(doc.page_content)

# 3. Improve prompts with examples
# 4. Add validation logic
```

## Extensions

### Add PDF Support

```python
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()
```

### Add Web Scraping

```python
from langchain_community.document_loaders import WebBaseLoader

def load_website(url: str):
    loader = WebBaseLoader(url)
    return loader.load()
```

### Add Multi-language Support

```python
from langdetect import detect

def detect_and_route(question: str):
    lang = detect(question)
    if lang == "ro":
        return romanian_prompt
    elif lang == "en":
        return english_prompt
```

## FAQ

**Q: Can this work with English documents?**

A: Yes! The multilingual embedding model supports 50+ languages. Just adjust the prompt template language.

**Q: How many documents can it handle?**

A: FAISS scales to millions of vectors. For most companies, thousands of documents work perfectly.

**Q: Does it work offline?**

A: Yes! After downloading models once, everything runs locally without internet.

**Q: How do I update documents?**

A: Re-run `create_vector_store()` with updated chunks, or add incrementally with `add_documents()`.

**Q: Can I use my own LLM?**

A: Yes! Any HuggingFace text-generation model works. Adjust `config.llm_model`.

## References

### Key Papers
- **RAG**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- **Dense Retrieval**: Karpukhin et al., "Dense Passage Retrieval" (2020)
- **FLAN-T5**: Chung et al., "Scaling Instruction-Finetuned Language Models" (2022)

### Resources
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)

---

**Version**: 1.0  
**Last Updated**: 2025  
**Language**: Romanian (primary), Multilingual support  
**License**: Educational/Research Use