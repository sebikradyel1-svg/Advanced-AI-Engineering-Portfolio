# 🏢 HR Knowledge Assistant

> AI-Powered Company Policy Q&A System using RAG (Retrieval-Augmented Generation)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live Demo:** [https://hr-rag-assistant.onrender.com](https://hr-rag-assistant.onrender.com)

![HR RAG Assistant Demo](docs/demo.gif)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Local Development](#-local-development)
- [Docker Deployment](#-docker-deployment)
- [Cloud Deployment](#-cloud-deployment)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)

---

## 🎯 Overview

HR Knowledge Assistant is a production-ready RAG (Retrieval-Augmented Generation) system that enables employees to query company policies using natural language. The system retrieves relevant information from uploaded documents and generates accurate, contextual answers with source citations.

### Problem Solved

- ❌ Employees waste time searching through lengthy policy documents
- ❌ HR teams repeatedly answer the same questions
- ❌ Policy information is scattered across multiple documents
- ✅ **Solution:** Instant, accurate answers with source citations

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Semantic Search** | FAISS vector database for fast similarity search |
| 💬 **Conversational AI** | Multi-turn conversations with context memory |
| 📄 **Document Upload** | Upload custom policy documents (.txt) |
| 📚 **Source Citations** | Every answer includes relevant source passages |
| 🎨 **Modern UI** | Clean Gradio interface with dark/light themes |
| 🐳 **Docker Ready** | One-command deployment with Docker |
| ☁️ **Cloud Native** | Configured for Render.com free tier |
| 🔄 **CI/CD** | GitHub Actions for automated testing and deployment |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                    (Gradio Web Application)                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY PROCESSING                           │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   User      │───▶│  Embedding   │───▶│    FAISS        │    │
│  │   Query     │    │   Model      │    │  Vector Search  │    │
│  └─────────────┘    │ (MiniLM-L6)  │    │    (Top-K)      │    │
│                     └──────────────┘    └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE GENERATION                          │
│  ┌─────────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Retrieved     │───▶│   FLAN-T5    │───▶│   Formatted  │   │
│  │   Context       │    │   LLM        │    │   Response   │   │
│  │   + Chat History│    │              │    │   + Sources  │   │
│  └─────────────────┘    └──────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │  Document       │    │          FAISS Index             │   │
│  │  Processor      │───▶│   (In-Memory Vector Store)       │   │
│  │  (Chunking)     │    │                                  │   │
│  └─────────────────┘    └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Document Ingestion:** Policy documents are chunked (500 chars) with overlap (50 chars)
2. **Embedding:** Chunks are embedded using `all-MiniLM-L6-v2` (384 dimensions)
3. **Storage:** Embeddings stored in FAISS index for fast retrieval
4. **Query:** User question is embedded and matched against stored chunks
5. **Generation:** Top-K relevant chunks + query sent to FLAN-T5 for answer generation
6. **Response:** Answer returned with source citations

---

## 🛠 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | FLAN-T5 Base | Answer generation |
| **Embeddings** | all-MiniLM-L6-v2 | Semantic text encoding |
| **Vector Store** | FAISS | Fast similarity search |
| **Orchestration** | LangChain | RAG pipeline management |
| **UI** | Gradio | Web interface |
| **Containerization** | Docker | Deployment packaging |
| **CI/CD** | GitHub Actions | Automated pipeline |
| **Hosting** | Render.com | Cloud deployment |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 4GB+ RAM (for model loading)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hr-rag-assistant.git
cd hr-rag-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Open your browser at `http://localhost:7860`

---

## 💻 Local Development

### Project Structure

```
hr-rag-assistant/
├── app.py                 # Main application with Gradio UI
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── render.yaml           # Render deployment config
├── .github/
│   └── workflows/
│       └── ci-cd.yml     # GitHub Actions pipeline
├── docs/
│   └── demo.gif          # Demo animation
└── README.md
```

### Running Tests

```bash
# Install test dependencies
pip install pytest flake8

# Run linting
flake8 app.py --max-line-length=120

# Run import tests
python -c "from app import HRKnowledgeRAGSystem; print('OK')"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `7860` | Server port |
| `TRANSFORMERS_CACHE` | `.cache` | Model cache directory |
| `HF_HOME` | `.cache` | HuggingFace home directory |

---

## 🐳 Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t hr-rag-assistant .

# Run the container
docker run -p 7860:7860 hr-rag-assistant

# Run with environment variables
docker run -p 7860:7860 -e PORT=7860 hr-rag-assistant
```

### Docker Compose (Optional)

```yaml
version: '3.8'
services:
  hr-rag:
    build: .
    ports:
      - "7860:7860"
    environment:
      - PORT=7860
    restart: unless-stopped
```

---

## ☁️ Cloud Deployment

### Deploy to Render.com

1. **Fork this repository** to your GitHub account

2. **Create a new Web Service** on [Render Dashboard](https://dashboard.render.com)

3. **Connect your GitHub repository**

4. **Configure the service:**
   - Environment: `Docker`
   - Plan: `Free`
   - Region: `Frankfurt` (or closest to you)

5. **Set environment variables** (if needed):
   ```
   PYTHONUNBUFFERED=1
   ```

6. **Deploy!** Render will automatically build and deploy your app.

### Setting up Auto-Deploy with GitHub Actions

1. Go to your Render service dashboard
2. Navigate to **Settings** → **Deploy Hook**
3. Copy the Deploy Hook URL
4. In your GitHub repo, go to **Settings** → **Secrets** → **Actions**
5. Add a new secret: `RENDER_DEPLOY_HOOK_URL` with the copied URL

Now every push to `main` will automatically deploy to Render!

---

## 📖 API Reference

### Core Classes

#### `RAGConfig`
Configuration dataclass for system parameters.

```python
@dataclass
class RAGConfig:
    chunk_size: int = 500          # Document chunk size
    chunk_overlap: int = 50        # Overlap between chunks
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"
    top_k_retrieval: int = 3       # Number of chunks to retrieve
```

#### `HRKnowledgeRAGSystem`
Main RAG system class.

```python
# Initialize
rag = HRKnowledgeRAGSystem(config=RAGConfig())

# Load documents
status = rag.load_documents("policies.txt")

# Query
answer, sources = rag.chat("How many vacation days?")

# Clear memory
rag.clear_memory()
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Embedding Dimension** | 384 |
| **Average Query Time** | ~2-3s (CPU) |
| **Memory Usage** | ~2GB |
| **Chunk Size** | 500 chars |
| **Top-K Retrieval** | 3 documents |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Sebastian** - AI Engineer & ML Specialist

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Portfolio: [Your Portfolio](https://yourportfolio.com)

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the excellent RAG framework
- [Hugging Face](https://huggingface.co/) for pre-trained models
- [Gradio](https://gradio.app/) for the beautiful UI components
- [Render](https://render.com/) for free hosting

---

<p align="center">
  Made with ❤️ for better HR knowledge management
</p>
