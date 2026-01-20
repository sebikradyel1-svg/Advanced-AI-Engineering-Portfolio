# 🏢 HR RAG Knowledge Assistant

An AI-powered HR policy Q&A system using Retrieval-Augmented Generation (RAG). Ask questions about company policies and get instant, accurate answers with source citations.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.11-green)
![Gradio](https://img.shields.io/badge/Gradio-4.21.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌐 Live Demo

**[Try it here →](https://advanced-ai-engineering-portfolio.onrender.com)**

> ⚠️ Note: Free tier may take 30-60 seconds to wake up on first visit.

---

## 🎯 Features

- **Instant Policy Lookup** - Ask questions in natural language
- **Source Citations** - See exactly which documents support each answer
- **Conversation Memory** - Follow-up questions understand context
- **Custom Document Upload** - Load your own HR policies (.txt)
- **Low-Memory Optimized** - Runs on 512MB RAM (Render free tier)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | FLAN-T5 Small (google/flan-t5-small) |
| **Embeddings** | all-MiniLM-L6-v2 (sentence-transformers) |
| **Vector Store** | FAISS (faiss-cpu) |
| **Framework** | LangChain |
| **UI** | Gradio |
| **Deployment** | Docker + Render |

---

## 📁 Project Structure

```
GRADIO_RAG/
├── app.py                  # Main application (Gradio UI + RAG logic)
├── Dockerfile              # Docker configuration for deployment
├── requirements.txt        # Python dependencies (CPU-optimized)
├── company_policies.txt    # Sample HR policies document
├── faiss_index/            # Pre-built vector index
│   ├── index.faiss
│   └── index.pkl
├── render.yaml             # Render deployment config
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Use the Live Demo
Click the demo link above and try it instantly!

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/sebikradyel1-svg/Advanced-AI-Engineering-Portfolio.git
cd Advanced-AI-Engineering-Portfolio/GRADIO_RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open `http://localhost:7860` in your browser.

### Option 3: Run with Docker

```bash
docker build -t hr-rag-assistant .
docker run -p 7860:7860 hr-rag-assistant
```

---

## 💡 Usage

1. **Click "Load Sample Policies"** to initialize the system
2. **Ask a question** like:
   - "How many vacation days do I get per year?"
   - "What are the standard working hours?"
   - "Is remote work allowed?"
   - "What medical benefits are provided?"
3. **View source citations** to verify answers

---

## ⚡ Optimizations for Low-Memory Deployment

This project is optimized to run on **512MB RAM** (Render free tier):

| Optimization | Description |
|--------------|-------------|
| **CPU-only PyTorch** | `torch==2.2.1+cpu` - No CUDA libraries (~1GB saved) |
| **FLAN-T5-Small** | Smaller model (~150MB vs ~1GB for base) |
| **Garbage Collection** | `gc.collect()` after each query |
| **torch.no_grad()** | Disabled gradients during inference |
| **low_cpu_mem_usage** | Efficient model loading |
| **Pre-built FAISS Index** | No embedding computation at runtime |
| **Thread Limiting** | `OMP_NUM_THREADS=1` to reduce memory spikes |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│   Gradio    │────▶│  RAG System │
│  Question   │     │     UI      │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
            ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
            │  Embeddings   │         │    FAISS      │         │   FLAN-T5     │
            │  (MiniLM-L6)  │         │  Vector DB    │         │   (Answer)    │
            └───────────────┘         └───────────────┘         └───────────────┘
```

**Flow:**
1. User asks a question
2. Question is embedded using MiniLM-L6-v2
3. FAISS retrieves top-3 relevant document chunks
4. Context + Question sent to FLAN-T5
5. Answer generated with source citations

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Cold Start | ~30-60 seconds |
| Query Response | ~3-5 seconds |
| Memory Usage | ~350MB |
| Model Load Time | ~15 seconds |

---

## 🔧 Configuration

Edit `RAGConfig` in `app.py` to customize:

```python
@dataclass
class RAGConfig:
    chunk_size: int = 500           # Document chunk size
    chunk_overlap: int = 50         # Overlap between chunks
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-small"
    top_k_retrieval: int = 3        # Number of chunks to retrieve
    faiss_index_path: str = "faiss_index"
```

---

## 📝 Adding Custom Documents

1. Create a `.txt` file with your HR policies
2. Either:
   - **Upload via UI**: Use the file upload in the app
   - **Pre-build index**: Run `build_index.py` locally and push `faiss_index/`

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| 502 Bad Gateway | Wait 30-60 sec for cold start |
| Out of Memory | Reduce `chunk_size` or `top_k_retrieval` |
| Slow responses | Normal for free tier CPU |
| Models not loading | Check Render logs for errors |

---

## 🗺️ Roadmap

- [ ] PDF document support
- [ ] Multi-language support
- [ ] Conversation export
- [ ] ONNX optimization for faster inference
- [ ] Streaming responses

---

## 📄 License

MIT License - feel free to use for your own projects!

---

## 👨‍💻 Author

**Sebastian** - AI Engineer

- GitHub: [@sebikradyel1-svg](https://github.com/sebikradyel1-svg)
- Portfolio: [Advanced-AI-Engineering-Portfolio](https://github.com/sebikradyel1-svg/Advanced-AI-Engineering-Portfolio)

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - RAG framework
- [HuggingFace](https://huggingface.co/) - Models and transformers
- [Gradio](https://gradio.app/) - UI framework
- [Render](https://render.com/) - Deployment platform

---

**⭐ Star this repo if you found it useful!**
