# 🤖 HR RAG Assistant - Production AI System

[![Deploy Status](https://img.shields.io/badge/deploy-render-success)](https://advanced-ai-engineering-portfolio.onrender.com/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

> **AI-powered HR document analysis system with production deployment, monitoring, and CI/CD automation.**

🔗 **Live Demo:** [https://advanced-ai-engineering-portfolio.onrender.com/](https://advanced-ai-engineering-portfolio.onrender.com/)

---

## 📋 Overview

The HR RAG Assistant is a production-ready AI system that provides instant, accurate answers to HR policy questions using Retrieval-Augmented Generation (RAG). Built with enterprise-grade architecture for low-memory deployment.

### Key Features
- ⚡ **Production Deployment** - Live on Render with auto-deploy
- 🤖 **RAG System** - FLAN-T5 + FAISS for intelligent Q&A
- 📊 **Real-time Monitoring** - Health checks, metrics, logging
- 🔄 **CI/CD Pipeline** - GitHub Actions automation
- 💾 **Memory Optimized** - Runs on 512MB RAM

---

## 🏗️ Architecture
```
User Query → Gradio UI → RAG System → FLAN-T5 Model → Response
                              ↓
                         FAISS Vector DB
                              ↓
                      HR Policy Documents
```

### Components
- **Frontend**: Gradio web interface
- **Vector DB**: FAISS for semantic search
- **LLM**: Google FLAN-T5-base (local)
- **Embeddings**: all-MiniLM-L6-v2
- **Monitoring**: Custom metrics + health checks
- **Deployment**: Render + GitHub Actions

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 2GB+ RAM (development)
- 512MB+ RAM (production)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/sebikradyel1-svg/Advanced-AI-Engineering-Portfolio.git
cd Advanced-AI-Engineering-Portfolio/GRADIO_RAG

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run locally
python app.py
```

Open: `http://localhost:7860`

---

## 💻 Usage

### Load Sample Policies
1. Click **"Load Sample Policies"** button
2. Wait for models to load (~30 seconds)
3. Start asking questions!

### Example Queries
- "How many vacation days do I get per year?"
- "What are the standard working hours?"
- "Is remote work allowed?"
- "What medical benefits are provided?"

### Upload Custom Documents
1. Prepare `.txt` file with your policies
2. Use "Upload Policy Document" section
3. System processes and indexes automatically

---

## 📊 Production Features

### CI/CD Pipeline
- **Auto-deploy** on every push to `main`
- **Automated testing** (linting, imports)
- **Health checks** post-deployment
- **GitHub Actions** workflow

### Monitoring & Logging
- **Multi-level logging** (console + files)
- **Metrics tracking** (uptime, queries, errors)
- **Health dashboard** in UI
- **Daily log rotation**

### Performance
- **Average response**: 3-5 seconds
- **Memory footprint**: <512MB RAM
- **Concurrent users**: 5-10
- **Uptime**: 99%+ (Render free tier)

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Frontend** | Gradio 4.x | Web interface |
| **Vector DB** | FAISS | Semantic search |
| **LLM** | FLAN-T5-base | Text generation |
| **Embeddings** | HuggingFace | Document vectorization |
| **Framework** | LangChain | RAG orchestration |
| **Deployment** | Render | Cloud hosting |
| **CI/CD** | GitHub Actions | Automation |

---

## 🌐 Deployment

### Deploy to Render

1. Fork this repository
2. Create Render account
3. New Web Service → Connect GitHub
4. Configure:
```
   Build: pip install -r requirements.txt
   Start: python app.py
```
5. Add environment variables (if needed)
6. Deploy! 🚀

### GitHub Actions Setup
1. Get Render API Key: https://dashboard.render.com/account/api-keys
2. Get Service ID from Render URL
3. Add GitHub Secrets:
   - `RENDER_API_KEY`
   - `RENDER_SERVICE_ID`
4. Push to `main` → auto-deploy!

---

## 📈 Roadmap

### Version 1.1
- [ ] Multi-language support (Romanian + English)
- [ ] Enhanced analytics dashboard
- [ ] User authentication
- [ ] Query history export

### Version 2.0
- [ ] Groq API integration option
- [ ] Larger model support
- [ ] Mobile app
- [ ] API endpoints

---

## 👤 Author

**Sebastian Kradyel**
- AI Engineer & ML Specialist
- 📧 Email: paulsebastianlradyel@gmail.com
- 💼 LinkedIn: www.linkedin.com/in/paul-sebastian-kradyel
- 🐙 GitHub: [@sebikradyel1-svg](https://github.com/sebikradyel1-svg)

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **LangChain** - RAG framework
- **HuggingFace** - Models and embeddings
- **Meta/FAISS** - Vector database
- **Google** - FLAN-T5 model
- **Gradio** - UI framework

---

<div align="center">

**⭐ If you find this project useful, please give it a star! ⭐**

Made with ❤️ by Sebastian Kradyel | 2026

</div>
