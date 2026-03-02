# 🛠 Advanced AI Engineering Portfolio

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Latest-yellow)](https://huggingface.co/transformers)
[![LangChain](https://img.shields.io/badge/🦜%20LangChain-Latest-green)](https://langchain.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen)](https://github.com/features/actions)

> Production-ready AI/ML projects demonstrating expertise in **LLMs**, **RAG Systems**, **RLHF**, **Fine-tuning**, **Computer Vision**, and **System Design**

---

## 🎯 Portfolio Overview

This repository showcases **9 production-ready projects** covering the full spectrum of modern AI engineering:

| Category | Projects | Key Technologies |
|----------|----------|------------------|
| **LLM Fine-tuning** | Legal Text Generator | Phi-3-mini QLoRA, 4-bit NF4, HuggingFace Hub |
| **RAG Systems** | GRADIO_RAG, RAG-OPENAI | Hybrid BM25+FAISS+RRF, RAGAS Evaluation, LangChain |
| **Computer Vision** | Universal Image Classifier | VGG16, Grad-CAM Explainability, TensorFlow |
| **RLHF & Alignment** | KL Guard RLHF, Reward Model | PPO, TRL, PyTorch |
| **Deep Learning** | Transformers, Text Classification | Custom Attention, BERT |

### 🏗️ System Design & Documentation
All major projects include **Architecture Decision Records (ADRs)**, **Mermaid system diagrams**, **evaluation frameworks**, and **trade-off analysis** — demonstrating production engineering practices beyond just code.

---

## 🔗 Live Demos

| Project | Demo Link | Platform | Status |
|---------|-----------|----------|--------|
| **Legal Text Generator** | [HuggingFace Space](https://huggingface.co/spaces/KradyelSebi/legal-text-generator) | HuggingFace | ✅ Live |
| **Phi-3 QLoRA Adapters** | [HuggingFace Hub](https://huggingface.co/KradyelSebi/legal-text-phi3-lora) | HuggingFace | ✅ Live |
| **GRADIO RAG System** | [Render](https://advanced-ai-engineering-portfolio.onrender.com/) | Render | ✅ Live |
| **Universal Image Classifier** | [HuggingFace Space](https://huggingface.co/spaces/KradyelSebi/image-classifier) | HuggingFace | ✅ Live |

---

## 🚀 Featured Projects

---

### 📣 1. Legal & Business Text Generator
> **Migrated from GPT-2 to Phi-3-mini (3.8B) using QLoRA 4-bit quantization**

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Try%20Now-blue)](https://huggingface.co/spaces/KradyelSebi/legal-text-generator)
[![Model](https://img.shields.io/badge/🤗%20Model-Phi3%20QLoRA-orange)](https://huggingface.co/KradyelSebi/legal-text-phi3-lora)

| Aspect | Details |
|--------|---------|
| **Location** | [`legal-text-generator/`](./legal-text-generator) |
| **Tech Stack** | Phi-3-mini 3.8B, QLoRA (4-bit NF4), BitsAndBytes, PEFT, HuggingFace Hub |
| **Previous** | GPT-2 Medium (355M) with LoRA |

**Highlights:**
- ✅ **10x model upgrade**: GPT-2 (355M) → Phi-3-mini (3.8B) via QLoRA 4-bit NF4 quantization
- ✅ Fits 3.8B model in **6.4GB VRAM** on RTX 3060 with double quantization
- ✅ QLoRA adapters (r=32, alpha=64) — only **2.44% trainable parameters** (50M of 2B)
- ✅ Training loss 1.24 → 0.88; paged AdamW 8-bit optimizer + gradient checkpointing
- ✅ **ADR-004** documents migration decision with alternatives analysis (Mistral-7B, Llama-3.2-3B) and VRAM budget breakdown
- ✅ Adapter weights on [HuggingFace Hub](https://huggingface.co/KradyelSebi/legal-text-phi3-lora)
- ✅ 8 document types: contracts, NDAs, privacy policies, emails, meeting minutes, proposals

**Skills Demonstrated:** `QLoRA` `4-bit Quantization` `PEFT` `Model Migration` `ADR Documentation` `VRAM Optimization`

---

### 🌍 2. Enterprise HR RAG System
> **Production RAG with Hybrid Retrieval (BM25+FAISS+RRF) and RAGAS Evaluation**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Render-green)](https://advanced-ai-engineering-portfolio.onrender.com/)

| Aspect | Details |
|--------|---------|
| **Location** | [`GRADIO_RAG/`](./GRADIO_RAG) |
| **Tech Stack** | Groq Llama 3.3 70B, LangChain, Hybrid BM25+FAISS+RRF, RAGAS, AWS EC2, Docker |
| **Architecture** | [ADRs](./GRADIO_RAG/docs/adr/) · [System Diagrams](./GRADIO_RAG/docs/) |

**Highlights:**
- ✅ **Hybrid retrieval**: BM25 sparse search + FAISS dense vectors + Reciprocal Rank Fusion (RRF)
- ✅ **RAGAS evaluation framework**: context_precision, faithfulness, answer_relevancy with synthetic test generation
- ✅ **Token-aware conversation memory** with automatic follow-up detection for multi-turn dialogues
- ✅ Sub-2s query response across 500+ indexed HR policy documents via Groq LPU
- ✅ **Architecture Decision Records**: embedding model selection, hybrid retrieval trade-offs, LLM provider evaluation
- ✅ **Mermaid system design diagrams** documenting data flow and component architecture
- ✅ 37-test suite with pytest (72% coverage); CI/CD via GitHub Actions; MLflow experiment tracking
- ✅ Prometheus-style monitoring, structured JSONL logging, health check endpoints

**Skills Demonstrated:** `Hybrid Retrieval` `BM25+FAISS+RRF` `RAGAS Evaluation` `ADRs` `System Design` `MLOps` `Docker` `AWS EC2`

---

### 🐾 3. Universal Image Classifier
> **Computer Vision with VGG16 Transfer Learning + Grad-CAM Explainability**

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Try%20Now-blue)](https://huggingface.co/spaces/KradyelSebi/image-classifier)

| Aspect | Details |
|--------|---------|
| **Location** | [`image-classifier-project/`](./image-classifier-project) |
| **Tech Stack** | TensorFlow, VGG16, Grad-CAM, OpenCV, Streamlit, HuggingFace Spaces |

**Highlights:**
- ✅ **95%+ accuracy** on multi-class classification across 12 categories (10,000+ images)
- ✅ **Grad-CAM explainability**: visual heatmaps showing model attention regions per prediction
- ✅ Top-k class Grad-CAM comparison for debugging misclassifications
- ✅ Data augmentation pipeline (rotation, flip, zoom, brightness) reducing overfitting by 30%
- ✅ Streamlit UI with real-time inference and interactive attention visualization
- ✅ **Production deployed** on HuggingFace Spaces

**Skills Demonstrated:** `Transfer Learning` `Grad-CAM/XAI` `CNN` `Image Classification` `TensorFlow` `Model Explainability`

---

### 🔎 4. RAG System with OpenAI
> **Enterprise RAG implementation using OpenAI embeddings and GPT**

| Aspect | Details |
|--------|---------|
| **Location** | [`RAG-OPENAI/`](./RAG-OPENAI) |
| **Tech Stack** | OpenAI API, LangChain, FAISS, Python |

**Highlights:**
- ✅ OpenAI embeddings for superior semantic understanding
- ✅ GPT-powered response generation
- ✅ Configurable chunking strategies
- ✅ Production-ready error handling

**Skills Demonstrated:** `OpenAI API` `RAG Pipeline` `Embeddings` `LangChain`

---

### 🎯 5. RLHF with KL-Divergence Guard
> **Reinforcement Learning from Human Feedback with safety constraints**

| Aspect | Details |
|--------|---------|
| **Location** | [`KL_Guard_RLHF/`](./KL_Guard_RLHF) |
| **Tech Stack** | PyTorch, TRL, PPO, Transformers |

**Highlights:**
- ✅ Custom **KL-divergence guard** (<5%) for distribution stability
- ✅ Prevents reward hacking and catastrophic forgetting
- ✅ Full PPO training loop with gradient checkpointing and FP16
- ✅ Stable optimization across 10,000+ training steps on RTX 3060

**Skills Demonstrated:** `RLHF` `PPO Algorithm` `Model Alignment` `PyTorch`

---

### ⭐ 6. Reward Model Training
> **Training reward models for human preference learning**

| Aspect | Details |
|--------|---------|
| **Location** | [`Reward-Model-Training/`](./Reward-Model-Training) |
| **Tech Stack** | PyTorch, Transformers, Custom Training Loop |

**Highlights:**
- ✅ Pairwise preference learning with Bradley-Terry model
- ✅ Integration with RLHF pipeline
- ✅ Custom loss functions for ranking

**Skills Demonstrated:** `Reward Modeling` `Preference Learning` `Custom Training`

---

### 📊 7. Custom Transformer Stock Predictor
> **Ground-up Transformer implementation for financial time-series forecasting**

| Aspect | Details |
|--------|---------|
| **Location** | [`Transformers/`](./Transformers) |
| **Tech Stack** | PyTorch, NumPy, Custom Architecture |

**Highlights:**
- ✅ **Multi-Head Attention from scratch** with Positional Encoding
- ✅ 15% RMSE improvement over LSTM baseline
- ✅ 50+ engineered technical indicators
- ✅ Walk-forward backtesting with interactive Plotly dashboard

**Skills Demonstrated:** `Transformer Architecture` `Attention Mechanisms` `Time Series`

---

### 📥 8. Text Sentiment Analysis (BERT)
> **BERT-based sentiment classification with transfer learning**

| Aspect | Details |
|--------|---------|
| **Location** | [`Transfer_Learning_Text/`](./Transfer_Learning_Text) |
| **Tech Stack** | BERT, HuggingFace Transformers, PyTorch |

**Highlights:**
- ✅ Fine-tuned BERT for sentiment analysis
- ✅ Custom tokenization pipeline
- ✅ Production-ready inference

**Skills Demonstrated:** `NLP` `BERT Fine-tuning` `Text Classification`

---

## 🏗️ System Design & Architecture

A key differentiator of this portfolio is the emphasis on **engineering discipline** beyond code:

| Practice | Where Applied |
|----------|---------------|
| **Architecture Decision Records (ADRs)** | HR RAG (embedding model, retrieval strategy, LLM provider), Legal Text Generator (Phi-3 migration) |
| **RAGAS Evaluation Framework** | HR RAG — context_precision, faithfulness, answer_relevancy metrics |
| **Mermaid System Diagrams** | HR RAG data flow, component architecture, retrieval pipeline |
| **Trade-off Analysis** | Phi-3 vs Mistral-7B vs Llama-3.2-3B (VRAM, quality, latency), Groq vs OpenAI vs local LLM |
| **VRAM Budget Planning** | QLoRA training: 2GB model + 0.1GB adapters + 0.3GB optimizer + 3-4GB activations = 6-7GB |
| **Evaluation-Driven Development** | RAGAS baseline → hybrid retrieval optimization → re-measure cycle |

---

## ⚙️ CI/CD Pipeline

| Workflow | Purpose |
|----------|---------|
| GitHub Actions | Automated testing, linting, deployment health checks |
| Docker | Containerized deployment for HR RAG on AWS EC2 |
| MLflow | Experiment tracking and parameter versioning |
| pytest | 37 tests, 72% code coverage |

---

## 🛠️ Technical Skills

### Core Technologies
```
Python │ PyTorch │ TensorFlow │ HuggingFace Transformers │ LangChain │ OpenAI API │ Groq API
```

### AI/ML Expertise
```
├── 🔑 LLM & NLP
│   ├── QLoRA/LoRA/PEFT Fine-tuning (Phi-3, GPT-2, BERT)
│   ├── Hybrid RAG (BM25 + FAISS + RRF Fusion)
│   ├── RAGAS Evaluation Framework
│   └── Prompt Engineering & Context Optimization
│
├── 📚 RAG Systems
│   ├── Hybrid Retrieval (BM25 sparse + FAISS dense + RRF)
│   ├── Vector Databases (FAISS, ChromaDB)
│   ├── Token-aware Conversation Memory
│   └── Semantic Chunking & Search
│
├── 🎯 RLHF & Alignment
│   ├── PPO with KL-Divergence Guards
│   ├── Reward Modeling & Preference Learning
│   └── Memory-efficient training (FP16, gradient checkpointing)
│
├── 👁️ Computer Vision
│   ├── Transfer Learning (VGG16)
│   ├── Grad-CAM Visual Explanations
│   ├── Data Augmentation Pipelines
│   └── TensorFlow/Keras
│
├── 🏗️ System Design
│   ├── Architecture Decision Records (ADRs)
│   ├── VRAM Budget Planning & Trade-off Analysis
│   ├── Mermaid System Diagrams
│   └── Evaluation-Driven Development
│
└── 🧠 Deep Learning
    ├── Transformers from scratch
    ├── 4-bit NF4 Quantization (BitsAndBytes)
    ├── Attention Mechanisms
    └── Time Series Forecasting
```

### MLOps & Deployment
```
Docker │ AWS EC2 │ GitHub Actions │ MLflow │ pytest │ HuggingFace Spaces/Hub │ Render │ Gradio │ Streamlit
```

---

## 📁 Repository Structure

```
Advanced-AI-Engineering-Portfolio/
│
├── 📣 legal-text-generator/           # Phi-3 QLoRA Fine-tuning (DEPLOYED)
│   ├── training/train_phi3.py         # QLoRA training script
│   ├── training/phi3_legal_lora/      # Adapter config & metadata
│   ├── app/app_phi3.py               # Phi-3 inference app
│   ├── docs/adr/ADR-004-phi3-migration.md
│   └── requirements_phi3.txt
│
├── 🌍 GRADIO_RAG/                     # Hybrid RAG + RAGAS Eval (DEPLOYED)
│   ├── app.py                         # Main RAG application
│   ├── hybrid_retriever.py            # BM25 + FAISS + RRF
│   ├── conversation_memory.py         # Token-aware memory
│   ├── evaluate_rag.py                # RAGAS evaluation
│   ├── docs/adr/                      # Architecture Decision Records
│   └── docs/diagrams/                 # Mermaid system diagrams
│
├── 🐾 image-classifier-project/       # VGG16 + Grad-CAM (DEPLOYED)
│   ├── app/streamlit_app.py           # Streamlit UI
│   ├── app/grad_cam.py                # Grad-CAM module
│   └── universal_classifier.py
│
├── 🔎 RAG-OPENAI/                     # RAG with OpenAI
├── 🎯 KL_Guard_RLHF/                  # RLHF with KL Guard
├── ⭐ Reward-Model-Training/          # Reward Model
├── 📊 Transformers/                   # Custom Transformer
├── 📥 Transfer_Learning_Text/         # BERT Fine-tuning
├── ⚙️ .github/workflows/              # CI/CD Pipelines
└── 📄 README.md
```

---

## 🎓 Certifications

| Certification | Issuer | Courses | Date | Verify |
|---------------|--------|---------|------|--------|
| **IBM AI Engineering Professional Certificate** | IBM | 13 courses | Oct 2025 | [Verify](https://coursera.org/verify/professional-cert/YPCZ6H8FMLV6) |
| **IBM Data Science Professional Certificate** | IBM | 12 courses | Sep 2025 | [Verify](https://coursera.org/verify/professional-cert/LKYLRICBD43U) |
| **AWS DevOps and AI Specialization** | Amazon Web Services | 3 courses | Jan 2026 | [Verify](https://coursera.org/verify/specialization/TH9OFS3NCUUV) |
| **Machine Learning in Production** | DeepLearning.AI (Andrew Ng) | 1 course | Jan 2026 | [Verify](https://coursera.org/verify/L3FE9JK5W80G) |

<details>
<summary>📚 <b>Full Course List (29 courses)</b></summary>

### IBM AI Engineering (13 courses)
- Machine Learning with Python
- Introduction to Deep Learning & Neural Networks with Keras
- Deep Learning with Keras and TensorFlow
- Introduction to Neural Networks and PyTorch
- Deep Learning with PyTorch
- AI Capstone Project with Deep Learning
- Generative AI and LLMs: Architecture and Data Preparation
- Gen AI Foundational Models for NLP & Language Understanding
- Generative AI Language Modeling with Transformers
- Generative AI Engineering and Fine-Tuning Transformers
- Generative AI Advance Fine-Tuning for LLMs
- Fundamentals of AI Agents Using RAG and LangChain
- Project: Generative AI Applications with RAG and LangChain

### IBM Data Science (12 courses)
- What is Data Science?
- Tools for Data Science
- Data Science Methodology
- Python for Data Science, AI & Development
- Python Project for Data Science
- Databases and SQL for Data Science with Python
- Data Analysis with Python
- Data Visualization with Python
- Machine Learning with Python
- Applied Data Science Capstone
- Generative AI: Elevate Your Data Science Career
- Data Scientist Career Guide and Interview Preparation

### AWS DevOps and AI (3 courses)
- DevOps and AI on AWS: Upgrading Apps with Generative AI
- DevOps and AI on AWS: CI/CD for Generative AI Applications
- DevOps and AI on AWS: AIOps

### DeepLearning.AI
- Machine Learning in Production (MLOps) - by Andrew Ng

</details>

---

## 📊 Project Complexity Matrix

| Project | Difficulty | Key Challenge | Status |
|---------|------------|---------------|--------|
| Legal Text Generator | ⭐⭐⭐⭐⭐ | QLoRA migration GPT-2→Phi-3 + ADR | ✅ Deployed |
| HR RAG System | ⭐⭐⭐⭐⭐ | Hybrid retrieval + RAGAS eval + ADRs | ✅ Deployed |
| Image Classifier | ⭐⭐⭐⭐ | VGG16 + Grad-CAM explainability | ✅ Deployed |
| KL_Guard_RLHF | ⭐⭐⭐⭐⭐ | PPO + KL divergence control | ✅ Complete |
| Custom Transformer | ⭐⭐⭐⭐⭐ | Attention from scratch | ✅ Complete |
| Reward Model | ⭐⭐⭐⭐ | Preference learning | ✅ Complete |
| RAG-OpenAI | ⭐⭐⭐ | OpenAI RAG pipeline | ✅ Complete |

---

## 📬 Contact

<p align="center">
  <a href="https://www.linkedin.com/in/paul-sebastian-kradyel">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin" alt="LinkedIn">
  </a>
  <a href="mailto:paulsebastiankradyel@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact-red?style=for-the-badge&logo=gmail" alt="Email">
  </a>
  <a href="https://huggingface.co/KradyelSebi">
    <img src="https://img.shields.io/badge/🤗%20HuggingFace-Profile-yellow?style=for-the-badge" alt="HuggingFace">
  </a>
</p>

---

<p align="center">
  <b>🚀 Open to AI/ML Engineering Opportunities</b><br>
  <i>Specializing in LLMs, RAG Systems, RLHF, Computer Vision, System Design, and Production ML</i>
</p>
