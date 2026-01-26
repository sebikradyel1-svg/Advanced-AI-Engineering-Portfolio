# 🤖 Advanced AI Engineering Portfolio

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Latest-yellow)](https://huggingface.co/transformers)
[![LangChain](https://img.shields.io/badge/🦜%20LangChain-Latest-green)](https://langchain.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen)](https://github.com/features/actions)

> Production-ready AI/ML projects demonstrating expertise in **LLMs**, **RAG Systems**, **RLHF**, **Fine-tuning**, and **Deep Learning**

---

## 🎯 Portfolio Overview

This repository showcases **9 production-ready projects** covering the full spectrum of modern AI engineering:

| Category | Projects | Key Technologies |
|----------|----------|------------------|
| **LLM Fine-tuning** | Legal Text Generator | LoRA, PEFT, GPT-2, Gradio |
| **RAG Systems** | GRADIO_RAG, RAG-OPENAI | Groq, OpenAI, LangChain, FAISS |
| **Computer Vision** | Universal Image Classifier | VGG16, Transfer Learning, TensorFlow |
| **RLHF & Alignment** | KL Guard RLHF, Reward Model | PPO, TRL, PyTorch |
| **Deep Learning** | Transformers, Text Classification | Custom Attention, BERT |

---

## 🔗 Live Demos

| Project | Demo Link | Platform | Status |
|---------|-----------|----------|--------|
| **Legal Text Generator** | [huggingface.co/spaces/KradyelSebi/legal-text-generator](https://huggingface.co/spaces/KradyelSebi/legal-text-generator) | HuggingFace | ✅ Live |
| **GRADIO RAG System** | [advanced-ai-engineering-portfolio.onrender.com](https://advanced-ai-engineering-portfolio.onrender.com/) | Render | ✅ Live |
| **Universal Image Classifier** | [huggingface.co/spaces/KradyelSebi/image-classifier](https://huggingface.co/spaces/KradyelSebi/image-classifier) | HuggingFace | ✅ Live |

---

## 🚀 Featured Projects

---

### 📜 1. Legal & Business Text Generator
> **Fine-tuned LLM for professional document generation using LoRA**

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Try%20Now-blue)](https://huggingface.co/spaces/KradyelSebi/legal-text-generator)

| Aspect | Details |
|--------|---------|
| **Location** | [`legal-text-generator/`](./legal-text-generator) |
| **Tech Stack** | GPT-2 Medium, LoRA/PEFT, Gradio, HuggingFace Spaces |
| **Live Demo** | [huggingface.co/spaces/KradyelSebi/legal-text-generator](https://huggingface.co/spaces/KradyelSebi/legal-text-generator) |

**Highlights:**
- ✅ Parameter-efficient fine-tuning (only **0.4% params** trained)
- ✅ 8 document types: contracts, NDAs, policies, emails, meeting minutes
- ✅ **Production deployed** on HuggingFace Spaces
- ✅ Custom training with 200+ legal document examples

**Skills Demonstrated:** `LLM Fine-tuning` `LoRA/PEFT` `Model Deployment` `Gradio`

---

### 🌐 2. GRADIO RAG System (Groq-Powered)
> **Interactive RAG chatbot with Groq LLM and Gradio interface**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Render-green)](https://advanced-ai-engineering-portfolio.onrender.com/)

| Aspect | Details |
|--------|---------|
| **Location** | [`GRADIO_RAG/`](./GRADIO_RAG) |
| **Tech Stack** | Groq API, LangChain, FAISS, Gradio, HuggingFace Embeddings |
| **Live Demo** | [advanced-ai-engineering-portfolio.onrender.com](https://advanced-ai-engineering-portfolio.onrender.com/) |

**Highlights:**
- ✅ Lightning-fast inference with **Groq LPU**
- ✅ Interactive web interface with Gradio
- ✅ Document upload and semantic search
- ✅ Pre-built FAISS index for instant loading

**Skills Demonstrated:** `RAG Architecture` `Groq API` `Vector Search` `Web Interface`

---

### 🐾 3. Universal Image Classifier
> **Computer Vision with Transfer Learning using VGG16**

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Try%20Now-blue)](https://huggingface.co/spaces/KradyelSebi/image-classifier)

| Aspect | Details |
|--------|---------|
| **Location** | [`Animal-Classifier/`](./Animal-Classifier) |
| **Tech Stack** | TensorFlow, VGG16, Keras, Gradio, HuggingFace Spaces |
| **Live Demo** | [huggingface.co/spaces/KradyelSebi/image-classifier](https://huggingface.co/spaces/KradyelSebi/image-classifier) |

**Highlights:**
- ✅ Pre-trained VGG16 backbone with custom classification head
- ✅ **95%+ accuracy** on multi-class classification
- ✅ **Production deployed** on HuggingFace Spaces
- ✅ Data augmentation pipeline for robust predictions

**Skills Demonstrated:** `Transfer Learning` `CNN` `Image Classification` `TensorFlow` `Model Deployment`

---

### 🔑 4. RAG System with OpenAI
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
- ✅ Custom **KL-divergence guard** for distribution stability
- ✅ Prevents reward hacking and catastrophic forgetting
- ✅ Full PPO training loop implementation
- ✅ Configurable KL penalty coefficients

<details>
<summary>📊 RLHF Pipeline Architecture</summary>

![RLHF Pipeline](images/RLHF%20Pipeline.png)

</details>

**Skills Demonstrated:** `RLHF` `PPO Algorithm` `Model Alignment` `PyTorch`

---

### ⭐ 6. Reward Model Training
> **Training reward models for human preference learning**

| Aspect | Details |
|--------|---------|
| **Location** | [`Reward-Model-Training/`](./Reward-Model-Training) |
| **Tech Stack** | PyTorch, Transformers, Custom Training Loop |

**Highlights:**
- ✅ Pairwise preference learning
- ✅ Bradley-Terry model implementation
- ✅ Integration with RLHF pipeline
- ✅ Custom loss functions for ranking

**Skills Demonstrated:** `Reward Modeling` `Preference Learning` `Custom Training`

---

### 📈 7. Custom Transformer Stock Predictor
> **Ground-up Transformer implementation for financial time-series forecasting**

| Aspect | Details |
|--------|---------|
| **Location** | [`Transformers/`](./Transformers) |
| **Tech Stack** | PyTorch, NumPy, Custom Architecture |

**Highlights:**
- ✅ **Multi-Head Attention from scratch**
- ✅ Positional Encoding implementation
- ✅ 15% improvement over LSTM baseline
- ✅ 50+ engineered technical indicators

<details>
<summary>📊 Transformer Encoder Architecture</summary>

![Transformer Encoder](images/Transformes_encoder.png)

</details>

**Skills Demonstrated:** `Transformer Architecture` `Attention Mechanisms` `Time Series`

---

### 📝 8. Text Sentiment Analysis (BERT)
> **BERT-based sentiment classification with transfer learning**

| Aspect | Details |
|--------|---------|
| **Location** | [`Transfer_Learning_Text/`](./Transfer_Learning_Text) |
| **Tech Stack** | BERT, HuggingFace Transformers, PyTorch |

**Highlights:**
- ✅ Fine-tuned BERT for sentiment analysis
- ✅ Custom tokenization pipeline
- ✅ Evaluation metrics and visualization
- ✅ Production-ready inference

<details>
<summary>📊 Transfer Learning Comparison</summary>

![Transfer Learning](images/2_transfer_learning_comparison.png)

</details>

**Skills Demonstrated:** `NLP` `BERT Fine-tuning` `Text Classification`

---

## ⚙️ CI/CD Pipeline

This repository includes **GitHub Actions** for automated workflows:

| Workflow | Purpose |
|----------|---------|
| Deployment Health Check | Automated testing and validation |

Location: [`.github/workflows/`](./.github/workflows)

---

## 🛠️ Technical Skills

### Core Technologies
```
Python │ PyTorch │ TensorFlow │ HuggingFace Transformers │ LangChain │ OpenAI API │ Groq API
```

### AI/ML Expertise
```
├── 🔤 LLM & NLP
│   ├── Fine-tuning (LoRA, PEFT, Full)
│   ├── Prompt Engineering
│   ├── Text Generation & Classification
│   └── BERT, GPT-2, Llama
│
├── 📚 RAG Systems
│   ├── Vector Databases (FAISS, ChromaDB)
│   ├── Document Processing & Chunking
│   ├── Semantic Search & Retrieval
│   └── OpenAI & Groq Integration
│
├── 🎯 RLHF & Alignment
│   ├── Reward Modeling
│   ├── PPO Training
│   ├── KL-Divergence Control
│   └── Human Preference Learning
│
├── 👁️ Computer Vision
│   ├── Transfer Learning (VGG16)
│   ├── Image Classification
│   ├── Data Augmentation
│   └── TensorFlow/Keras
│
└── 🧠 Deep Learning
    ├── Transformers (from scratch)
    ├── CNNs & Transfer Learning
    ├── Attention Mechanisms
    └── Time Series Forecasting
```

### MLOps & Deployment
```
Docker │ Git │ GitHub Actions │ HuggingFace Spaces │ Render │ Gradio │ TensorBoard │ FAISS
```

---

## 📁 Repository Structure

```
Advanced-AI-Engineering-Portfolio/
│
├── 📜 legal-text-generator/           # LLM Fine-tuning with LoRA (DEPLOYED)
│   ├── training/
│   ├── app/
│   └── screenshots/
│
├── 🌐 GRADIO_RAG/                     # RAG with Groq + Gradio (DEPLOYED)
│   └── gradio_rag_app.py
│
├── 🐾 Animal-Classifier/              # CNN Transfer Learning (DEPLOYED)
│   └── universal_classifier.py
│
├── 🔑 RAG-OPENAI/                     # RAG with OpenAI
│   └── rag_openai.py
│
├── 🎯 KL_Guard_RLHF/                  # RLHF with KL Guard
│   └── KL_Guard_RLHF.py
│
├── ⭐ Reward-Model-Training/          # Reward Model
│   └── reward_model_training.py
│
├── 📈 Transformers/                   # Custom Transformer
│   └── transformer_stock_prediction.py
│
├── 📝 Transfer_Learning_Text/         # BERT Fine-tuning
│   └── transfer_learning_text.py
│
├── 📂 docs/                           # Documentation & FAISS indices
│
├── 🖼️ images/                         # Architecture diagrams
│
├── ⚙️ .github/workflows/              # CI/CD Pipelines
│
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

| Project | Difficulty | Lines of Code | Key Challenge | Status |
|---------|------------|---------------|---------------|--------|
| Legal Text Generator | ⭐⭐⭐⭐ | 3000+ | LoRA fine-tuning + deployment | ✅ Deployed |
| GRADIO_RAG | ⭐⭐⭐ | 500+ | Groq integration + UI | ✅ Deployed |
| Image Classifier | ⭐⭐⭐ | 400+ | Transfer learning + deployment | ✅ Deployed |
| KL_Guard_RLHF | ⭐⭐⭐⭐⭐ | 800+ | PPO + KL divergence control | ✅ Complete |
| Custom Transformer | ⭐⭐⭐⭐⭐ | 600+ | Attention from scratch | ✅ Complete |
| Reward Model | ⭐⭐⭐⭐ | 400+ | Preference learning | ✅ Complete |

---

## 📫 Contact

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
  <i>Specializing in LLMs, RAG Systems, RLHF, Computer Vision, and Production ML</i>
</p>
