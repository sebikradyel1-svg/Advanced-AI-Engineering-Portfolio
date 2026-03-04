# 📜 Legal & Business Text Generator

> **Phi-3-mini (3.8B) fine-tuned with QLoRA 4-bit quantization** for professional legal and business document generation. Migrated from GPT-2 Medium with documented architecture decision.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/KradyelSebi/legal-text-generator)
[![Model Weights](https://img.shields.io/badge/🤗%20Model-Phi3%20QLoRA-orange)](https://huggingface.co/KradyelSebi/legal-text-phi3-lora)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

This project demonstrates **Parameter-Efficient Fine-Tuning** using **QLoRA (4-bit quantization + LoRA)** to create a domain-specific language model for generating legal and business documents.

### Evolution: GPT-2 → Phi-3-mini

| | GPT-2 Medium (v1) | Phi-3-mini (v2 — Current) |
|--|-------------------|--------------------------|
| **Parameters** | 355M | 3.8B (10x increase) |
| **Method** | LoRA | QLoRA (4-bit NF4) |
| **Trainable** | 0.4% (1.5M) | 2.44% (50M) |
| **VRAM Usage** | 4-6 GB | 6.4 GB |
| **Coherence** | 5/10 | 8/10 |
| **Legal Accuracy** | 4/10 | 7/10 |
| **Instruction Following** | 6/10 | 9/10 |
| **Context Window** | 1K tokens | 4K tokens |
| **Training Loss** | ~0.9 | 1.24 → 0.88 |
| **Training Time** | ~45 min | ~5 min |

**Why migrate?** See [ADR-004: Phi-3 Migration Decision](docs/adr/ADR-004-phi3-migration.md) for full alternatives analysis and VRAM budget breakdown.

---

## 🚀 Live Demos & Models

| Resource | Link |
|----------|------|
| **Live Demo (GPT-2)** | [HuggingFace Space](https://huggingface.co/spaces/KradyelSebi/legal-text-generator) |
| **Phi-3 QLoRA Weights** | [HuggingFace Hub](https://huggingface.co/KradyelSebi/legal-text-phi3-lora) |
| **Full Portfolio** | [GitHub Repository](https://github.com/sebikradyel1-svg/Advanced-AI-Engineering-Portfolio) |

---

## 🏗️ Architecture

### Phi-3-mini QLoRA (Current)

```
┌──────────────────────────────────────────────────┐
│           microsoft/phi-3-mini-4k-instruct       │
│              (3.8B parameters)                    │
│         4-bit NF4 Quantization ❄️                │
│         Double Quantization Enabled               │
├──────────────────────────────────────────────────┤
│              QLoRA Adapters                       │
│        (50M trainable parameters)                 │
│              TRAINABLE 🔥                         │
│                                                   │
│  ┌──────────┐ ┌──────────┐ ┌───────────────┐    │
│  │ qkv_proj │ │  o_proj  │ │ gate_up_proj  │    │
│  │   LoRA   │ │   LoRA   │ │     LoRA      │    │
│  └──────────┘ └──────────┘ └───────────────┘    │
│                             ┌───────────────┐    │
│                             │  down_proj    │    │
│                             │     LoRA      │    │
│                             └───────────────┘    │
└──────────────────────────────────────────────────┘
```

### QLoRA Configuration

```python
# Quantization
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=float16,
    bnb_4bit_use_double_quant=True
)

# LoRA Adapters
LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### VRAM Budget (RTX 3060 — 6.4GB available)

```
Base model (4-bit):     ~2.0 GB
LoRA adapters:          ~0.1 GB
Optimizer states:       ~0.3 GB
Activations + cache:    ~3.0 GB
─────────────────────────────
Total:                  ~5.4 GB ✅ (headroom: 1.0 GB)
```

### GPT-2 LoRA (Legacy v1)

```
┌─────────────────────────────────────────────────┐
│                  GPT-2 Medium                    │
│               (355M parameters)                  │
│                   FROZEN ❄️                      │
├─────────────────────────────────────────────────┤
│              LoRA Adapters                       │
│           (1.5M parameters)                      │
│              TRAINABLE 🔥                        │
│                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│  │ c_attn  │    │ c_proj  │    │  c_fc   │     │
│  │  LoRA   │    │  LoRA   │    │  LoRA   │     │
│  └─────────┘    └─────────┘    └─────────┘     │
└─────────────────────────────────────────────────┘
```

---

## 📊 Training Results

### Phi-3-mini QLoRA

| Metric | Value |
|--------|-------|
| Training Epochs | 3 |
| Training Steps | 24 |
| Initial Loss | 1.24 |
| Final Loss | 0.88 |
| Training Time | 4 min 58 sec |
| VRAM Used | 2.87 GB allocated |
| Optimizer | paged_adamw_8bit |
| Hardware | RTX 3060 Laptop GPU (6.4 GB) |

### GPT-2 LoRA (Legacy)

| Metric | Value |
|--------|-------|
| Training Epochs | 5 |
| Final Loss | ~0.9 |
| Training Time | ~45 min |
| Learning Rate | 2e-4 |

---

## 📋 Supported Document Types

| Category | Examples |
|----------|----------|
| **Contracts** | Employment agreements, confidentiality clauses, termination clauses |
| **NDAs** | Mutual NDAs, unilateral NDAs, IP protection clauses |
| **Policies** | Privacy policies, data retention, cookie policies |
| **Correspondence** | Professional emails, deadline extensions, proposals |
| **Legal Documents** | Terms of Service, limitation of liability, disclaimers |
| **Business Documents** | Meeting minutes, executive summaries, recommendations |

---

## 📝 Example Outputs (Phi-3)

### Confidentiality Clause
**Input:** `Write a confidentiality clause for a software development contract`

**Output:**
> CONFIDENTIALITY CLAUSE
>
> This Agreement, effective as of [Effective Date]... all non-public information concerning Software Development Project(s) shall constitute 'Confidential Information'. The Contractor agrees to hold all Confidential Information in strict confidence and not disclose to any third party without prior written consent...

### Deadline Extension Email
**Input:** `Write a professional email requesting deadline extension`

**Output:**
> Subject: Request for Extension on Project Deadline
>
> Dear Mr./Ms. [Client's Last Name],
>
> I hope this message finds you well. Due to unforeseen technical challenges encountered during development, it has become necessary to extend the proposed completion date by two weeks...

---

## 🖥️ Local Development

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (RTX 3060+ recommended for Phi-3)
- 8GB+ VRAM for QLoRA training

### Installation

```bash
git clone https://github.com/sebikradyel1-svg/Advanced-AI-Engineering-Portfolio.git
cd Advanced-AI-Engineering-Portfolio/legal-text-generator

pip install -r requirements_phi3.txt
```

### Training Phi-3 QLoRA

```bash
# Quick test (15-20 min)
python training/train_phi3.py --preset quick

# Quality training (1-2 hours) — RECOMMENDED
python training/train_phi3.py --preset quality

# Test a trained model
python training/train_phi3.py --test_only --adapter_path ./phi3_legal_lora
```

### Training GPT-2 LoRA (Legacy)

```bash
python training/english_legal_llm_finetuning.py --preset quality
```

### Running the Web App

```bash
# GPT-2 version (deployed on HuggingFace)
python app/app.py

# Phi-3 version (local, requires GPU)
python app/app_phi3.py
```

---

## 🏗️ Architecture Decision Records

| ADR | Decision | Key Trade-off |
|-----|----------|---------------|
| [ADR-004](docs/adr/ADR-004-phi3-migration.md) | Migrate GPT-2 → Phi-3-mini QLoRA | 10x quality vs. 3x inference latency |

### Alternatives Considered (ADR-004)

| Option | Parameters | VRAM | Risk | Decision |
|--------|-----------|------|------|----------|
| **Phi-3-mini** | 3.8B | 6-7 GB | Low | ✅ Selected |
| Mistral-7B | 7.2B | 10-12 GB | OOM risk | ❌ Too large |
| Llama-3.2-3B | 3.2B | 5-6 GB | OK | ❌ Less instruction-tuned |
| Keep GPT-2 | 355M | 4 GB | None | ❌ Quality ceiling |

---

## 📁 Project Structure

```
legal-text-generator/
├── training/
│   ├── train_phi3.py                       # QLoRA training (Phi-3)
│   ├── english_legal_llm_finetuning.py     # LoRA training (GPT-2)
│   └── phi3_legal_lora/                    # Adapter weights & config
│       ├── adapter_config.json
│       ├── adapter_model.safetensors       # 201 MB
│       └── training_metadata.json
├── app/
│   ├── app.py                              # GPT-2 Gradio app (deployed)
│   └── app_phi3.py                         # Phi-3 inference app (local)
├── docs/
│   └── adr/
│       └── ADR-004-phi3-migration.md       # Migration decision record
├── requirements.txt                         # GPT-2 dependencies
├── requirements_phi3.txt                    # Phi-3 QLoRA dependencies
└── README.md
```

---

## 🔧 Requirements

### Phi-3 QLoRA (requirements_phi3.txt)
```
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
accelerate>=0.28.0
bitsandbytes>=0.43.0
datasets>=2.18.0
tensorboard>=2.16.0
gradio>=4.0.0
```

### GPT-2 LoRA (requirements.txt)
```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
accelerate>=0.24.0
gradio>=4.0.0
```

---

## 🎓 Skills Demonstrated

- **QLoRA Fine-tuning** — 4-bit NF4 quantization with double quantization
- **PEFT/LoRA** — Parameter-efficient training (0.4% → 2.44%)
- **Model Migration** — Documented upgrade path with ADR
- **VRAM Optimization** — Budget planning for consumer GPU constraints
- **HuggingFace Hub** — Model weight hosting and versioning
- **Gradio Deployment** — Interactive ML demo on HuggingFace Spaces
- **Architecture Decisions** — Trade-off analysis with alternatives comparison

---

## 👤 Author

**Paul Sebastian Kradyel** — AI Engineer & ML Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/paul-sebastian-kradyel)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/sebikradyel1-svg)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Profile-yellow)](https://huggingface.co/KradyelSebi)

---

## ⚠️ Disclaimer

Generated content is for educational and demonstration purposes. Review by qualified legal professionals is required before use in actual legal documents.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

