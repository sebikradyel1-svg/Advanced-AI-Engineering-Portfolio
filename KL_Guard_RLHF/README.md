# 🎯 RLHF with KL-Divergence Guard

> Reinforcement Learning from Human Feedback with Safety Constraints

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![TRL](https://img.shields.io/badge/TRL-Latest-green.svg)](https://github.com/huggingface/trl)

## 📋 Overview

Implementation of Reinforcement Learning from Human Feedback (RLHF) with a custom **KL-divergence guard** to ensure model alignment stays safe and stable. This prevents reward hacking and catastrophic forgetting during policy optimization.

## 🎯 Features

- **Custom KL-divergence guard** - Maintains distribution stability during training
- **PPO algorithm implementation** - Full Proximal Policy Optimization loop
- **Reward hacking prevention** - Constraints prevent model from gaming the reward
- **Catastrophic forgetting protection** - Preserves base model capabilities
- **Configurable KL penalty coefficients** - Fine-tune alignment strength
- **Memory-efficient training** - Gradient checkpointing for consumer GPUs

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch |
| RL Library | TRL (Transformer Reinforcement Learning) |
| Algorithm | PPO (Proximal Policy Optimization) |
| Base Model | GPT-2 / LLaMA |
| Hardware | RTX 3060 12GB VRAM |

## 📁 Project Structure

```
KL_Guard_RLHF/
├── KL_Guard_RLHF.py          # Main RLHF training script
├── reward_model.py           # Reward model integration
├── config.py                 # Training configuration
└── README.md
```

## 🚀 Quick Start

```python
from KL_Guard_RLHF import RLHFTrainer

# Initialize trainer with KL guard
trainer = RLHFTrainer(
    model_name="gpt2",
    kl_coef=0.1,           # KL penalty coefficient
    max_kl_divergence=0.2  # Maximum allowed KL divergence
)

# Train with RLHF
trainer.train(
    reward_model=reward_model,
    num_steps=10000
)
```

## 📊 RLHF Pipeline Architecture

```
┌─────────────────┐
│   Base Model    │
│    (Policy)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Generate       │────▶│  Reward Model   │
│  Response       │     │  (Score)        │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│           PPO Update                     │
│  ┌─────────────────────────────────┐    │
│  │     KL-Divergence Guard         │    │
│  │  (Prevents distribution drift)  │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## 🔧 Configuration

```python
config = {
    "kl_coef": 0.1,              # KL penalty weight
    "max_kl_divergence": 0.2,    # Clip threshold
    "ppo_epochs": 4,             # PPO update epochs
    "batch_size": 8,             # Training batch size
    "learning_rate": 1e-5,       # Learning rate
    "gradient_checkpointing": True  # Memory optimization
}
```

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Steps | 10,000+ |
| KL Divergence | < 0.2 (maintained) |
| Model Coherence | ✅ Preserved |
| Reward Improvement | +15% |

## 🔑 Key Components

### KL-Divergence Guard
```python
def kl_guard(self, new_logprobs, old_logprobs):
    """Compute KL divergence and apply penalty"""
    kl_div = (old_logprobs - new_logprobs).mean()
    
    if kl_div > self.max_kl_divergence:
        # Apply stronger penalty
        return kl_div * self.kl_coef * 2
    return kl_div * self.kl_coef
```

## 📝 Key Learnings

- KL divergence is crucial for stable RLHF training
- Gradient checkpointing enables training on consumer GPUs
- Reward model quality directly impacts alignment
- PPO with clipping prevents catastrophic policy updates

## 📚 References

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [TRL Library](https://github.com/huggingface/trl)

## 📫 Contact

- **LinkedIn:** [linkedin.com/in/paul-sebastian-kradyel](https://linkedin.com/in/paul-sebastian-kradyel)
- **Email:** paulsebastiankradyel@gmail.com
- **GitHub:** [github.com/sebikradyel1-svg](https://github.com/sebikradyel1-svg)
