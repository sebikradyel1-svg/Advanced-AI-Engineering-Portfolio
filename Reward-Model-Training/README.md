# ⭐ Reward Model Training

> Training Reward Models for Human Preference Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Latest-yellow)](https://huggingface.co/transformers)

## 📋 Overview

Implementation of reward model training for RLHF (Reinforcement Learning from Human Feedback). The reward model learns to predict human preferences from pairwise comparisons, enabling alignment of language models with human values.

## 🎯 Features

- **Pairwise preference learning** - Learns from human preference data
- **Bradley-Terry model** - Probabilistic ranking framework
- **Integration with RLHF pipeline** - Seamless connection to PPO training
- **Custom loss functions** - Optimized for ranking tasks
- **Efficient training** - Works on consumer GPUs

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch |
| Models | HuggingFace Transformers |
| Base Architecture | GPT-2 / BERT |
| Loss Function | Bradley-Terry / Cross-Entropy |

## 📁 Project Structure

```
Reward-Model-Training/
├── reward_model_training.py   # Main training script
├── dataset.py                 # Preference dataset loader
├── model.py                   # Reward model architecture
└── README.md
```

## 🚀 Quick Start

```python
from reward_model_training import RewardModelTrainer

# Initialize trainer
trainer = RewardModelTrainer(
    base_model="gpt2",
    learning_rate=1e-5
)

# Train on preference data
trainer.train(
    preference_dataset="path/to/preferences.json",
    epochs=3
)

# Save trained model
trainer.save("reward_model_checkpoint")
```

## 📊 Model Architecture

```
Input Text
    │
    ▼
┌─────────────────┐
│  Base LLM       │
│  (GPT-2/BERT)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reward Head    │
│  (Linear → 1)   │
└────────┬────────┘
         │
         ▼
   Scalar Reward
```

## 🔧 Training Process

### Bradley-Terry Loss
```python
def bradley_terry_loss(chosen_rewards, rejected_rewards):
    """
    Compute loss based on preference pairs
    P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    """
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
```

### Preference Data Format
```json
{
    "prompt": "Write a helpful response about...",
    "chosen": "Here's a detailed and accurate answer...",
    "rejected": "I don't know, figure it out yourself..."
}
```

## 📈 Results

| Metric | Score |
|--------|-------|
| Preference Accuracy | 78% |
| Validation Loss | 0.45 |
| Ranking Correlation | 0.82 |

## 🔑 Key Components

### Reward Model Class
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model)
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask)
        reward = self.reward_head(outputs.last_hidden_state[:, -1])
        return reward
```

## 📝 Key Learnings

- Preference data quality is crucial for good reward models
- Bradley-Terry provides interpretable probability scores
- Regularization prevents reward model overfitting
- Reward models should be periodically retrained

## 📚 References

- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Bradley-Terry Model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)

## 📫 Contact

- **LinkedIn:** [linkedin.com/in/paul-sebastian-kradyel](https://linkedin.com/in/paul-sebastian-kradyel)
- **Email:** paulsebastiankradyel@gmail.com
- **GitHub:** [github.com/sebikradyel1-svg](https://github.com/sebikradyel1-svg)
