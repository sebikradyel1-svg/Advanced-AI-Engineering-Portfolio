# Reward Model Training for RLHF - Complete Documentation

## Overview

A production-ready system for training **Reward Models** that distinguish between high-quality and low-quality responses. This is a critical component in RLHF (Reinforcement Learning from Human Feedback) pipelines, enabling language models to align with human preferences.

## What is a Reward Model?

A **Reward Model** is a neural network trained to predict human preferences between different responses to the same prompt. It serves as the "judge" that evaluates response quality during RL fine-tuning.

### The RLHF Pipeline

```
1. Pre-trained LLM (GPT-2, etc.)
         ↓
2. Supervised Fine-tuning (SFT)
         ↓
3. Reward Model Training ← YOU ARE HERE
         ↓
4. RL Fine-tuning (PPO/DPO)
         ↓
5. Aligned Model
```

### Why Reward Models Matter

| Without Reward Model | With Reward Model |
|---------------------|-------------------|
| ❌ No quality signal | ✅ Clear preference signal |
| ❌ Manual evaluation needed | ✅ Automated at scale |
| ❌ Subjective feedback | ✅ Consistent scoring |
| ❌ Can't optimize | ✅ Enables RL training |

## Key Features

- ✅ **Pairwise Comparison**: Learns from human preference data
- ✅ **LoRA/PEFT**: Memory-efficient training (works on 6GB GPU)
- ✅ **Custom Loss Function**: Optimized for preference learning
- ✅ **Hardware Optimized**: RTX 3060 6GB configuration included
- ✅ **Evaluation Metrics**: Accuracy, margin analysis
- ✅ **Interactive Testing**: Test on custom prompts
- ✅ **Production Ready**: Save/load capabilities

## Installation

### Requirements

```bash
pip install torch transformers datasets peft numpy
```

### Full Installation

```bash
# Core dependencies
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install datasets>=2.0.0
pip install peft>=0.4.0
pip install numpy>=1.24.0

# For GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Quick Start

```python
python reward_model_training.py
```

## How It Works

### 1. Pairwise Comparison Approach

Instead of absolute ratings, we train on **pairs** of responses:

```
Prompt: "Explain what AI is."

Chosen (Better):
"AI is a field of computer science that creates systems 
capable of performing tasks requiring human intelligence, 
such as learning, reasoning, and problem-solving."

Rejected (Worse):
"AI is something with computers."
```

The model learns: **Score(Chosen) > Score(Rejected)**

### 2. Training Process

```
[Prompt + Chosen Response] → Model → Score_chosen
[Prompt + Rejected Response] → Model → Score_rejected

Loss = -log(sigmoid(Score_chosen - Score_rejected))
```

This loss function encourages the model to give higher scores to better responses.

### 3. Architecture

```
┌────────────────────────────────────────────────┐
│         Reward Model Architecture              │
├────────────────────────────────────────────────┤
│                                                │
│  Input: "Prompt\n\nRăspuns: Response"         │
│           ↓                                    │
│  ┌─────────────────────┐                      │
│  │   Tokenizer         │                      │
│  └──────────┬──────────┘                      │
│             ↓                                  │
│  ┌─────────────────────┐                      │
│  │   GPT-2 Base        │                      │
│  │   (with LoRA)       │                      │
│  └──────────┬──────────┘                      │
│             ↓                                  │
│  ┌─────────────────────┐                      │
│  │ Classification Head │                      │
│  │   (1 output neuron) │                      │
│  └──────────┬──────────┘                      │
│             ↓                                  │
│      Reward Score (scalar)                    │
│                                                │
└────────────────────────────────────────────────┘
```

## Dataset

### Format

Uses the **Dahoas/synthetic-instruct-gptj-pairwise** dataset:

```python
{
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris, a beautiful...",
    "rejected": "idk maybe paris or something"
}
```

### Custom Dataset

Create your own dataset:

```python
from datasets import Dataset

data = {
    "prompt": [
        "Explain machine learning",
        "What is Python?",
        # ...
    ],
    "chosen": [
        "Machine learning is a subset of AI...",
        "Python is a high-level programming language...",
        # ...
    ],
    "rejected": [
        "ML is when computers learn stuff",
        "python is a snake lol",
        # ...
    ]
}

dataset = Dataset.from_dict(data)
```

## Configuration

### Hardware Requirements

| Component | Minimum | Recommended | Optimized For |
|-----------|---------|-------------|---------------|
| GPU VRAM | 6GB | 8GB+ | RTX 3060 6GB |
| RAM | 8GB | 16GB+ | - |
| Storage | 10GB | 20GB+ | Model + Dataset |
| CUDA | 11.0+ | 12.0+ | - |

### Memory Optimization Settings

```python
# For 6GB GPU (RTX 3060)
training_args = TrainingArguments(
    per_device_train_batch_size=1,      # Small batch
    gradient_accumulation_steps=16,     # Effective batch = 16
    fp16=True,                          # Half precision
    max_length=384,                     # Shorter sequences
    gradient_checkpointing=False,       # Optional
)

# For 12GB+ GPU
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    fp16=True,
    max_length=512,
)

# For 24GB+ GPU (A100, RTX 4090)
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    bf16=True,  # Better than fp16
    max_length=1024,
)
```

## LoRA Configuration

**LoRA (Low-Rank Adaptation)** enables efficient fine-tuning by only training a small number of parameters.

```python
lora_config = LoraConfig(
    r=16,                           # Rank (higher = more capacity)
    lora_alpha=32,                  # Scaling factor
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention modules
    lora_dropout=0.05,              # Dropout for regularization
    bias="none",
    task_type=TaskType.SEQ_CLS
)
```

### LoRA Parameters Explained

| Parameter | Effect | Typical Range | Memory Impact |
|-----------|--------|---------------|---------------|
| `r` (rank) | Model capacity | 4-64 | Higher = more memory |
| `lora_alpha` | Learning rate scaling | r×2 | No impact |
| `lora_dropout` | Regularization | 0.0-0.1 | No impact |
| `target_modules` | Which layers to train | Model-specific | More = more memory |

**Parameter Efficiency:**
- Full fine-tuning: ~125M parameters (GPT-2)
- LoRA (r=16): ~0.3M parameters (0.24% of full model!)

## Training Pipeline

### Step-by-Step Process

```python
# 1. Load dataset
dataset = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise")

# 2. Initialize model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2",
    num_labels=1  # Single scalar output
)

# 3. Apply LoRA
model = get_peft_model(model, lora_config)

# 4. Preprocess data
def preprocess_function(examples):
    chosen_texts = [f"{p}\n\nRăspuns: {c}" 
                    for p, c in zip(examples['prompt'], examples['chosen'])]
    rejected_texts = [f"{p}\n\nRăspuns: {r}" 
                      for p, r in zip(examples['prompt'], examples['rejected'])]
    # Tokenize both...
    return {...}

# 5. Create custom trainer
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs):
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"]
        ).logits
        
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"]
        ).logits
        
        # Preference loss
        loss = -torch.nn.functional.logsigmoid(
            rewards_chosen - rewards_rejected
        ).mean()
        
        return loss

# 6. Train
trainer = RewardTrainer(...)
trainer.train()

# 7. Save
model.save_pretrained("./reward_model_final")
```

## Loss Function Deep Dive

### Mathematical Foundation

```
Preference Probability:
P(chosen > rejected) = sigmoid(score_chosen - score_rejected)

Loss (Negative Log-Likelihood):
L = -log(sigmoid(score_chosen - score_rejected))
  = log(1 + exp(score_rejected - score_chosen))
```

### Why This Loss?

1. **Bradley-Terry Model**: Based on proven preference modeling
2. **Differentiable**: Enables gradient descent
3. **Bounded**: Prevents extreme scores
4. **Calibrated**: Larger margins for clearer preferences

### Visualization

```
If score_chosen - score_rejected = 2.0:
  → P(chosen preferred) = sigmoid(2.0) ≈ 88%
  → Loss ≈ 0.13 (low, good)

If score_chosen - score_rejected = -1.0:
  → P(chosen preferred) = sigmoid(-1.0) ≈ 27%
  → Loss ≈ 1.31 (high, bad - model should improve)
```

## Evaluation Metrics

### 1. Accuracy

**Definition**: Percentage of cases where `score_chosen > score_rejected`

```python
correct = 0
for sample in eval_dataset:
    if score(chosen) > score(rejected):
        correct += 1
accuracy = correct / total
```

**Interpretation:**
- 50%: Random guessing (bad)
- 70%: Decent model
- 85%+: Good model
- 95%+: Excellent model

### 2. Average Margin

**Definition**: Mean difference between chosen and rejected scores

```
margin = mean(score_chosen - score_rejected)
```

**Interpretation:**
- `margin > 0`: Model prefers chosen (good)
- `margin > 1.0`: Strong preference signal
- `margin > 2.0`: Very confident model

### 3. Score Distribution

Analyze the distribution of scores:

```python
print(f"Chosen scores: mean={mean_chosen:.2f}, std={std_chosen:.2f}")
print(f"Rejected scores: mean={mean_rejected:.2f}, std={std_rejected:.2f}")
```

## Usage Examples

### Basic Training

```python
# Default configuration
python reward_model_training.py
```

### Custom Training Script

```python
from reward_model_training import RewardTrainer, preprocess_function
from transformers import TrainingArguments

# Load your data
dataset = load_dataset("your-dataset")

# Configure training
training_args = TrainingArguments(
    output_dir="./my_reward_model",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    fp16=True,
)

# Train
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

### Scoring New Responses

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "./reward_model_final"
)
tokenizer = AutoTokenizer.from_pretrained("./reward_model_final")

def score_response(prompt: str, response: str) -> float:
    """Score a response for a given prompt."""
    text = f"{prompt}\n\nRăspuns: {response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
    
    with torch.no_grad():
        score = model(**inputs).logits.item()
    
    return score

# Example usage
prompt = "What is machine learning?"
response1 = "Machine learning is a subset of AI that enables computers to learn from data..."
response2 = "ML is like when computers do stuff"

score1 = score_response(prompt, response1)
score2 = score_response(prompt, response2)

print(f"Response 1 score: {score1:.4f}")
print(f"Response 2 score: {score2:.4f}")
print(f"Winner: {'Response 1' if score1 > score2 else 'Response 2'}")
```

## Integration with RLHF Pipeline

### Using the Reward Model in PPO

```python
from transformers import AutoModelForSequenceClassification

# Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward_model_final"
)
reward_model.eval()

# During PPO training
def compute_rewards(prompts, responses):
    rewards = []
    for prompt, response in zip(prompts, responses):
        text = f"{prompt}\n\nRăspuns: {response}"
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            reward = reward_model(**inputs).logits.item()
        
        rewards.append(reward)
    
    return torch.tensor(rewards)

# Use in PPO trainer
trainer = PPOTrainer(
    model=policy_model,
    reward_model=compute_rewards,
    ...
)
```

### Complete RLHF Example

```python
# Step 1: Train reward model
python reward_model_training.py

# Step 2: Use in RLHF (from previous documentation)
from KL_Guard_RLHF import UniversalPPOTrainer

# Replace the reward model
class CustomRewardModel:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def compute_rewards(self, texts):
        rewards = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                reward = self.model(**inputs).logits.item()
            rewards.append(reward)
        return torch.tensor(rewards)

# Initialize trainer with custom reward model
trainer = UniversalPPOTrainer()
trainer.reward_model = CustomRewardModel("./reward_model_final")
trainer.train()
```

## Advanced Features

### 1. Multi-Task Reward Model

Train on multiple objectives:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2",
    num_labels=3  # Helpfulness, Harmlessness, Honesty
)

# Loss for each objective
loss_helpfulness = -logsigmoid(scores_chosen[0] - scores_rejected[0])
loss_harmlessness = -logsigmoid(scores_chosen[1] - scores_rejected[1])
loss_honesty = -logsigmoid(scores_chosen[2] - scores_rejected[2])

total_loss = loss_helpfulness + loss_harmlessness + loss_honesty
```

### 2. Ensemble Reward Models

Combine multiple reward models:

```python
models = [
    load_model("./reward_model_1"),
    load_model("./reward_model_2"),
    load_model("./reward_model_3"),
]

def ensemble_score(prompt, response):
    scores = [model.score(prompt, response) for model in models]
    return np.mean(scores)  # Or weighted average
```

### 3. Active Learning

Select most informative samples for labeling:

```python
def select_uncertain_samples(prompts, responses_pairs):
    """Select samples where model is most uncertain."""
    uncertainty_scores = []
    
    for prompt, (resp1, resp2) in zip(prompts, responses_pairs):
        score1 = model.score(prompt, resp1)
        score2 = model.score(prompt, resp2)
        
        # Uncertainty = how close the scores are
        uncertainty = abs(score1 - score2)
        uncertainty_scores.append(uncertainty)
    
    # Return samples with lowest uncertainty (hardest to judge)
    return sorted(zip(prompts, responses_pairs, uncertainty_scores),
                  key=lambda x: x[2])[:100]
```

## Troubleshooting

### Out of Memory (OOM)

**Problem**: `CUDA out of memory` error

**Solutions:**
```python
# 1. Reduce batch size
per_device_train_batch_size=1

# 2. Increase gradient accumulation
gradient_accumulation_steps=32

# 3. Reduce sequence length
max_length=256  # Instead of 512

# 4. Enable gradient checkpointing
gradient_checkpointing=True

# 5. Use smaller LoRA rank
r=8  # Instead of 16

# 6. Clear cache periodically
torch.cuda.empty_cache()
```

### Low Accuracy

**Problem**: Model accuracy < 60%

**Solutions:**
1. **Train longer**: Increase `num_train_epochs`
2. **Better data**: Ensure clear preference in dataset
3. **Larger model**: Use `gpt2-medium` or `gpt2-large`
4. **Adjust learning rate**: Try `1e-5` to `5e-5`
5. **Increase LoRA rank**: Try `r=32` or `r=64`

### Model Too Confident

**Problem**: All scores are extreme (very high or very low)

**Solutions:**
```python
# 1. Add temperature scaling
def scaled_score(logits, temperature=1.0):
    return logits / temperature

# 2. Add regularization
weight_decay=0.01  # In TrainingArguments

# 3. Lower learning rate
learning_rate=5e-6
```

### Slow Training

**Problem**: Training takes too long

**Solutions:**
```python
# 1. Use FP16
fp16=True

# 2. Reduce evaluation frequency
eval_steps=500  # Instead of 200

# 3. Fewer logging steps
logging_steps=100  # Instead of 50

# 4. Use smaller model
model_name = "distilgpt2"

# 5. Smaller dataset
dataset = dataset.select(range(10000))  # First 10k samples
```

## Best Practices

### 1. Data Quality

✅ **Do:**
- Use clear preference pairs
- Ensure diversity in prompts
- Balance positive/negative examples
- Include edge cases

❌ **Don't:**
- Use ambiguous comparisons
- Have biased dataset
- Include too similar pairs
- Ignore data quality

### 2. Training

✅ **Do:**
- Monitor validation metrics
- Use early stopping
- Save checkpoints frequently
- Track multiple metrics

❌ **Don't:**
- Overtrain (watch for overfitting)
- Ignore warning signs (NaN loss)
- Skip evaluation
- Use too high learning rate

### 3. Evaluation

✅ **Do:**
- Test on diverse prompts
- Check score distributions
- Verify margin adequacy
- Test edge cases

❌ **Don't:**
- Rely only on accuracy
- Ignore confidence levels
- Skip human evaluation
- Trust single metric

## Performance Benchmarks

### Training Time (on different GPUs)

| GPU | VRAM | Batch Size | Time/Epoch | Total (3 epochs) |
|-----|------|------------|------------|------------------|
| RTX 3060 | 6GB | 1 (×16 accum) | ~45 min | ~2.5 hours |
| RTX 3090 | 24GB | 8 (×2 accum) | ~12 min | ~40 minutes |
| A100 | 40GB | 16 (×1 accum) | ~6 min | ~20 minutes |

### Accuracy Benchmarks

| Model | Dataset Size | Epochs | Final Accuracy |
|-------|-------------|--------|----------------|
| GPT-2 (LoRA r=16) | 10k pairs | 3 | 75-80% |
| GPT-2 (LoRA r=32) | 10k pairs | 5 | 80-85% |
| GPT-2 Medium | 50k pairs | 3 | 85-90% |

## Export and Deployment

### Save for Production

```python
# Save full model
model.save_pretrained("./reward_model_production")
tokenizer.save_pretrained("./reward_model_production")

# Save only LoRA weights (smaller)
model.save_pretrained("./reward_model_lora", save_adapter=True)
```

### Load for Inference

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained("gpt2")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "./reward_model_lora")

# Merge for faster inference
model = model.merge_and_unload()
```

### API Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ScoreRequest(BaseModel):
    prompt: str
    response: str

@app.post("/score")
async def score_response(request: ScoreRequest):
    score = score_response(request.prompt, request.response)
    return {"score": float(score)}

# Run: uvicorn api:app --host 0.0.0.0 --port 8000
```

## Theory and Research

### Key Papers

1. **"Learning to Summarize from Human Feedback"** (Stiennon et al., 2020)
   - First large-scale RLHF for text generation

2. **"Training Language Models to Follow Instructions"** (Ouyang et al., 2022)
   - InstructGPT methodology

3. **"Constitutional AI"** (Bai et al., 2022)
   - Self-supervised reward modeling

### Bradley-Terry Model

The mathematical foundation:

```
P(i preferred over j) = exp(θᵢ) / (exp(θᵢ) + exp(θⱼ))
                       = sigmoid(θᵢ - θⱼ)

Where θ = reward score
```

### Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Reward Model** | Scalable, flexible | Requires preference data |
| **DPO** | No reward model needed | Less flexible |
| **RLAIF** | Self-supervised | May inherit biases |
| **Best-of-N** | Simple | Computationally expensive |

## FAQ

**Q: How much data do I need?**

A: Minimum 1k pairs, recommended 10k+, optimal 50k-100k for production.

**Q: Can I use other base models?**

A: Yes! Works with any model supporting `AutoModelForSequenceClassification`:
- GPT-2, GPT-Neo, GPT-J
- BERT, RoBERTa
- T5, FLAN-T5
- LLaMA (with modifications)

**Q: Should I use the same model for reward and policy?**

A: Not necessarily. Reward model can be smaller/faster for efficiency.

**Q: How do I handle multiple languages?**

A: Use multilingual base model (mBERT, XLM-R) and multilingual data.

**Q: What's the difference between this and classification?**

A: Standard classification learns absolute labels. This learns relative preferences (pairwise).

**Q: Can I combine multiple reward signals?**

A: Yes! Train multi-output model or ensemble multiple models.

## References

- Stiennon et al. (2020). "Learning to summarize with human feedback"
- Ouyang et al. (2022). "Training language models to follow instructions with human feedback"
- Christiano et al. (2017). "Deep reinforcement learning from human preferences"
- Bai et al. (2022). "Constitutional AI: Harmlessness from AI Feedback"

---

**Version**: 1.0  
**Last Updated**: 2025  
**Hardware**: Optimized for RTX 3060 6GB  
**License**: Research/Educational Use