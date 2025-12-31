# RLHF with PPO - Universal Implementation with KL Divergence Control

## Overview

A robust, universal implementation of Reinforcement Learning from Human Feedback (RLHF) using Proximal Policy Optimization (PPO) with adaptive KL divergence control. This implementation works with any version of TRL (Transformer Reinforcement Learning) library and includes a manual implementation for maximum stability and compatibility.

## What is RLHF?

**Reinforcement Learning from Human Feedback (RLHF)** is a technique to fine-tune language models based on human preferences:

1. **Base Model**: Start with a pre-trained language model (e.g., GPT-2)
2. **Reward Model**: Train a model that predicts human preferences
3. **RL Fine-tuning**: Use PPO to optimize the model based on rewards
4. **KL Constraint**: Keep the model close to the original to prevent degradation

## Key Features

- ✅ **Universal Compatibility**: Works with any TRL version or standalone
- ✅ **Manual Implementation**: Complete PPO implementation without TRL dependencies
- ✅ **KL Divergence Control**: Adaptive coefficient adjustment
- ✅ **Robust Reward Model**: Sentiment-based with heuristic fallback
- ✅ **Automatic Fallback**: Handles missing dependencies gracefully
- ✅ **Educational**: Detailed explanations and visualizations
- ✅ **Production-Ready**: Error handling and logging throughout

## Why KL Divergence Matters

### The Problem Without KL Constraint

Without KL divergence control, the model can:

1. **Policy Collapse**: Generate repetitive, meaningless text
2. **Reward Hacking**: Find shortcuts to maximize rewards artificially
3. **Catastrophic Forgetting**: Lose language capabilities from pre-training

### The Solution

```
Loss = Policy_Loss - β * KL(π || π_ref)
```

Where:
- `π` = Current policy (trained model)
- `π_ref` = Reference policy (frozen original model)
- `β` = KL coefficient (adaptive)

**Adaptive Control:**
- If KL > target × 1.5 → Increase β (more constraint)
- If KL < target / 1.5 → Decrease β (less constraint)

### KL Values Interpretation

| KL Value | Meaning | Action |
|----------|---------|--------|
| 0.001 | Too conservative | Relax constraint |
| 0.01 | ✅ Optimal sweet spot | Maintain |
| 0.1 | Risk of instability | Increase constraint |
| 1.0 | Too aggressive | Strong constraint needed |

## Installation

### Basic Requirements

```bash
pip install torch transformers datasets numpy
```

### Optional (for TRL support)

```bash
pip install trl
```

**Note:** The implementation works perfectly fine without TRL using the manual implementation.

### Full Installation

```bash
# Core dependencies
pip install torch>=1.10.0
pip install transformers>=4.20.0
pip install datasets>=2.0.0
pip install numpy>=1.20.0

# Optional
pip install trl>=0.4.0  # For TRL-based training
```

## Quick Start

### Basic Training

```python
from KL_Guard_RLHF import UniversalPPOTrainer

# Initialize trainer
trainer = UniversalPPOTrainer()

# Train
trainer.train(num_steps=50)

# Evaluate
test_prompts = [
    "The movie was",
    "This restaurant is",
    "My experience was"
]
trainer.evaluate(test_prompts)
```

### Command Line Usage

```bash
python KL_Guard_RLHF.py
```

## Configuration

### UniversalConfig Class

```python
class UniversalConfig:
    MODEL_NAME = "gpt2"           # Base model
    LEARNING_RATE = 1e-5          # Learning rate
    BATCH_SIZE = 2                # Batch size
    MINI_BATCH_SIZE = 1           # Mini-batch for optimization
    
    KL_COEF = 0.2                 # Initial KL coefficient
    KL_TARGET = 0.01              # Target KL divergence
    EPSILON_CLIP = 0.2            # PPO clipping parameter
    VALUE_CLIP = 0.2              # Value function clipping
    
    MAX_LENGTH = 80               # Max generation length
    TEMPERATURE = 1.0             # Sampling temperature
    
    NUM_STEPS = 50                # Training steps
```

### Customizing Configuration

```python
from KL_Guard_RLHF import UniversalPPOTrainer, UniversalConfig

# Modify configuration
config = UniversalConfig()
config.MODEL_NAME = "distilgpt2"  # Smaller model
config.LEARNING_RATE = 5e-6       # Lower learning rate
config.NUM_STEPS = 100            # More training steps
config.KL_TARGET = 0.02           # Higher KL target

# Initialize with custom config
trainer = UniversalPPOTrainer()
trainer.config = config
trainer.train()
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────┐
│         UniversalPPOTrainer                 │
├─────────────────────────────────────────────┤
│                                             │
│  ┌────────────────┐  ┌──────────────────┐  │
│  │  Active Model  │  │  Reference Model │  │
│  │  (Trainable)   │  │    (Frozen)      │  │
│  └────────────────┘  └──────────────────┘  │
│           │                   │             │
│           └───────┬───────────┘             │
│                   │                         │
│         ┌─────────▼──────────┐              │
│         │ KL Controller      │              │
│         │ (Adaptive β)       │              │
│         └────────────────────┘              │
│                   │                         │
│         ┌─────────▼──────────┐              │
│         │  Reward Model      │              │
│         │ (Sentiment/Heur.)  │              │
│         └────────────────────┘              │
│                                             │
└─────────────────────────────────────────────┘
```

### Training Pipeline

1. **Generate**: Model generates responses to prompts
2. **Evaluate**: Reward model scores responses
3. **Compute KL**: Calculate divergence from reference model
4. **Optimize**: Update model with PPO + KL penalty
5. **Adapt**: Adjust KL coefficient based on divergence
6. **Repeat**: Continue for specified steps

## Core Classes

### 1. KLDivergenceController

Manages KL divergence between the training and reference models.

```python
controller = KLDivergenceController(
    initial_coef=0.2,  # Starting coefficient
    target=0.01        # Target KL value
)

# Compute KL between two distributions
kl = controller.compute_kl(logits, ref_logits)

# Update coefficient adaptively
controller.update_coefficient(current_kl)

# Get penalty for loss function
penalty = controller.get_penalty(kl_value)
```

**Key Methods:**
- `compute_kl()`: Calculates KL divergence
- `update_coefficient()`: Adaptively adjusts β
- `get_penalty()`: Returns weighted KL penalty

### 2. RobustRewardModel

Provides rewards for generated text with multiple fallback strategies.

```python
reward_model = RobustRewardModel()

# Compute rewards for batch of texts
rewards = reward_model.compute_rewards([
    "This movie was amazing!",
    "Terrible experience."
])
# Returns: tensor([2.0, -0.5])
```

**Reward Strategies:**

1. **Primary**: DistilBERT sentiment analysis
   - Positive → +2.0 × confidence
   - Negative → -0.5 × confidence

2. **Fallback**: Heuristic word counting
   - Counts positive/negative words
   - Normalized by text length

### 3. UniversalPPOTrainer

Main training class with manual PPO implementation.

```python
trainer = UniversalPPOTrainer()

# Training step
stats = trainer.train_step(prompts)

# Full training
trainer.train(num_steps=50)

# Evaluation
trainer.evaluate(test_prompts)
```

## Training Process

### Step-by-Step Explanation

```python
def train_step(prompts):
    # 1. Generate responses
    responses = model.generate(prompts)
    
    # 2. Compute rewards
    rewards = reward_model.score(responses)
    
    # 3. Calculate KL divergence
    kl = compute_kl(model_logits, ref_model_logits)
    
    # 4. PPO update
    loss = -rewards.mean() + kl_coef * kl
    
    # 5. Backpropagation
    loss.backward()
    optimizer.step()
    
    # 6. Adapt KL coefficient
    kl_controller.update(kl)
```

### Training Output Example

```
🎯 ÎNCEPE ANTRENAMENTUL MANUAL
============================================================
📊 Configurație:
   • Model: gpt2
   • Pași: 50
   • Batch size: 2
   • KL Target: 0.01
   • Mod: MANUAL (fără TRL)
============================================================

📈 Pas 0/50
   • Reward: 0.156
   • KL Div: 0.0234
   • KL Coef: 0.200
   • Reward Loss: -0.156
   • Total Loss: 0.0047
   • Exemplu: The movie was incredibly entertaining...

📈 Pas 5/50
   • Reward: 0.312
   • KL Div: 0.0198
   • KL Coef: 0.220
   • Reward Loss: -0.312
   • Total Loss: 0.0044

...

✅ Antrenament manual completat!
```

## Advanced Usage

### Custom Reward Function

```python
class CustomRewardModel:
    def compute_rewards(self, texts: List[str]) -> torch.Tensor:
        rewards = []
        for text in texts:
            # Your custom logic
            score = your_scoring_function(text)
            rewards.append(score)
        return torch.tensor(rewards, dtype=torch.float32)

# Use custom reward
trainer = UniversalPPOTrainer()
trainer.reward_model = CustomRewardModel()
```

### Custom Training Loop

```python
trainer = UniversalPPOTrainer()

for epoch in range(num_epochs):
    for batch in data_loader:
        # Custom batch processing
        prompts = batch['text']
        
        # Train step
        stats = trainer.train_step(prompts)
        
        # Custom logging
        if step % log_interval == 0:
            print(f"Epoch {epoch}, Step {step}")
            print(f"Reward: {stats['mean_reward']:.3f}")
```

### Fine-tuning Pre-trained Model

```python
# Load your fine-tuned model
trainer = UniversalPPOTrainer()
trainer.model = AutoModelForCausalLM.from_pretrained("your-model")
trainer.ref_model = AutoModelForCausalLM.from_pretrained("your-model")

# Freeze reference
for param in trainer.ref_model.parameters():
    param.requires_grad = False

# Continue training
trainer.train(num_steps=100)
```

## Monitoring and Debugging

### Key Metrics to Watch

1. **Mean Reward**
   - Should gradually increase
   - Sudden drops indicate problems

2. **KL Divergence**
   - Should hover around target (0.01)
   - Too high → Model diverging too much
   - Too low → Model not learning enough

3. **KL Coefficient**
   - Adapts automatically
   - Oscillates around optimal value

4. **Losses**
   - Reward loss should decrease
   - Total loss should stabilize

### Debugging Common Issues

**Problem: Reward not increasing**
```python
# Check reward model
rewards = trainer.reward_model.compute_rewards(["test"])
print(rewards)  # Should be non-zero

# Increase learning rate
trainer.config.LEARNING_RATE = 1e-4
```

**Problem: High KL divergence**
```python
# Increase KL constraint
trainer.config.KL_COEF = 0.5
trainer.kl_controller.coef = 0.5
```

**Problem: Model collapse (repetitive text)**
```python
# Increase temperature
trainer.config.TEMPERATURE = 1.5

# Decrease KL target
trainer.config.KL_TARGET = 0.005
```

## Evaluation

### Testing Trained Model

```python
# After training
test_prompts = [
    "The movie was",
    "I think this product is",
    "My experience at the restaurant was",
    "The customer service was"
]

trainer.evaluate(test_prompts)
```

### Output Example

```
🧪 EVALUARE FINALĂ
============================================================

📝 Prompt 1: The movie was
   Răspuns: The movie was absolutely fantastic and I loved every moment
   Score: 1.850

📝 Prompt 2: I think this product is
   Răspuns: I think this product is excellent quality and great value
   Score: 1.623
```

## Best Practices

### 1. Start Small
```python
# Begin with few steps for testing
trainer.train(num_steps=10)

# Gradually increase
trainer.train(num_steps=50)
trainer.train(num_steps=100)
```

### 2. Monitor KL Carefully
```python
# Check KL history
print(f"KL values: {trainer.kl_controller.history}")
print(f"Final KL: {trainer.kl_controller.history[-1]:.4f}")
```

### 3. Save Checkpoints
```python
# Save model periodically
if step % 10 == 0:
    trainer.model.save_pretrained(f"checkpoint_{step}")
    trainer.tokenizer.save_pretrained(f"checkpoint_{step}")
```

### 4. Experiment with Prompts
```python
# Use diverse prompts
prompts = [
    "The movie was",
    "This book is",
    "I feel that",
    "The experience was",
    "In my opinion",
    "This product is",
    "My thoughts on",
    "Overall, it was"
]
```

## Understanding the Math

### PPO Loss Function

```
L_PPO = L_CLIP + c₁L_VF - c₂S[π]

Where:
- L_CLIP: Clipped policy loss
- L_VF: Value function loss
- S: Entropy bonus
- c₁, c₂: Coefficients
```

### KL Divergence Formula

```
KL(P||Q) = Σ P(x) log(P(x)/Q(x))

In our case:
- P = π(a|s) - Current policy
- Q = π_ref(a|s) - Reference policy
```

### Combined Objective

```
L_total = -E[rewards] + β * KL(π||π_ref)

Goal: Maximize rewards while staying close to reference
```

## Troubleshooting

### Import Errors

```bash
# TRL not found - This is OK!
⚠️ TRL nu este instalat sau incompatibil
# System automatically uses manual implementation
```

### CUDA Out of Memory

```python
# Reduce batch size
config.BATCH_SIZE = 1

# Reduce sequence length
config.MAX_LENGTH = 40

# Use CPU
# System automatically defaults to CPU
```

### Poor Quality Generations

```python
# Adjust temperature
config.TEMPERATURE = 0.8  # Lower = more focused

# Increase training steps
trainer.train(num_steps=100)

# Check reward model
rewards = trainer.reward_model.compute_rewards(["test"])
```

## Extensions

### Add Custom Callbacks

```python
class TrainingCallback:
    def on_step_begin(self, step):
        print(f"Starting step {step}")
    
    def on_step_end(self, step, stats):
        if stats['mean_reward'] > 1.0:
            print("🎉 High reward achieved!")

# Use callback
callback = TrainingCallback()
# Integrate into training loop
```

### Multi-objective Rewards

```python
class MultiObjectiveReward:
    def compute_rewards(self, texts):
        rewards = []
        for text in texts:
            sentiment_score = self.sentiment(text)
            length_score = self.length_penalty(text)
            diversity_score = self.diversity(text)
            
            total = 0.5*sentiment_score + 0.3*length_score + 0.2*diversity_score
            rewards.append(total)
        return torch.tensor(rewards)
```

## References

### Key Papers

1. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
2. **RLHF**: Christiano et al., "Deep Reinforcement Learning from Human Preferences" (2017)
3. **InstructGPT**: Ouyang et al., "Training language models to follow instructions" (2022)

### Resources

- [OpenAI Spinning Up in Deep RL](https://spinningup.openai.com/)
- [HuggingFace TRL Documentation](https://huggingface.co/docs/trl/)
- [PPO Explained](https://arxiv.org/abs/1707.06347)

## FAQ

**Q: Why not just use the TRL library directly?**

A: Our implementation provides:
- Compatibility with any TRL version
- Educational value (see exactly how it works)
- Flexibility for customization
- Fallback when TRL has issues

**Q: What's the difference between the manual and TRL implementations?**

A: Both achieve the same goal, but:
- Manual: More control, easier debugging, always works
- TRL: More optimized, potentially faster, requires correct version

**Q: How many training steps do I need?**

A: Depends on your goal:
- Demo/Testing: 10-30 steps
- Development: 50-100 steps
- Production: 500-1000+ steps

**Q: Can I use this with other models besides GPT-2?**

A: Yes! Change `MODEL_NAME` to any causal LM:
```python
config.MODEL_NAME = "gpt2-medium"
config.MODEL_NAME = "EleutherAI/gpt-neo-125M"
config.MODEL_NAME = "distilgpt2"
```

## License

This implementation is provided for educational and research purposes.

## Citation

```bibtex
@software{universal_rlhf_ppo,
  title={Universal RLHF Implementation with KL Control},
  author={Your Name},
  year={2025},
  note={Manual PPO implementation for maximum compatibility}
}
```

---

**Version**: 1.0  
**Last Updated**: 2025  
**Compatibility**: Python 3.7+, PyTorch 1.10+, Transformers 4.20+