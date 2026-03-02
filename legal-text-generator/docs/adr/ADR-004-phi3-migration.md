# ADR-004: Migration from GPT-2 to Phi-3-mini with QLoRA

**Status:** Accepted  
**Date:** 2026-03-02  
**Decision Maker:** Paul Sebastian Kradyel  
**Supersedes:** GPT-2 Medium with LoRA

## Context

The Legal Text Generator used GPT-2 Medium (355M parameters) with LoRA fine-tuning. While functional, output quality was limited by the model's size — generated text often lacked coherence in longer documents, missed legal nuances, and required significant editing.

## Decision Drivers

- Output quality must be suitable for realistic legal document drafts
- Must train on a single RTX 3060 (12GB VRAM)
- Must deploy on HuggingFace Spaces (free tier, 16GB RAM)
- Training time should be reasonable (< 4 hours)
- Must reuse existing 200+ template dataset

## Options Considered

### Option A: Phi-3-mini-4k-instruct with QLoRA (Chosen)
- **Parameters:** 3.8B (10.7x GPT-2 Medium)
- **VRAM for training:** ~6-7GB (4-bit quantization)
- **Quality:** Strong instruction following, good reasoning
- **Training time:** 1-2 hours on RTX 3060

### Option B: Mistral-7B with QLoRA
- **Parameters:** 7.2B
- **VRAM for training:** ~10-12GB (tight fit on 12GB)
- **Quality:** Excellent, but risky on 12GB card
- **Risk:** OOM during training with longer sequences

### Option C: Llama-3.2-3B with QLoRA
- **Parameters:** 3.2B
- **Quality:** Good, similar tier to Phi-3-mini
- **Downside:** Less instruction-tuned, newer with fewer community resources

### Option D: Keep GPT-2 Medium, improve dataset
- Low effort, no new dependencies
- Fundamental quality ceiling from 355M parameters
- Diminishing returns on dataset improvements

## Decision

**Chose Option A: Phi-3-mini with QLoRA**

## Rationale

1. **Quality leap:** 3.8B → dramatically better coherence, reasoning, and instruction following vs 355M. Phi-3 was specifically designed for quality at small scale.

2. **Fits hardware constraint:** QLoRA (NF4 + double quantization) compresses the model to ~2GB. With LoRA adapters, optimizer states, and activations, total VRAM is ~6-7GB — well within RTX 3060's 12GB.

3. **Instruction-tuned base:** Phi-3-mini-4k-instruct already understands the `<|user|>` / `<|assistant|>` format. The model starts from a much better baseline than GPT-2, which has no instruction tuning.

4. **Safe VRAM budget:** Unlike Mistral-7B which would use 10-12GB (risky), Phi-3-mini leaves ~5GB headroom — no OOM concerns even with longer sequences.

5. **Same dataset, different format:** The 200+ legal templates are reused. Only the prompt format changes from `### Instruction / ### Response` to Phi-3 chat template.

## Implementation

```
# Training (1-2 hours)
python train_phi3.py --preset quality

# Inference
python app_phi3.py
```

Key technical changes:
- BitsAndBytesConfig with NF4 quantization
- prepare_model_for_kbit_training() for QLoRA
- Target modules: qkv_proj, o_proj, gate_up_proj, down_proj (Phi-3 architecture)
- paged_adamw_8bit optimizer (memory efficient)
- Gradient checkpointing enabled

## Consequences

### Positive
- Dramatically better text quality and coherence
- Model understands complex multi-part instructions
- Better legal terminology and document structure
- Longer coherent output (4K context vs 1K)

### Negative
- Larger download for base model (~2.5GB quantized vs ~1.5GB GPT-2)
- Requires bitsandbytes library (CUDA-only for training)
- Slightly slower inference than GPT-2 (~2-3s vs ~1s per generation)
- HuggingFace Spaces free tier may need CPU inference (slower)

### Metrics

| Metric | GPT-2 Medium | Phi-3-mini (Expected) |
|--------|-------------|----------------------|
| Coherence (subjective) | 5/10 | 8/10 |
| Legal accuracy | 4/10 | 7/10 |
| Instruction following | 6/10 | 9/10 |
| Training VRAM | 4GB | 6-7GB |
| Inference latency | ~1s | ~2-3s |
| Adapter size | ~6MB | ~20MB |
