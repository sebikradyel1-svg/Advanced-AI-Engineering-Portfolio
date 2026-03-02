#!/usr/bin/env python3
"""
Legal & Business Text Generator — Phi-3-mini QLoRA Fine-Tuning
===============================================================

Migration from GPT-2 Medium (355M) → Phi-3-mini-4k-instruct (3.8B)
using QLoRA (4-bit quantization) to fit on RTX 3060 12GB.

Why Phi-3-mini over GPT-2?
  - 10x more parameters (3.8B vs 355M) → much better text quality
  - Instruction-tuned base → already understands instruction format
  - 4-bit quantization → fits in ~4GB VRAM (vs 2.5GB for GPT-2 fp16)
  - Better reasoning and coherence for legal document generation

VRAM Budget (RTX 3060 12GB):
  - Model weights (4-bit): ~2.0 GB
  - LoRA adapters (fp16):  ~0.1 GB
  - Optimizer states:      ~0.3 GB
  - Activations + KV cache: ~3-4 GB
  - Total:                 ~6-7 GB ✓ (leaves headroom)

Usage:
    # Quick test (15-20 min)
    python train_phi3.py --preset quick

    # Quality training (1-2 hours) — RECOMMENDED
    python train_phi3.py --preset quality

    # Extended training (3-4 hours)
    python train_phi3.py --preset extended

Author: Paul Sebastian Kradyel — AI Engineering Portfolio
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================

BASE_MODEL = "microsoft/phi-3-mini-4k-instruct"

TRAINING_PRESETS = {
    "quick": {
        "description": "Quick test (15-20 min on RTX 3060)",
        "epochs": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "warmup_ratio": 0.1,
        "max_length": 512,
    },
    "quality": {
        "description": "Quality training (1-2 hours) — RECOMMENDED",
        "epochs": 3,
        "lora_r": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "warmup_ratio": 0.1,
        "max_length": 768,
    },
    "extended": {
        "description": "Extended training (3-4 hours)",
        "epochs": 5,
        "lora_r": 64,
        "lora_alpha": 128,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "warmup_ratio": 0.15,
        "max_length": 768,
    },
}

# Phi-3 target modules (attention + MLP for full coverage)
PHI3_TARGET_MODULES = [
    "qkv_proj",    # fused QKV attention
    "o_proj",      # attention output
    "gate_up_proj", # fused gate+up MLP
    "down_proj",   # MLP down projection
]


# ============================================================================
# DATASET (reuse existing legal/business templates)
# ============================================================================

def create_legal_business_dataset():
    """
    Import the existing dataset from the GPT-2 training script.
    This keeps the same 200+ templates — only the prompt format changes.
    """
    # Import from existing script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from english_legal_llm_finetuning_LONG import create_legal_business_dataset as load_dataset
        dataset = load_dataset()
        logger.info(f"Loaded {len(dataset)} examples from existing dataset")
        return dataset
    except ImportError:
        logger.warning("Could not import existing dataset, using built-in subset")
        return _fallback_dataset()


def _fallback_dataset():
    """Minimal dataset if import fails — just for testing."""
    return [
        {
            "instruction": "Write a confidentiality clause for a software development employment contract.",
            "output": "CONFIDENTIALITY AND PROPRIETARY INFORMATION\n\n1. Definition of Confidential Information\n\"Confidential Information\" means all non-public information relating to the Company's business, including: trade secrets, source code, algorithms, customer lists, business plans, and financial data.\n\n2. Non-Disclosure Obligations\nEmployee agrees to: (a) hold all Confidential Information in strict confidence; (b) not disclose to any third party without prior written consent; (c) use Confidential Information solely for employment duties.\n\n3. Duration\nThese obligations shall survive termination for five (5) years, or indefinitely for trade secrets.",
        },
        {
            "instruction": "Draft a professional email requesting a deadline extension from a client.",
            "output": "Subject: Request for Project Deadline Extension\n\nDear [Client Name],\n\nI hope this message finds you well. I am writing regarding the timeline for our current project.\n\nAfter careful review of remaining deliverables, I would like to request a one-week extension to our deadline. This additional time will allow us to ensure the highest quality standards are met.\n\nI sincerely apologize for any inconvenience and am happy to discuss this further at your convenience.\n\nBest regards,\n[Your Name]",
        },
    ]


def format_for_phi3(instruction: str, output: str) -> str:
    """
    Format a training example using Phi-3 chat template.

    GPT-2 format:  ### Instruction:\n{inst}\n\n### Response:\n{out}
    Phi-3 format:  <|user|>\n{inst}<|end|>\n<|assistant|>\n{out}<|end|>
    """
    return f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"


def create_tokenized_dataset(dataset, tokenizer, max_length=768, eval_split=0.1):
    """Tokenize dataset and split into train/eval."""
    formatted = []
    skipped = 0

    for item in dataset:
        text = format_for_phi3(item["instruction"], item["output"])
        tokens = tokenizer(text, truncation=False)
        if len(tokens["input_ids"]) <= max_length:
            formatted.append({"text": text})
        else:
            skipped += 1

    if skipped > 0:
        logger.warning(f"Skipped {skipped} examples exceeding {max_length} tokens")

    logger.info(f"Dataset: {len(formatted)} examples (skipped {skipped})")

    hf_dataset = Dataset.from_list(formatted)

    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = hf_dataset.map(tokenize_fn, batched=True, remove_columns=["text"], desc="Tokenizing")

    split = tokenized.train_test_split(test_size=eval_split, seed=42)
    logger.info(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")
    return split["train"], split["test"]


# ============================================================================
# MODEL SETUP
# ============================================================================

def load_model_and_tokenizer(lora_r=32, lora_alpha=64):
    """
    Load Phi-3-mini with 4-bit quantization (QLoRA).

    Memory breakdown:
      - 4-bit weights: ~2 GB
      - LoRA adapters:  ~100 MB
      - Total model:    ~2.1 GB
    """
    logger.info(f"Loading {BASE_MODEL} with 4-bit quantization...")

    # QLoRA config: NF4 quantization with double quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # flash_attention_2 needs Ampere+
    )

    # Prepare for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # LoRA config targeting Phi-3 architecture
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=PHI3_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Log parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA r={lora_r}, alpha={lora_alpha}")
    logger.info(f"Target modules: {PHI3_TARGET_MODULES}")
    logger.info(f"Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
    logger.info(f"Total: {total:,}")

    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / 1e9
        logger.info(f"VRAM allocated: {vram_gb:.2f} GB")

    return model, tokenizer


# ============================================================================
# TRAINING
# ============================================================================

def train(model, tokenizer, train_dataset, eval_dataset, args):
    """Run QLoRA training with evaluation and checkpointing."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        logging_first_step=True,
        report_to=["tensorboard"],
        # Eval & save
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Optimization
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",  # Memory-efficient optimizer for QLoRA
        gradient_checkpointing=True,  # Saves ~30% VRAM
        # Other
        remove_unused_columns=False,
        seed=42,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    logger.info(f"Starting training: {args.epochs} epochs, lr={args.learning_rate}")
    logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")

    trainer.train()

    # Save adapter weights + tokenizer
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    metadata = {
        "base_model": BASE_MODEL,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": PHI3_TARGET_MODULES,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "trained_at": datetime.now().isoformat(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(f"{output_dir}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return trainer


# ============================================================================
# INFERENCE TEST
# ============================================================================

def test_model(model, tokenizer):
    """Quick generation test after training."""
    test_prompts = [
        "Write a confidentiality clause for a software development employment contract.",
        "Draft a professional email requesting a deadline extension from a client.",
        "Write the introduction section of a mutual non-disclosure agreement.",
        "Create a data retention policy section for a tech company.",
        "Draft a termination clause for an at-will employment agreement.",
    ]

    logger.info("\n" + "=" * 60)
    logger.info("GENERATION TEST — Phi-3-mini QLoRA")
    logger.info("=" * 60)

    model.eval()
    for i, instruction in enumerate(test_prompts, 1):
        prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract assistant response
        if "<|assistant|>" in generated:
            response = generated.split("<|assistant|>")[-1]
            response = response.replace("<|end|>", "").replace("<|endoftext|>", "").strip()
        else:
            response = generated

        logger.info(f"\n--- Test {i} ---")
        logger.info(f"Instruction: {instruction}")
        logger.info(f"Response:\n{response[:800]}...")
        logger.info("-" * 40)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phi-3-mini QLoRA Fine-Tuning for Legal Text")
    parser.add_argument("--preset", choices=list(TRAINING_PRESETS.keys()), help="Training preset")
    parser.add_argument("--output_dir", default="./phi3_legal_lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--eval_split", type=float, default=0.1)
    parser.add_argument("--test_only", action="store_true", help="Only run generation test")
    args = parser.parse_args()

    # Apply preset
    if args.preset:
        preset = TRAINING_PRESETS[args.preset]
        logger.info(f"Preset: {args.preset} — {preset['description']}")
        for key, val in preset.items():
            if key != "description":
                setattr(args, key, val)

    set_seed(42)

    logger.info("=" * 60)
    logger.info("PHI-3-MINI QLoRA FINE-TUNING — Legal Text Generator")
    logger.info("=" * 60)
    logger.info(f"Base model: {BASE_MODEL}")
    logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    model, tokenizer = load_model_and_tokenizer(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    if args.test_only:
        test_model(model, tokenizer)
        return

    # Load and tokenize dataset
    raw_dataset = create_legal_business_dataset()
    train_ds, eval_ds = create_tokenized_dataset(
        raw_dataset, tokenizer, max_length=args.max_length, eval_split=args.eval_split
    )

    # Train
    trainer = train(model, tokenizer, train_ds, eval_ds, args)

    # Test
    test_model(model, tokenizer)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Adapter saved to: {args.output_dir}")
    logger.info(f"TensorBoard: tensorboard --logdir {args.output_dir}/logs")
    logger.info(f"\nTo deploy: copy {args.output_dir}/ to HuggingFace Space model/ folder")
    logger.info(f"Update app.py to use base_model='{BASE_MODEL}'")


if __name__ == "__main__":
    main()
