"""
Antrenarea unui Model de Recompensă pentru Distincția Calității Răspunsurilor
Folosind RLHF (Reinforcement Learning from Human Feedback)
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# Configurare dispozitiv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Folosim dispozitivul: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("🎯 Configurație optimizată pentru RTX 3060 6GB")
else:
    print("⚠️  Atenție: Rulezi pe CPU - antrenamentul va fi FOARTE lent!")

# ==================== 1. ÎNCĂRCAREA DATELOR ====================
print("\n[1/8] Încărcarea setului de date...")
dataset = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise", split="train")

# Vizualizare exemplu
print(f"Număr total de exemple: {len(dataset)}")
print("\nExemplu din dataset:")
print(f"Prompt: {dataset[0]['prompt'][:100]}...")
print(f"Chosen: {dataset[0]['chosen'][:100]}...")
print(f"Rejected: {dataset[0]['rejected'][:100]}...")

# ==================== 2. CONFIGURAREA MODELULUI ====================
print("\n[2/8] Configurarea modelului de bază...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 nu are pad_token implicit
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model pentru clasificare secvențială (output scalar pentru scor)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,  # Un singur scor de recompensă
    torch_dtype=torch.float32  # FP32 pentru stabilitate pe 6GB VRAM
)
model.config.pad_token_id = tokenizer.pad_token_id

# ==================== 3. CONFIGURAREA LoRA/PEFT ====================
print("\n[3/8] Configurarea LoRA pentru antrenament eficient...")
lora_config = LoraConfig(
    r=16,  # Rang LoRA
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],  # Module țintă pentru GPT-2
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Gradient checkpointing - dezactivat temporar pentru debugging
# model.gradient_checkpointing_enable()
model.to(device)
# ==================== 4. PREPROCESAREA DATELOR ====================
print("\n[4/8] Preprocesarea datelor...")

def preprocess_function(examples: Dict) -> Dict:
    """
    Pregătește perechi de (prompt + chosen) și (prompt + rejected) pentru tokenizare.
    """
    # Process chosen
    chosen_texts = [f"{prompt}\n\nRăspuns: {chosen}" for prompt, chosen in zip(examples['prompt'], examples['chosen'])]
    chosen_encodings = tokenizer(
        chosen_texts,
        truncation=True,
        max_length=384,  # Redus pentru 6GB VRAM
        padding="max_length",
        return_tensors="pt"
    )
    
    # Process rejected
    rejected_texts = [f"{prompt}\n\nRăspuns: {rejected}" for prompt, rejected in zip(examples['prompt'], examples['rejected'])]
    rejected_encodings = tokenizer(
        rejected_texts,
        truncation=True,
        max_length=384,  # Redus pentru 6GB VRAM
        padding="max_length",
        return_tensors="pt"
    )
    
    # Return as dictionary with lists
    return {
        "input_ids_chosen": chosen_encodings["input_ids"].tolist(),
        "attention_mask_chosen": chosen_encodings["attention_mask"].tolist(),
        "input_ids_rejected": rejected_encodings["input_ids"].tolist(),
        "attention_mask_rejected": rejected_encodings["attention_mask"].tolist(),
    }

# Aplicare preprocesare
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizare"
)

# Flatten the dataset since we now have 2x the examples
processed_dataset = processed_dataset.flatten()

# Split pentru train și evaluare
dataset_dict = processed_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]

print(f"Set de antrenament: {len(train_dataset)} exemple")
print(f"Set de evaluare: {len(eval_dataset)} exemple")

# ==================== 5. DATA COLLATOR CUSTOMIZAT ====================
print("\n[5/8] Configurarea data collator...")

@dataclass
class RewardDataCollator:
    """
    Data Collator pentru perechi de (chosen, rejected).
    Pregătește batch-urile pentru antrenament.
    """
    tokenizer: AutoTokenizer
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Colectează și padding-uiește datele pentru un batch.
        """
        batch = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        
        for feature in features:
            batch["input_ids_chosen"].append(torch.tensor(feature["input_ids_chosen"]))
            batch["attention_mask_chosen"].append(torch.tensor(feature["attention_mask_chosen"]))
            batch["input_ids_rejected"].append(torch.tensor(feature["input_ids_rejected"]))
            batch["attention_mask_rejected"].append(torch.tensor(feature["attention_mask_rejected"]))
        
        # Stack tensors
        batch = {k: torch.stack(v) for k, v in batch.items()}
        
        return batch

# Inițializare data collator
data_collator = RewardDataCollator(tokenizer=tokenizer)

# ==================== 6. REWARD TRAINER CUSTOMIZAT ====================
print("\n[6/8] Configurarea trainer-ului...")

class RewardTrainer(Trainer):
    """
    Trainer customizat pentru modelul de recompensă.
    Învață să maximizeze diferența între scorurile chosen și rejected.
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Loss personalizat: preferăm ca scorul chosen > scorul rejected
        
        Args:
            model: Modelul de recompensă
            inputs: Dict cu input_ids și attention_mask pentru chosen și rejected
            return_outputs: Dacă să returneze și output-urile
            **kwargs: Parametri suplimentari (num_items_in_batch, etc.) pentru compatibilitate
        """
        # Forward pass pentru chosen
        rewards_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"]
        ).logits
        
        # Forward pass pentru rejected
        rewards_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"]
        ).logits
        
        # Loss: vrem ca reward_chosen - reward_rejected să fie > 0
        # Folosim -log(sigmoid(diff)) care este echivalent cu binary cross-entropy
        loss = -torch.nn.functional.logsigmoid(
            rewards_chosen - rewards_rejected
        ).mean()
        
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected
            }
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override pentru evaluare - calculează loss-ul custom și returnează predicții.
        """
        with torch.no_grad():
            # Compute loss
            loss = self.compute_loss(model, inputs)
            
            # Compute predictions (scorurile)
            rewards_chosen = model(
                input_ids=inputs["input_ids_chosen"],
                attention_mask=inputs["attention_mask_chosen"]
            ).logits
            
            rewards_rejected = model(
                input_ids=inputs["input_ids_rejected"],
                attention_mask=inputs["attention_mask_rejected"]
            ).logits
            
        if prediction_loss_only:
            return (loss, None, None)
        
        # Returnează loss, predictions (chosen scores), labels (rejected scores pentru comparație)
        return (loss, rewards_chosen, rewards_rejected)

# Configurare antrenament - OPTIMIZAT pentru RTX 3060 6GB
training_args = TrainingArguments(
    output_dir="./reward_model_output",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Foarte mic pentru 6GB VRAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch = 16
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,  # Gradient clipping pentru stabilitate FP16
    warmup_steps=100,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=True,  # RTX 3060 suportă FP16 foarte bine
    fp16_full_eval=True,  # Economisim VRAM la evaluare
    bf16=False,
    gradient_checkpointing=False,  # Dezactivat temporar pentru debugging
    optim="adamw_torch",  # Optimizer eficient
    torch_compile=False,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,  # Pentru Windows
    dataloader_pin_memory=True,  # Performanță mai bună
)

# Inițializare trainer cu data collator
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,  # ✅ ADĂUGAT
)
# ==================== 7. ANTRENAMENT ====================
print("\n[7/8] Începerea antrenamentului...\n")
trainer.train()

# Salvare model final
model.save_pretrained("./reward_model_final")
tokenizer.save_pretrained("./reward_model_final")
print("\n✅ Model salvat în: ./reward_model_final")

# ==================== 8. EVALUARE ====================
print("\n" + "="*60)
print("EVALUARE FINALĂ")
print("="*60)

def evaluate_reward_model(model, eval_dataset, num_samples=100):
    """
    Evaluează acuratețea modelului pe seturi de perechi.
    Acuratețea = % din cazuri când scorul chosen > scorul rejected
    """
    model.eval()
    correct = 0
    total = 0
    
    rewards_chosen_list = []
    rewards_rejected_list = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(eval_dataset))):
            sample = eval_dataset[i]
            
            # Evaluare chosen
            reward_chosen = model(
                input_ids=torch.tensor([sample["input_ids_chosen"]]).to(device),
                attention_mask=torch.tensor([sample["attention_mask_chosen"]]).to(device)
            ).logits.item()
            
            # Evaluare rejected
            reward_rejected = model(
                input_ids=torch.tensor([sample["input_ids_rejected"]]).to(device),
                attention_mask=torch.tensor([sample["attention_mask_rejected"]]).to(device)
            ).logits.item()
            
            rewards_chosen_list.append(reward_chosen)
            rewards_rejected_list.append(reward_rejected)
            
            # Verificare preferință
            if reward_chosen > reward_rejected:
                correct += 1
            total += 1
    
    accuracy = correct / total
    avg_chosen = np.mean(rewards_chosen_list)
    avg_rejected = np.mean(rewards_rejected_list)
    margin = avg_chosen - avg_rejected
    
    print(f"\n📊 REZULTATE EVALUARE (pe {total} exemple):")
    print(f"  • Acuratețe (Accuracy): {accuracy:.2%}")
    print(f"  • Scor mediu pentru 'chosen': {avg_chosen:.4f}")
    print(f"  • Scor mediu pentru 'rejected': {avg_rejected:.4f}")
    print(f"  • Marjă medie (chosen - rejected): {margin:.4f}")
    
    return accuracy, avg_chosen, avg_rejected

accuracy, avg_chosen, avg_rejected = evaluate_reward_model(
    model, 
    eval_dataset, 
    num_samples=200
)

# ==================== 9. TEST INTERACTIV ====================
print("\n" + "="*60)
print("TEST INTERACTIV")
print("="*60)

def score_response(prompt: str, response: str) -> float:
    """
    Calculează scorul de recompensă pentru un răspuns dat.
    """
    text = f"{prompt}\n\nRăspuns: {response}"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=384
    ).to(device)
    
    model.eval()
    with torch.no_grad():
        score = model(**inputs).logits.item()
    return score

# Exemplu de test
test_prompt = "Explică ce este inteligența artificială."
response_good = "Inteligența artificială (AI) este un domeniu al informaticii care se ocupă cu crearea de sisteme capabile să efectueze sarcini care necesită inteligență umană, precum învățarea, raționamentul și rezolvarea problemelor."
response_bad = "AI e ceva cu computere."

score_good = score_response(test_prompt, response_good)
score_bad = score_response(test_prompt, response_bad)

print(f"\nPrompt: {test_prompt}")
print(f"\nRăspuns calitate înaltă: {response_good}")
print(f"Scor: {score_good:.4f}")
print(f"\nRăspuns calitate scăzută: {response_bad}")
print(f"Scor: {score_bad:.4f}")
print(f"\n✅ Modelul preferă răspunsul bun: {score_good > score_bad}")

print("\n" + "="*60)
print("ANTRENAMENT FINALIZAT CU SUCCES! 🎉")
print("="*60)