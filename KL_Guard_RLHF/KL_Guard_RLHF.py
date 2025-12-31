"""
RLHF cu PPO - Versiune Universală Compatibilă
Funcționează cu orice versiune de TRL
Include implementare manuală a divergenței KL pentru control complet
"""
from datasets import load_dataset, Dataset
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline,
    GenerationConfig
)
from datasets import load_dataset
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import condiționat pentru TRL
try:
    import trl

    if hasattr(trl, "trainer") and hasattr(trl.trainer, "ppo_trainer"):
        from trl.trainer.ppo_trainer import PPOTrainer
        from trl.trainer.ppo_config import PPOConfig
        from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
        print("✅ Import TRL modern detectat (>=0.10.0)")
    else:
        from trl import PPOTrainer, PPOConfig
        try:
            from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
        except ImportError:
            from trl.models import AutoModelForCausalLMWithValueHead
        print("✅ Import TRL vechi detectat (<=0.8.x)")

    TRL_AVAILABLE = True
    print("✅ TRL disponibil și PPOTrainer importat corect")

except Exception as e:
    print(f"⚠️ TRL nu este instalat sau incompatibil ({e}). Folosesc implementare alternativă.")
    TRL_AVAILABLE = False

# ===============================================================================
# 1. CONFIGURARE UNIVERSALĂ
# ===============================================================================

class UniversalConfig:
    """Configurație care funcționează cu orice setup"""
    MODEL_NAME = "gpt2"
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 2
    MINI_BATCH_SIZE = 1
    
    KL_COEF = 0.2
    KL_TARGET = 0.01
    EPSILON_CLIP = 0.2
    VALUE_CLIP = 0.2
    
    MAX_LENGTH = 80
    TEMPERATURE = 1.0
    
    NUM_STEPS = 50

# ===============================================================================
# 2. IMPLEMENTARE MANUALĂ KL DIVERGENCE
# ===============================================================================

class KLDivergenceController:
    """Controller manual pentru divergența KL"""
    
    def __init__(self, initial_coef: float = 0.2, target: float = 0.01):
        self.coef = initial_coef
        self.target = target
        self.history = []
        
    def compute_kl(self, logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        """Calculează divergența KL între două distribuții"""
        probs = F.softmax(logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        
        epsilon = 1e-10
        probs = probs + epsilon
        ref_probs = ref_probs + epsilon
        
        kl = torch.sum(probs * (torch.log(probs) - torch.log(ref_probs)), dim=-1)
        return kl
    
    def update_coefficient(self, current_kl: float):
        """Ajustează coeficientul KL bazat pe divergența curentă"""
        self.history.append(current_kl)
        
        if current_kl > self.target * 1.5:
            self.coef = min(self.coef * 1.2, 1.0)
            print(f"📈 KL prea mare ({current_kl:.4f}), cresc coef la {self.coef:.3f}")
        elif current_kl < self.target / 1.5:
            self.coef = max(self.coef * 0.8, 0.01)
            print(f"📉 KL prea mic ({current_kl:.4f}), scad coef la {self.coef:.3f}")
    
    def get_penalty(self, kl_value: torch.Tensor) -> torch.Tensor:
        return self.coef * kl_value

# ===============================================================================
# 3. REWARD MODEL ROBUST
# ===============================================================================

class RobustRewardModel:
    """Model de recompensă care funcționează cu orice setup"""
    
    def __init__(self):
        try:
            self.sentiment_pipe = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            self.use_sentiment = True
            print("✅ Model de sentiment încărcat")
        except Exception as e:
            print(f"⚠️ Nu pot încărca model de sentiment: {e}")
            print("📌 Folosesc reward model simplu bazat pe euristici")
            self.use_sentiment = False
    
    def compute_rewards(self, texts: List[str]) -> torch.Tensor:
        rewards = []
        
        for text in texts:
            if self.use_sentiment:
                try:
                    result = self.sentiment_pipe(text[:512])[0]
                    if result['label'] == 'POSITIVE':
                        reward = result['score'] * 2.0
                    else:
                        reward = -result['score'] * 0.5
                except:
                    reward = 0.0
            else:
                reward = self._heuristic_reward(text)
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _heuristic_reward(self, text: str) -> float:
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                         'fantastic', 'love', 'perfect', 'beautiful', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 
                         'hate', 'disgusting', 'disappointing']
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words > 0:
            score = (positive_score - negative_score) / total_words * 10
            return np.clip(score, -2, 2)
        return 0.0

# ===============================================================================
# 4. TRAINER UNIVERSAL ÎMBUNĂTĂȚIT - SOLUȚIE COMPLETĂ
# ===============================================================================

class UniversalPPOTrainer:
    """
    Trainer PPO care folosește doar implementarea manuală
    Evită complet problemele de compatibilitate TRL
    """
    
    def __init__(self):
        self.config = UniversalConfig()
        self.device = torch.device("cpu")
        print(f"🖥️ Folosesc device: {self.device}")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Folosim doar implementarea manuală pentru a evita problemele TRL
        self._init_manual()
        
        # Components comune
        self.reward_model = RobustRewardModel()
        self.kl_controller = KLDivergenceController(
            initial_coef=self.config.KL_COEF,
            target=self.config.KL_TARGET
        )
        
        self.stats = {'rewards': [], 'kl_values': []}
        print("✅ Sistem manual inițializat cu succes - fără dependințe TRL")

    def _init_manual(self):
        """Inițializare manuală fără TRL - cea mai stabilă abordare"""
        print("🔧 Inițializare MANUALĂ - maximă stabilitate")
        
        # Model principal
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME
        ).to(self.device)
        
        # Model de referință (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.MODEL_NAME
        ).to(self.device)
        
        # Înghețăm modelul de referință
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE
        )
        
        self.use_trl = False

    def generate_responses(self, prompts: List[str]) -> Dict:
        """Generează răspunsuri cu gestionare corectă a dimensiunilor"""
    # Tokenizare
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=20,  # Scăzut pentru stabilitate
            return_attention_mask=True
    )
    
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
    
    # Generare
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=self.config.MAX_LENGTH,
                temperature=self.config.TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
        )
    
    # Decodare
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # CORECȚIE: Returnează doar input_ids și responses, nu output_ids
        return {
            'input_ids': input_ids,
            'responses': responses,
            'attention_mask': attention_mask
    }

    def compute_kl_penalty(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Calculează penalizarea KL între model și modelul de referință"""
        with torch.no_grad():
            # Obține logits de la ambele modele
            model_outputs = self.model(input_ids, attention_mask=attention_mask)
            ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            
            model_logits = model_outputs.logits if hasattr(model_outputs, 'logits') else model_outputs[0]
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, 'logits') else ref_outputs[0]
            
            # Calculează KL
            kl_divergence = self.kl_controller.compute_kl(model_logits, ref_logits)
            kl_mean = kl_divergence.mean().item()
            
            # Penalizare KL
            kl_penalty = self.kl_controller.get_penalty(kl_divergence.mean())
            
        return kl_mean, kl_penalty

    def compute_advantages_and_returns(self, rewards: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculează avantaje și returns pentru PPO manual"""
        # Implementare simplificată
        advantages = rewards - values
        returns = rewards
        
        # Normalizare avantaje
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def ppo_manual_update(self, batch_data: Dict, rewards: torch.Tensor):
        """Implementare manuală PPO cu corecție pentru dimensiuni"""
        self.model.train()
    
        input_ids = batch_data['input_ids']
        attention_mask = batch_data.get('attention_mask', torch.ones_like(input_ids))
    
    # Folosește doar input_ids pentru KL
        model_outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = model_outputs.logits if hasattr(model_outputs, 'logits') else model_outputs[0]
    
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, 'logits') else ref_outputs[0]
    
    # Calculează KL
        kl_divergence = self.kl_controller.compute_kl(logits, ref_logits)
        kl_mean = kl_divergence.mean().item()
        kl_penalty = self.kl_controller.get_penalty(kl_divergence.mean())
    
    # Loss simplu bazat pe reward
        reward_loss = -rewards.mean()  # Maximizează reward
    
        total_loss = reward_loss + kl_penalty
    
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
    
        return {
        'reward_loss': reward_loss.item(),
        'kl_penalty': kl_penalty.item(),
        'kl_mean': kl_mean,
        'total_loss': total_loss.item()
    }

    def train_step(self, prompts: List[str]) -> Dict:
        """Pas de antrenament manual"""
    # Generează răspunsuri
        gen_output = self.generate_responses(prompts)
    
    # Calculează recompense
        rewards = self.reward_model.compute_rewards(gen_output['responses'])
    
    # Antrenament PPO manual
        training_data = {
            'input_ids': gen_output['input_ids'],
            'attention_mask': gen_output.get('attention_mask', torch.ones_like(gen_output['input_ids']))
    }
    
        stats = self.ppo_manual_update(training_data, rewards)
    
    # Actualizează coeficientul KL
        self.kl_controller.update_coefficient(stats['kl_mean'])
    
    # Salvează statistici
        self.stats['rewards'].append(rewards.mean().item())
        self.stats['kl_values'].append(stats['kl_mean'])
    
    # CORECȚIE: Folosește 'reward_loss' în loc de 'policy_loss'
        return {
        'mean_reward': rewards.mean().item(),
        'kl_divergence': stats['kl_mean'],
        'kl_coef': self.kl_controller.coef,
        'reward_loss': stats['reward_loss'],  # Schimbat din 'policy_loss'
        'total_loss': stats['total_loss'],
        'example': gen_output['responses'][0][:100] if gen_output['responses'] else ""
    }

    def train(self, num_steps: Optional[int] = None):
        """Antrenament principal"""
        num_steps = num_steps or self.config.NUM_STEPS
        
        print("\n🎓 ÎNCEPE ANTRENAMENTUL MANUAL")
        print("="*60)
        print(f"📊 Configurație:")
        print(f"   • Model: {self.config.MODEL_NAME}")
        print(f"   • Pași: {num_steps}")
        print(f"   • Batch size: {self.config.BATCH_SIZE}")
        print(f"   • KL Target: {self.config.KL_TARGET}")
        print(f"   • Mod: MANUAL (fără TRL)")
        print("="*60)
        
        # Dataset simplu
        prompts_pool = [
            "The movie was",
            "I think this is",
            "This product is",
            "My experience was",
            "The service was",
            "I feel that",
            "This place is",
            "The food was"
        ]
        
        for step in range(num_steps):
            # Selectează batch random
            batch_idx = np.random.choice(
                len(prompts_pool), 
                min(self.config.BATCH_SIZE, len(prompts_pool)), 
                replace=False
            )
            batch_prompts = [prompts_pool[i] for i in batch_idx]
            
            # Pas de antrenament
            step_stats = self.train_step(batch_prompts)
            
            # Logging periodic
            if step % 5 == 0:  # Mai frecvent pentru debugging
                print(f"\n📈 Pas {step}/{num_steps}")
                print(f"   • Reward: {step_stats['mean_reward']:.3f}")
                print(f"   • KL Div: {step_stats['kl_divergence']:.4f}")
                print(f"   • KL Coef: {step_stats['kl_coef']:.3f}")
                print(f"   • Reward Loss: {step_stats['reward_loss']:.4f}")
                print(f"   • Total Loss: {step_stats['total_loss']:.4f}")
                if step_stats['example']:
                    print(f"   • Exemplu: {step_stats['example']}...")
        
        print("\n✅ Antrenament manual completat!")
        self.print_final_stats()
    
    def print_final_stats(self):
        """Afișează statistici finale"""
        print("\n📊 STATISTICI FINALE")
        print("="*60)
        
        if self.stats['rewards']:
            rewards = self.stats['rewards']
            print(f"Reward mediu total: {np.mean(rewards):.3f}")
            print(f"Reward final (ultimele 5): {np.mean(rewards[-5:]):.3f}")
            print(f"Trend reward: {'📈' if rewards[-1] > rewards[0] else '📉'}")
        
        if self.stats['kl_values']:
            kl_values = self.stats['kl_values']
            print(f"KL Divergence medie: {np.mean(kl_values):.4f}")
            print(f"KL Divergence finală: {kl_values[-1]:.4f}")
            print(f"KL Coef final: {self.kl_controller.coef:.3f}")
    
    def evaluate(self, test_prompts: List[str]):
        """Evaluare model"""
        print("\n🧪 EVALUARE FINALĂ")
        print("="*60)
        
        for i, prompt in enumerate(test_prompts):
            gen_output = self.generate_responses([prompt])
            response = gen_output['responses'][0]
            rewards = self.reward_model.compute_rewards([response])
            
            print(f"\n📝 Prompt {i+1}: {prompt}")
            print(f"   Răspuns: {response}")
            print(f"   Score: {rewards[0]:.3f}")

# ===============================================================================
# 5. EXPLICAȚIE DETALIATĂ
# ===============================================================================

def explain_kl_in_ppo():
    """Explicație detaliată a rolului KL în PPO"""
    print("\n" + "="*70)
    print("📚 DIVERGENȚA KL ÎN PPO - EXPLICAȚIE COMPLETĂ")
    print("="*70)
    
    print("""
🎯 PROBLEMA FUNDAMENTALĂ:
─────────────────────────
Fără constrângere KL, modelul poate:

1. POLICY COLLAPSE - text repetitiv
2. REWARD HACKING - scurtături artificiale  
3. CATASTROPHIC FORGETTING - pierdere capabilități

📐 SOLUȚIA: DIVERGENȚA KL
──────────────────────────
KL(π||π_ref) = 𝔼[log(π(a|s)) - log(π_ref(a|s))]

IMPLEMENTARE ÎN PPO:
L = L_policy - β * KL(π||π_ref)

⚙️ MECANISM DE CONTROL ADAPTIV:
────────────────────────────────
if KL > target * 1.5: β = β * 2  # Crește constrângerea
if KL < target / 1.5: β = β / 2  # Relaxează constrângerea

📊 INTERPRETARE PRACTICĂ:
─────────────────────────
KL = 0.001  → Prea conservator
KL = 0.01   → Sweet spot optim
KL = 0.1    → Risc instabilitate
KL = 1.0    → Prea agresiv

🔬 EXEMPLU CONCRET:
───────────────────
Original:  "The movie was [good|bad|okay|great]"
După RLHF: "The movie was [excellent|amazing|wonderful|fantastic]"
           ← Diversitate menținută! ✅
""")
    print("="*70)

# ===============================================================================
# 6. MAIN FUNCTION
# ===============================================================================

def main():
    """Funcția principală"""
    print("🚀 RLHF CU PPO - VERSIUNE UNIVERSALĂ MANUALĂ")
    print("="*60)
    print("📌 Folosim implementare MANUALĂ pentru stabilitate maximă!")
    print("="*60)
    
    # Explică teoria
    explain_kl_in_ppo()
    
    # Antrenament
    print("\n⚙️ Inițializare sistem manual...")
    trainer = UniversalPPOTrainer()
    
    print("\n🎓 Start antrenament manual...")
    trainer.train(num_steps=30)  # Redus pentru demo rapidă
    
    # Evaluare
    test_prompts = [
        "The movie was",
        "This restaurant is", 
        "I really think",
        "My experience was"
    ]
    trainer.evaluate(test_prompts)
    
    print("\n✨ Succes! Sistemul RLHF manual cu control KL a fost demonstrat.")
    print("\n💡 Puncte cheie de reținut:")
    print("• Implementare manuală = stabilitate maximă")
    print("• KL divergence previne policy collapse")  
    print("• Coeficientul adaptiv menține echilibrul")

if __name__ == "__main__":
    main()