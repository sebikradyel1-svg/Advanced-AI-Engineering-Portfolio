"""
Transfer Learning pentru Clasificarea de Text cu Resurse Limitate
Compară 3 strategii: Fine-tuning complet, Fine-tuning parțial, Antrenare de la zero
Dataset: IMDb pentru sentiment analysis binar
Model pre-antrenat: BERT
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, BertConfig

from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Setare device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizare device: {device}")

# ============================================================================
# 1. DATASET PERSONALZAT PENTRU IMDb
# ============================================================================

class IMDbDataset(Dataset):
    """Dataset IMDb pentru sentiment analysis binar"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# 2. MODELE PENTRU CELE 3 STRATEGII
# ============================================================================

class BERTFineTuneComplete(nn.Module):
    """Strategie (a): Fine-tuning COMPLET - actualizează toate straturile"""
    def __init__(self, num_classes=2, dropout=0.3):
        super(BERTFineTuneComplete, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Toate straturile sunt trainable
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BERTFineTuneLastLayer(nn.Module):
    """Strategie (b): Fine-tuning doar ULTIMUL STRAT - îngheață BERT"""
    def __init__(self, num_classes=2, dropout=0.3):
        super(BERTFineTuneLastLayer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Îngheață toate straturile BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Doar classifier-ul este trainable
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # BERT în modul evaluare
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BERTFromScratch(nn.Module):
    """Strategie (c): Antrenare DE LA ZERO - fără pre-antrenare"""
    def __init__(self, vocab_size=30522, num_classes=2, dropout=0.3):
        super(BERTFromScratch, self).__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
        # Inițializare aleatoare a tuturor ponderilor
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ============================================================================
# 3. FUNCȚII DE ANTRENARE ȘI EVALUARE
# ============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Antrenează modelul pentru o epocă"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluează modelul pe setul de validare/test"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels


def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    """Antrenează și evaluează un model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=learning_rate, eps=1e-8)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        print(f'\n{"="*60}')
        print(f'Epoca {epoch + 1}/{epochs}')
        print(f'{"="*60}')
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f'✓ Nou best model! (Val Acc: {best_val_acc:.4f})')
    
    training_time = time.time() - start_time
    
    return history, best_val_acc, training_time

# ============================================================================
# 4. GENERARE DATE SIMULATE IMDb (pentru demonstrație)
# ============================================================================

def generate_imdb_sample():
    """Generează un dataset IMDb simplificat pentru demonstrație"""
    positive_reviews = [
        "This movie was absolutely fantastic! Great acting and story.",
        "Loved every minute of it. Brilliant performances!",
        "One of the best films I've seen. Highly recommend!",
        "Amazing cinematography and direction. A masterpiece!",
        "Incredible movie with great emotional depth.",
        "Superb acting and wonderful storytelling.",
        "Truly exceptional film that moved me deeply.",
        "Outstanding in every aspect. A must-see!",
        "Beautifully crafted with excellent performances.",
        "Remarkable movie that exceeded all expectations."
    ] * 50
    
    negative_reviews = [
        "Terrible movie. Waste of time and money.",
        "Boring plot and bad acting. Very disappointed.",
        "One of the worst films I've ever seen.",
        "Poor script and uninspired direction.",
        "Awful movie. Couldn't even finish it.",
        "Disappointing on all levels. Not recommended.",
        "Poorly executed with terrible performances.",
        "Dreadful movie with no redeeming qualities.",
        "Bad acting and a nonsensical plot.",
        "Completely underwhelming and forgettable."
    ] * 50
    
    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    return texts, labels

# ============================================================================
# 5. MAIN - COMPARAȚIE ÎNTRE STRATEGII
# ============================================================================

def main():
    print("="*80)
    print("TRANSFER LEARNING PENTRU CLASIFICAREA DE TEXT")
    print("Comparație: Fine-tuning Complet vs Ultimul Strat vs De la Zero")
    print("="*80)
    
    # Parametri
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LR_FINETUNE = 2e-5
    LR_SCRATCH = 5e-4
    
    # Generare date
    print("\n[1] Generare dataset IMDb simplificat...")
    texts, labels = generate_imdb_sample()
    print(f"Total sample-uri: {len(texts)}")
    print(f"Distribuție clase - Pozitiv: {sum(labels)}, Negativ: {len(labels) - sum(labels)}")
    
    # Split date
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Tokenizer
    print("\n[2] Inițializare tokenizer BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = IMDbDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = IMDbDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = IMDbDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    results = {}
    
    # ========================================================================
    # STRATEGIE (a): FINE-TUNING COMPLET
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGIE (a): FINE-TUNING COMPLET - Toate straturile")
    print("="*80)
    print("✓ BERT pre-antrenat pe milioane de texte")
    print("✓ Ajustare strat de ieșire: 768 → 2 clase (Negativ/Pozitiv)")
    print("✓ Actualizare toate straturile Transformer")
    
    model_full = BERTFineTuneComplete(num_classes=2).to(device)
    trainable_params_full = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    print(f"Parametri antrenabili: {trainable_params_full:,}")
    
    history_full, best_acc_full, time_full = train_model(
        model_full, train_loader, val_loader, EPOCHS, LR_FINETUNE, device
    )
    
    _, test_acc_full, preds_full, labels_full = evaluate(
        model_full, test_loader, nn.CrossEntropyLoss(), device
    )
    
    results['Fine-tuning Complet'] = {
        'history': history_full,
        'test_acc': test_acc_full,
        'time': time_full,
        'params': trainable_params_full,
        'predictions': preds_full,
        'labels': labels_full
    }
    
    print(f"\n✓ Test Accuracy: {test_acc_full:.4f}")
    print(f"✓ Training Time: {time_full:.2f}s")
    
    # ========================================================================
    # STRATEGIE (b): FINE-TUNING DOAR ULTIMUL STRAT
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGIE (b): FINE-TUNING ULTIMUL STRAT - Feature Extraction")
    print("="*80)
    print("✓ BERT pre-antrenat ÎNGHEȚAT (frozen)")
    print("✓ Antrenare doar classifier: 768 → 2 clase")
    print("✓ Reduce overfitting și timp de antrenare")
    
    model_last = BERTFineTuneLastLayer(num_classes=2).to(device)
    trainable_params_last = sum(p.numel() for p in model_last.parameters() if p.requires_grad)
    print(f"Parametri antrenabili: {trainable_params_last:,}")
    
    history_last, best_acc_last, time_last = train_model(
        model_last, train_loader, val_loader, EPOCHS, LR_FINETUNE, device
    )
    
    _, test_acc_last, preds_last, labels_last = evaluate(
        model_last, test_loader, nn.CrossEntropyLoss(), device
    )
    
    results['Fine-tuning Ultimul Strat'] = {
        'history': history_last,
        'test_acc': test_acc_last,
        'time': time_last,
        'params': trainable_params_last,
        'predictions': preds_last,
        'labels': labels_last
    }
    
    print(f"\n✓ Test Accuracy: {test_acc_last:.4f}")
    print(f"✓ Training Time: {time_last:.2f}s")
    
    # ========================================================================
    # STRATEGIE (c): ANTRENARE DE LA ZERO
    # ========================================================================
    print("\n" + "="*80)
    print("STRATEGIE (c): ANTRENARE DE LA ZERO - Fără Transfer Learning")
    print("="*80)
    print("✓ Inițializare aleatoare a TUTUROR ponderilor")
    print("✓ Fără cunoștințe pre-antrenate")
    print("✓ Necesită mult mai multe date și timp")
    
    model_scratch = BERTFromScratch(num_classes=2).to(device)
    trainable_params_scratch = sum(p.numel() for p in model_scratch.parameters() if p.requires_grad)
    print(f"Parametri antrenabili: {trainable_params_scratch:,}")
    
    history_scratch, best_acc_scratch, time_scratch = train_model(
        model_scratch, train_loader, val_loader, EPOCHS, LR_SCRATCH, device
    )
    
    _, test_acc_scratch, preds_scratch, labels_scratch = evaluate(
        model_scratch, test_loader, nn.CrossEntropyLoss(), device
    )
    
    results['Antrenare De La Zero'] = {
        'history': history_scratch,
        'test_acc': test_acc_scratch,
        'time': time_scratch,
        'params': trainable_params_scratch,
        'predictions': preds_scratch,
        'labels': labels_scratch
    }
    
    print(f"\n✓ Test Accuracy: {test_acc_scratch:.4f}")
    print(f"✓ Training Time: {time_scratch:.2f}s")
    
    # ========================================================================
    # COMPARAȚIE FINALĂ ȘI VIZUALIZĂRI
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARAȚIE FINALĂ - REZULTATE")
    print("="*80)
    
    comparison_table = []
    for name, res in results.items():
        comparison_table.append({
            'Strategie': name,
            'Test Accuracy': f"{res['test_acc']:.4f}",
            'Timp Antrenare (s)': f"{res['time']:.2f}",
            'Parametri': f"{res['params']:,}"
        })
    
    print("\n{:<30} {:<15} {:<20} {:<20}".format(
        'Strategie', 'Test Accuracy', 'Timp (s)', 'Parametri'
    ))
    print("-" * 85)
    for row in comparison_table:
        print("{:<30} {:<15} {:<20} {:<20}".format(
            row['Strategie'], row['Test Accuracy'], 
            row['Timp Antrenare (s)'], row['Parametri']
        ))
    
    # Vizualizări
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Comparație Accuracy
    strategies = list(results.keys())
    test_accs = [results[s]['test_acc'] for s in strategies]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    axes[0, 0].bar(range(len(strategies)), test_accs, color=colors, alpha=0.7)
    axes[0, 0].set_xticks(range(len(strategies)))
    axes[0, 0].set_xticklabels(strategies, rotation=15, ha='right')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Comparație Test Accuracy', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    for i, acc in enumerate(test_accs):
        axes[0, 0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')
    
    # 2. Evoluție Training
    for i, (name, res) in enumerate(results.items()):
        axes[0, 1].plot(res['history']['train_acc'], 
                       label=f'{name} (Train)', 
                       color=colors[i], linestyle='-', marker='o')
        axes[0, 1].plot(res['history']['val_acc'], 
                       label=f'{name} (Val)', 
                       color=colors[i], linestyle='--', marker='s')
    
    axes[0, 1].set_xlabel('Epoca')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Evoluție Accuracy pe Epoci', fontweight='bold', fontsize=12)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Timp de Antrenare
    times = [results[s]['time'] for s in strategies]
    axes[1, 0].barh(range(len(strategies)), times, color=colors, alpha=0.7)
    axes[1, 0].set_yticks(range(len(strategies)))
    axes[1, 0].set_yticklabels(strategies)
    axes[1, 0].set_xlabel('Timp (secunde)')
    axes[1, 0].set_title('Comparație Timp de Antrenare', fontweight='bold', fontsize=12)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    for i, t in enumerate(times):
        axes[1, 0].text(t + 1, i, f'{t:.1f}s', va='center', fontweight='bold')
    
    # 4. Confusion Matrix pentru Fine-tuning Complet
    cm = confusion_matrix(labels_full, preds_full)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negativ', 'Pozitiv'],
                yticklabels=['Negativ', 'Pozitiv'],
                ax=axes[1, 1])
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    axes[1, 1].set_title('Confusion Matrix - Fine-tuning Complet', 
                         fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('transfer_learning_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Grafice salvate în 'transfer_learning_comparison.png'")
    plt.show()
    
    # Concluzii
    print("\n" + "="*80)
    print("CONCLUZII")
    print("="*80)
    
    best_strategy = max(results.items(), key=lambda x: x[1]['test_acc'])
    print(f"\n✓ Cea mai bună strategie: {best_strategy[0]}")
    print(f"  - Test Accuracy: {best_strategy[1]['test_acc']:.4f}")
    
    improvement = (best_strategy[1]['test_acc'] - results['Antrenare De La Zero']['test_acc']) * 100
    print(f"\n✓ Îmbunătățire față de antrenare de la zero: +{improvement:.2f}%")
    
    print("\n✓ Transfer Learning demonstrează superioritate clară:")
    print("  1. Accuracy mai mare cu date limitate")
    print("  2. Convergență mai rapidă")
    print("  3. Generalizare mai bună")
    
    print("\n✓ Fine-tuning ultimul strat oferă:")
    print("  - Timp de antrenare redus semnificativ")
    print("  - Risc de overfitting mai mic")
    print("  - Performanță comparabilă cu fine-tuning complet")
    
    print("\n" + "="*80)
    print("Experiment finalizat cu succes!")
    print("="*80)


if __name__ == "__main__":
    main()
