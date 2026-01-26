# 📝 Text Sentiment Analysis with BERT

> BERT-based Sentiment Classification using Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-Latest-yellow)](https://huggingface.co/transformers)

## 📋 Overview

A sentiment analysis system built using transfer learning with BERT (Bidirectional Encoder Representations from Transformers). The model fine-tunes pre-trained BERT weights for domain-specific sentiment classification tasks.

## 🎯 Features

- **Fine-tuned BERT** - Leverages pre-trained language understanding
- **Custom tokenization pipeline** - Handles text preprocessing
- **Multi-class sentiment** - Positive, Negative, Neutral classification
- **Evaluation metrics** - Accuracy, Precision, Recall, F1-Score
- **Production-ready inference** - Fast prediction pipeline

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | PyTorch |
| Model | BERT (bert-base-uncased) |
| Library | HuggingFace Transformers |
| Tokenizer | BertTokenizer |

## 📁 Project Structure

```
Transfer_Learning_Text/
├── transfer_learning_text.py  # Main training/inference script
├── dataset.py                 # Data loading utilities
├── model/                     # Saved model checkpoints
└── README.md
```

## 🚀 Quick Start

```python
from transfer_learning_text import SentimentClassifier

# Load model
classifier = SentimentClassifier()
classifier.load_model('model/bert_sentiment.pt')

# Predict sentiment
text = "This product is absolutely amazing!"
prediction = classifier.predict(text)
print(f"Sentiment: {prediction}")  # Output: Positive
```

## 📊 Model Architecture

```
Input Text
    │
    ▼
┌─────────────────┐
│  BertTokenizer  │
│  (Tokenize)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BERT Model     │
│  (12 layers)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  [CLS] Token    │
│  Embedding      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Classification │
│  Head (Dense)   │
└────────┬────────┘
         │
         ▼
   Sentiment Class
```

## 🔧 Training

```bash
python transfer_learning_text.py \
    --train \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --max_length 128
```

### Training Configuration
```python
config = {
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01
}
```

## 📈 Results

| Metric | Score |
|--------|-------|
| Accuracy | 92% |
| Precision | 91% |
| Recall | 90% |
| F1-Score | 90.5% |

## 🔑 Key Components

### Fine-tuning BERT
```python
class SentimentClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(self.dropout(pooled_output))
```

## 📝 Key Learnings

- BERT's bidirectional context improves sentiment understanding
- Fine-tuning last layers is often sufficient
- Learning rate warmup stabilizes training
- Class imbalance requires weighted loss functions

## 📚 References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Transfer Learning in NLP](https://ruder.io/transfer-learning/)

## 📫 Contact

- **LinkedIn:** [linkedin.com/in/paul-sebastian-kradyel](https://linkedin.com/in/paul-sebastian-kradyel)
- **Email:** paulsebastiankradyel@gmail.com
- **GitHub:** [github.com/sebikradyel1-svg](https://github.com/sebikradyel1-svg)
