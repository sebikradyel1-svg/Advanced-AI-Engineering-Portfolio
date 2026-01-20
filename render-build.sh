#!/usr/bin/env bash
# Render Build Script - Pre-downloads ML models during build phase
# This caches models so they don't need to download at runtime!

set -e

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🤖 Pre-downloading embedding model (all-MiniLM-L6-v2)..."
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading sentence-transformers/all-MiniLM-L6-v2...')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('✅ Embedding model cached!')
"

echo "🧠 Pre-downloading LLM (FLAN-T5-base)..."
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print('Downloading google/flan-t5-base...')
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
print('✅ LLM model cached!')
"

echo "✅ Build complete! Models are cached and ready."
