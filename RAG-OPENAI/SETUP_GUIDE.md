# 🚀 Ghid Rapid de Setup

Acest ghid te ajută să configurezi proiectul în 5 minute!

---

## ⚡ Setup Rapid (TL;DR)

```bash
# 1. Clonează repo
git clone https://github.com/username/ator-rag-chatbot.git
cd ator-rag-chatbot

# 2. Environment virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# SAU: venv\Scripts\activate  # Windows

# 3. Instalează
pip install -r requirements.txt

# 4. Configurare API Key
cp .env.example .env
nano .env  # Adaugă: OPENAI_API_KEY=sk-your-key

# 5. Adaugă documente
# Pune fișierele .txt în ator_documents/

# 6. Rulează!
python ator_rag_openai.py
```

---

## 📋 Checklist Complet

### ✅ Înainte de a începe

- [ ] Python 3.8+ instalat
- [ ] Git instalat
- [ ] Cont OpenAI cu API key
- [ ] 2GB+ RAM disponibil

### ✅ Setup inițial

- [ ] Repository clonat local
- [ ] Environment virtual creat și activat
- [ ] Dependențele instalate (`pip install -r requirements.txt`)
- [ ] Fișier `.env` creat din `.env.example`
- [ ] API key OpenAI adăugat în `.env`
- [ ] Verificat că `.env` e în `.gitignore`

### ✅ Documente

- [ ] Folder `ator_documents/` creat
- [ ] Fișiere `.txt` adăugate (minim 1)
- [ ] Verificat encoding UTF-8 pentru română
- [ ] Testat că se încarcă corect

### ✅ Prima rulare

- [ ] `python ator_rag_openai.py` rulează fără erori
- [ ] Mesaj: "✅ API Key găsit"
- [ ] Mesaj: "✅ Sistem RAG gata de utilizare"
- [ ] Test cu întrebare simplă funcționează

---

## 🔧 Comenzi Utile

### Verificare instalare

```bash
# Verifică versiune Python
python --version  # Ar trebui 3.8+

# Verifică pip
pip --version

# Verifică toate package-urile
pip list
```

### Debugging

```bash
# Test API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('Key exists:', bool(os.getenv('OPENAI_API_KEY')))"

# Verifică documente
ls -la ator_documents/

# Vezi ce e în Git
git status
git ls-files | grep .env  # Nu ar trebui să apară .env!
```

### Actualizare

```bash
# Update package-uri
pip install --upgrade -r requirements.txt

# Reinstalare curată
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

---

## ⚠️ Troubleshooting Rapid

### "ModuleNotFoundError: No module named 'langchain'"
```bash
pip install -r requirements.txt
```

### "OPENAI_API_KEY is not set"
```bash
# Verifică .env există
ls -la .env

# Verifică conținut
cat .env

# Ar trebui: OPENAI_API_KEY=sk-...
```

### "No documents found"
```bash
# Verifică folder
ls ator_documents/

# Creează folder dacă lipsește
mkdir ator_documents

# Adaugă un fișier test
echo "Test document pentru ATOR" > ator_documents/test.txt
```

### Import error pentru sentence-transformers
```bash
# Reinstalează explicit
pip install sentence-transformers --upgrade
```

### Probleme cu encoding la documente
```bash
# Convertește fișier la UTF-8
iconv -f ISO-8859-1 -t UTF-8 input.txt > output.txt
```

---

## 🎯 Test Final

După setup, rulează acest test:

```bash
python -c "
from ator_rag_openai import ATORKnowledgeRAGSystem
rag = ATORKnowledgeRAGSystem()
if rag.setup():
    print('✅ TOTUL FUNCȚIONEAZĂ!')
    response = rag.chat('Test întrebare')
    print(f'Răspuns primit: {len(response)} caractere')
else:
    print('❌ Ceva nu funcționează')
"
```

Ar trebui să vezi:
```
🚀 Inițializare ATOR RAG System cu OpenAI
✅ API Key găsit: sk-proj...xyz
📂 Încărcare documente din: ./ator_documents
...
✅ Sistem RAG cu OpenAI gata de utilizare!
✅ TOTUL FUNCȚIONEAZĂ!
```

---

## 📚 Next Steps

După setup:

1. **Citește README.md** pentru documentație completă
2. **Citește SECURITY.md** pentru protecția API key-ului
3. **Adaugă documente** în `ator_documents/`
4. **Testează** cu întrebări reale
5. **Customizează** parametrii în cod dacă e necesar

---

## 💡 Tips & Tricks

### Salvare conversații
```python
# În Python script
answers = []
questions = ["Q1", "Q2", "Q3"]
for q in questions:
    answer = rag.chat(q)
    answers.append({"q": q, "a": answer})

import json
with open('conversation_log.json', 'w') as f:
    json.dump(answers, f, indent=2, ensure_ascii=False)
```

### Benchmark viteza
```python
import time
start = time.time()
rag.chat("Test question")
print(f"Timp răspuns: {time.time() - start:.2f}s")
```

### Cache vector store
```python
# Vector store se salvează automat în ator_vector_store/
# Pentru a-l reîncărca rapid:
# (Add în cod: load vector store instead of rebuilding)
```

---

**🎉 Gata! Acum ai proiectul functional. Distrează-te cu RAG-ul!**

Pentru probleme: deschide un Issue pe GitHub sau verifică SECURITY.md și README.md pentru mai multe detalii.
