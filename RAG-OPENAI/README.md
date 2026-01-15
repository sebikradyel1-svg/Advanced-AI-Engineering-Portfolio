# 🤖 ATOR RAG Chatbot System

**Sistem RAG inteligent pentru ATOR Banatul de Munte** | **Intelligent RAG System for ATOR Banatul de Munte**

Un chatbot conversațional bazat pe tehnologia Retrieval-Augmented Generation (RAG) care răspunde la întrebări despre ATOR Banatul de Munte folosind OpenAI și documentație locală.

A conversational chatbot based on Retrieval-Augmented Generation (RAG) technology that answers questions about ATOR Banatul de Munte using OpenAI and local documentation.

---

## 🌟 Caracteristici / Features

- ✅ **RAG cu OpenAI** - Răspunsuri precise bazate pe documentele tale / Accurate answers based on your documents
- 🇷🇴 **Suport multilingv** - Optimizat pentru limba română / Optimized for Romanian language
- 💾 **Vector Store local** - Embeddings gratuite cu HuggingFace / Free embeddings with HuggingFace
- 🔒 **Securitate API Key** - Fără chei hardcodate în cod / No hardcoded keys in code
- 💬 **Mod interactiv** - Chat în consolă cu memorie conversațională / Console chat with conversation memory
- 📚 **Document processing** - Procesare automată documente .txt / Automatic .txt document processing

---

## 📋 Cerințe / Requirements

- Python 3.8+
- OpenAI API Key
- 2GB+ RAM (pentru embeddings)
- Internet connection

---

## 🚀 Instalare rapidă / Quick Installation

### 1️⃣ Clonează repository-ul / Clone the repository

```bash
git clone https://github.com/username/ator-rag-chatbot.git
cd ator-rag-chatbot
```

### 2️⃣ Creează environment virtual (recomandat) / Create virtual environment (recommended)

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Instalează dependențele / Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configurează API Key-ul / Configure API Key

**Opțiunea A: Folosind fișier .env (RECOMANDAT) / Using .env file (RECOMMENDED)**

```bash
# Copiază template-ul / Copy template
cp .env.example .env

# Editează .env și adaugă API key-ul tău
# Edit .env and add your API key
nano .env  # sau orice editor
```

Conținut `.env`:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Opțiunea B: Variabilă de mediu / Environment variable**

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-your-actual-api-key-here"

# Windows CMD
set OPENAI_API_KEY=sk-your-actual-api-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
```

### 5️⃣ Pregătește documentele / Prepare documents

Creează folderul cu documente și adaugă fișiere `.txt`:

```bash
mkdir ator_documents
# Adaugă fișierele tale .txt în ator_documents/
```

**Structura recomandată / Recommended structure:**
```
ator_documents/
├── despre_ator.txt
├── voluntariat.txt
├── activitati.txt
├── filiale.txt
└── ...
```

### 6️⃣ Rulează aplicația / Run the application

```bash
python ator_rag_openai.py
```

---

## 💡 Utilizare / Usage

### Mod interactiv / Interactive Mode

După pornire, vei vedea:

```
🎯 ATOR CHATBOT cu OpenAI - MOD INTERACTIV
============================================================
Întreabă orice despre ATOR Banatul de Munte!

Comenzi speciale:
  - 'exit' sau 'quit' = ieșire
  - 'clear' = șterge istoricul conversației
  - 'test' = rulează întrebări de test
============================================================

💬 Tu:
```

### Exemple de întrebări / Example questions

```
💬 Tu: Cum mă pot înscrie ca voluntar la ATOR?
🤖 Răspuns: Pentru a te înscrie ca voluntar la ATOR...

💬 Tu: Care sunt filialele ATOR?
🤖 Răspuns: ATOR Banatul de Munte are filiale în...

💬 Tu: Primesc certificat de voluntariat?
🤖 Răspuns: Da, vei primi un certificat oficial...
```

### Comenzi speciale / Special Commands

- `test` - Rulează un set de întrebări predefinite
- `clear` - Șterge istoricul conversației
- `exit` / `quit` - Închide aplicația

---

## 🔧 Configurare avansată / Advanced Configuration

### Personalizare parametri / Customizing parameters

Editează clasa `ATORRAGConfig` în `ator_rag_openai.py`:

```python
@dataclass
class ATORRAGConfig:
    chunk_size: int = 600          # Mărimea chunk-urilor de text
    chunk_overlap: int = 100        # Overlap între chunk-uri
    openai_model: str = "gpt-4"    # Sau "gpt-3.5-turbo"
    top_k_retrieval: int = 4        # Număr documente recuperate
```

### Folosire ca librărie / Using as library

```python
from ator_rag_openai import ATORKnowledgeRAGSystem

# Inițializare
rag = ATORKnowledgeRAGSystem()
rag.setup()

# Întrebare unică
answer = rag.chat("Cum mă pot înscrie ca voluntar?")
print(answer)

# Batch testing
questions = [
    "Care sunt filialele ATOR?",
    "Primesc certificat de voluntariat?",
]
rag.batch_test(questions)
```

---

## 🔒 Securitate API Key / API Key Security

### ✅ CE FACE codul CORECT / What the code does CORRECTLY

1. **NU există API key hardcodat** - Codul nu conține niciun API key în el
2. **Încarcă din environment** - API key-ul se ia automat din variabile de mediu
3. **Suportă .env** - Citește automat fișierul `.env` dacă există
4. **Este în .gitignore** - Fișierul `.env` nu se urcă pe GitHub

### 🔐 Cum funcționează protecția / How protection works

**Înainte (GREȘIT ❌):**
```python
os.environ["OPENAI_API_KEY"] = 'sk-actual-key-here'  # PERICOL!
```

**Acum (CORECT ✅):**
```python
# Cod fără API key hardcodat
api_key = os.getenv("OPENAI_API_KEY")  # Se citește din environment
```

**Flow de securitate:**

1. Tu creezi fișier `.env` LOCAL pe calculatorul tău
2. Adaugi API key-ul în `.env`
3. `.env` este în `.gitignore` → NU se urcă pe GitHub
4. Codul citește din `.env` când rulează
5. Alții clonează repo-ul → nu au API key-ul tău
6. Ei își creează propriul `.env` cu propriul key

---

## 💰 Costuri estimate / Estimated Costs

Cu GPT-3.5-Turbo:
- 1 întrebare simplă: ~$0.002-0.005 (aprox. 1-2 bani)
- 10 întrebări: ~$0.02-0.05 (aprox. 10-25 bani)
- 100 întrebări: ~$0.20-0.50 (aprox. 1-2.5 lei)

Cu GPT-4:
- Aproximativ 10-15x mai scump decât GPT-3.5-Turbo
- Răspunsuri de calitate superioară

---

## 📁 Structura Proiectului / Project Structure

```
ator-rag-chatbot/
│
├── ator_rag_openai.py          # Main application
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── LICENSE                      # License file
│
├── ator_documents/              # Your .txt documents (not in git)
│   ├── despre_ator.txt
│   ├── voluntariat.txt
│   └── ...
│
└── ator_vector_store/           # FAISS vector store (auto-generated)
    ├── index.faiss
    └── index.pkl
```

---

## 🐛 Troubleshooting

### Eroare: "OPENAI_API_KEY is not set"

**Soluție:**
1. Verifică dacă fișierul `.env` există
2. Verifică dacă conține `OPENAI_API_KEY=sk-...`
3. Verifică dacă ai instalat `python-dotenv`: `pip install python-dotenv`

### Eroare: "No documents found"

**Soluție:**
1. Verifică dacă folderul `ator_documents/` există
2. Adaugă fișiere `.txt` în folder
3. Verifică encoding-ul fișierelor (ar trebui să fie UTF-8)

### Eroare: "Rate limit exceeded"

**Soluție:**
- Ai depășit limita OpenAI API
- Așteaptă câteva minute
- Verifică contul OpenAI pentru limita ta

### Embeddings sunt lente la prima rulare

**Normal!** HuggingFace descarcă modelul de embeddings (aproximativ 500MB).
După prima rulare, totul va fi mai rapid.

---

## 🤝 Contribuții / Contributing

Contribuțiile sunt binevenite! / Contributions are welcome!

1. Fork repository-ul
2. Creează branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Deschide Pull Request

---

## 📝 TODO / Roadmap

- [ ] Suport pentru fișiere PDF și DOCX
- [ ] Interface web cu Streamlit/Gradio
- [ ] Cache pentru răspunsuri frecvente
- [ ] Suport pentru imagini în documente
- [ ] Export conversații în format JSON
- [ ] Integrare cu baze de date SQL
- [ ] API REST pentru integrări externe

---

## 📄 Licență / License

MIT License - vezi fișierul [LICENSE](LICENSE)

---

## 👤 Autor / Author

**Sebastian** - AI Engineer & ML Specialist
- Organizație: ATOR Banatul de Munte
- Region: Caraș-Severin, România

---

## 🙏 Mulțumiri / Acknowledgments

- [LangChain](https://www.langchain.com/) - Framework pentru RAG
- [OpenAI](https://openai.com/) - GPT models
- [HuggingFace](https://huggingface.co/) - Multilingual embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

---

## 📞 Contact & Support

Pentru întrebări sau probleme:
- Deschide un [GitHub Issue](https://github.com/username/ator-rag-chatbot/issues)
- Email: your-email@example.com

---

**⭐ Dacă acest proiect te-a ajutat, lasă un star pe GitHub! / If this project helped you, leave a star on GitHub!**
