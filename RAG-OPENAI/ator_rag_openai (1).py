# ATOR RAG System cu OpenAI - VERSIUNEA CEA MAI BUNĂ
# Răspunsuri excelente în română, rapide, precise
# Best Answers in Romanian, Fast, Accurate

import os
from typing import List
from dataclasses import dataclass
from glob import glob

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# =====================================================================
# 🔒 SECURITATE API KEY - API KEY SECURITY
# =====================================================================
# NU MAI EXISTĂ API KEY HARDCODAT ÎN COD!
# NO MORE HARDCODED API KEY IN CODE!
#
# API key-ul se încarcă automat din:
# The API key is automatically loaded from:
#
# 1. Variabilă de mediu (recomandat / recommended):
#    export OPENAI_API_KEY="sk-..."
#
# 2. Fișier .env (pentru dezvoltare locală / for local development):
#    OPENAI_API_KEY=sk-...
#
# IMPORTANT: .env este în .gitignore și NU se urcă pe GitHub!
# IMPORTANT: .env is in .gitignore and is NOT uploaded to GitHub!
# =====================================================================

# Încarcă variabilele din .env dacă există
# Load variables from .env if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv nu este instalat. Instalează cu: pip install python-dotenv")
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")

# System Configuration
@dataclass
class ATORRAGConfig:
    chunk_size: int = 600
    chunk_overlap: int = 100
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    openai_model: str = "gpt-3.5-turbo"  # Sau "gpt-4" pentru răspunsuri și mai bune
    top_k_retrieval: int = 4
    docs_path: str = "./ator_documents"
    vector_store_path: str = "./ator_vector_store"

class ATORDocumentProcessor:
    """
    Procesează documente text pentru sistemul RAG
    Processes text documents for the RAG system
    """
    def __init__(self, config: ATORRAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ".", " "]
        )

    def process_directory(self, dir_path: str) -> List[Document]:
        """
        Procesează toate fișierele .txt dintr-un director
        Process all .txt files from a directory
        """
        print(f"📂 Încărcare documente din / Loading documents from: {dir_path}")
        
        all_docs = []
        txt_files = glob(os.path.join(dir_path, "*.txt"))
        
        if not txt_files:
            print(f"⚠️  Nu s-au găsit fișiere .txt în {dir_path}")
            print(f"⚠️  No .txt files found in {dir_path}")
        
        for file_path in txt_files:
            try:
                print(f"  📄 {os.path.basename(file_path)}")
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
                chunks = self.text_splitter.split_documents(documents)
                all_docs.extend(chunks)
            except Exception as e:
                print(f"  ❌ Eroare la / Error at {file_path}: {str(e)}")
        
        print(f"✅ Total {len(all_docs)} chunk-uri create din / chunks created from {len(txt_files)} fișiere / files")
        return all_docs

class ATORKnowledgeRAGSystem:
    """
    Sistem RAG complet pentru ATOR Banatul de Munte
    Complete RAG system for ATOR Banatul de Munte
    """
    def __init__(self, config: ATORRAGConfig = ATORRAGConfig()):
        self.config = config
        self.vector_db = None
        self.qa_chain = None
        print(f"🚀 Inițializare ATOR RAG System cu OpenAI")
        print(f"🚀 Initializing ATOR RAG System with OpenAI")
        
        # 🔒 VERIFICARE API KEY - API KEY CHECK
        self._check_api_key()

    def _check_api_key(self):
        """
        Verifică dacă API key-ul OpenAI este disponibil
        Checks if OpenAI API key is available
        """
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("\n" + "="*80)
            print("❌ EROARE: OPENAI_API_KEY nu este setat!")
            print("❌ ERROR: OPENAI_API_KEY is not set!")
            print("="*80)
            print("\n📋 METODE DE SETARE / SETUP METHODS:")
            print("\n1️⃣  Variabilă de mediu (Linux/Mac):")
            print('   export OPENAI_API_KEY="sk-..."')
            print("\n   Variabilă de mediu (Windows CMD):")
            print('   set OPENAI_API_KEY=sk-...')
            print("\n   Variabilă de mediu (Windows PowerShell):")
            print('   $env:OPENAI_API_KEY="sk-..."')
            print("\n2️⃣  Fișier .env (crează fișier .env în folder):")
            print('   OPENAI_API_KEY=sk-...')
            print('   (apoi instalează: pip install python-dotenv)')
            print("\n3️⃣  Hardcodat în cod (NU RECOMANDAT pentru GitHub!):")
            print('   os.environ["OPENAI_API_KEY"] = "sk-..."')
            print("\n" + "="*80)
            print("⚠️  Sistemul va continua, dar API calls vor eșua!")
            print("⚠️  System will continue, but API calls will fail!")
            print("="*80 + "\n")
        else:
            # Afișează doar primele și ultimele caractere pentru confirmare
            masked_key = f"{api_key[:7]}...{api_key[-4:]}" if len(api_key) > 11 else "sk-***"
            print(f"✅ API Key găsit / found: {masked_key}")

    def setup_llm(self):
        """
        Configurare OpenAI LLM
        Configure OpenAI LLM
        """
        print(f"🧠 Încărcare OpenAI / Loading OpenAI: {self.config.openai_model}...")
        
        # API key-ul este luat automat din environment de către ChatOpenAI
        # The API key is automatically taken from environment by ChatOpenAI
        return ChatOpenAI(
            model=self.config.openai_model,
            temperature=0.3,  # Balans între creativitate și acuratețe
            max_tokens=500
        )

    def setup(self):
        """
        Inițializare completă a sistemului RAG
        Complete RAG system initialization
        """
        
        # Verifică dacă există documente
        if not os.path.exists(self.config.docs_path):
            print(f"❌ Eroare: Directorul {self.config.docs_path} nu există!")
            print(f"❌ Error: Directory {self.config.docs_path} does not exist!")
            print("👉 Asigură-te că ai folderul ator_documents/ cu fișierele .txt")
            print("👉 Make sure you have the ator_documents/ folder with .txt files")
            return False
        
        # 1. Procesează documentele
        processor = ATORDocumentProcessor(self.config)
        docs = processor.process_directory(self.config.docs_path)
        
        if not docs:
            print("❌ Nu s-au găsit documente pentru procesare!")
            print("❌ No documents found for processing!")
            return False
        
        # 2. Setup Embeddings (local pentru a economisi bani)
        print("📤 Inițializare embeddings multilingve / Initializing multilingual embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embeddings_model
        )
        
        # 3. Creare Vector Store
        print("🗄️  Construire Vector Database / Building Vector Database...")
        self.vector_db = FAISS.from_documents(docs, embeddings)
        
        # Salvare opțională
        if self.config.vector_store_path:
            os.makedirs(self.config.vector_store_path, exist_ok=True)
            self.vector_db.save_local(self.config.vector_store_path)
            print(f"💾 Vector DB salvat în / saved in: {self.config.vector_store_path}")
        
        # 4. Setup OpenAI LLM
        llm = self.setup_llm()
        
        # Prompt optimizat pentru română
        template = """Ești asistentul virtual al ATOR Banatul de Munte, o organizație de tineret ortodox din Caraș-Severin.

Răspunde la întrebare bazându-te pe contextul de mai jos. Dacă întrebarea nu are răspuns în context, spune politicos că nu ai această informație.

Context:
{context}

Istoric conversație:
{chat_history}

Întrebare: {question}

Răspuns (în română, prietenos, profesional):"""

        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_db.as_retriever(
                search_kwargs={"k": self.config.top_k_retrieval}
            ),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=False
        )
        
        print("✅ Sistem RAG cu OpenAI gata de utilizare!")
        print("✅ RAG System with OpenAI ready to use!")
        return True

    def chat(self, question: str):
        """
        Conversație cu sistemul RAG
        Conversation with RAG system
        """
        if not self.qa_chain:
            return "❌ Sistemul nu este inițializat. Rulează setup() mai întâi."
        
        try:
            print(f"\n❓ Întrebare / Question: {question}")
            print("🤔 Procesare / Processing...")
            
            result = self.qa_chain.invoke({"question": question})
            answer = result['answer']
            
            print(f"🤖 Răspuns / Answer: {answer}")
            
            # Afișează surse (optional)
            if result.get('source_documents'):
                print(f"\n📚 Surse folosite / Sources used: {len(result['source_documents'])} documente / documents")
            
            return answer
            
        except Exception as e:
            error_msg = f"❌ Eroare la procesare / Processing error: {str(e)}"
            print(error_msg)
            return error_msg

    def batch_test(self, questions: List[str]):
        """
        Testează mai multe întrebări deodată
        Test multiple questions at once
        """
        print(f"\n{'='*80}")
        print(f"🧪 TEST BATCH - {len(questions)} întrebări / questions")
        print(f"{'='*80}\n")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}]")
            self.chat(question)
            print(f"{'-'*80}")

def demo_interactive_mode():
    """
    Mod interactiv în consolă
    Interactive console mode
    """
    print("\n" + "="*80)
    print("🎯 ATOR CHATBOT cu OpenAI - MOD INTERACTIV")
    print("🎯 ATOR CHATBOT with OpenAI - INTERACTIVE MODE")
    print("="*80)
    print("Întreabă orice despre ATOR Banatul de Munte!")
    print("Ask anything about ATOR Banatul de Munte!")
    print("\nComenzi speciale / Special commands:")
    print("  - 'exit' sau 'quit' = ieșire / exit")
    print("  - 'clear' = șterge istoricul conversației / clear conversation history")
    print("  - 'test' = rulează întrebări de test / run test questions")
    print("="*80 + "\n")
    
    # Inițializare sistem
    rag_system = ATORKnowledgeRAGSystem()
    
    if not rag_system.setup():
        print("❌ Inițializare eșuată! / Initialization failed!")
        return
    
    # Întrebări de test predefinite
    test_questions = [
        "Cum mă pot înscrie ca voluntar la ATOR?",
        "Care sunt filialele ATOR?",
        "Primesc certificat de voluntariat?",
        "Ce beneficii am dacă devin voluntar?",
        "Costă ceva să fiu voluntar?",
        "Ce vârstă trebuie să am pentru voluntariat?",
        "Unde este sediul central ATOR?"
    ]
    
    print(f"\n💰 COST ESTIMAT per răspuns / ESTIMATED COST per answer: ~$0.002-0.005 (aprox. 1-2 bani)")
    print(f"📊 10 întrebări / questions = ~$0.02-0.05 (aprox. 10-25 bani)\n")
    
    # Loop interactiv
    while True:
        try:
            user_input = input("\n💬 Tu / You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'iesire']:
                print("\n👋 La revedere! / Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                rag_system.qa_chain.memory.clear()
                print("🗑️  Istoric conversație șters! / Conversation history cleared!")
                continue
            
            if user_input.lower() == 'test':
                rag_system.batch_test(test_questions)
                continue
            
            # Răspunde la întrebare
            rag_system.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 La revedere! / Goodbye!")
            break
        except Exception as e:
            print(f"❌ Eroare / Error: {str(e)}")

if __name__ == "__main__":
    # Verifică dacă există documentele
    if not os.path.exists("ator_documents"):
        print("❌ EROARE: Directorul 'ator_documents' nu există!")
        print("❌ ERROR: Directory 'ator_documents' does not exist!")
        print("\n📋 PAȘI NECESARI / REQUIRED STEPS:")
        print("1. Asigură-te că ai folderul ator_documents/ cu fișierele .txt")
        print("   Make sure you have the ator_documents/ folder with .txt files")
        print("2. Setează OPENAI_API_KEY (vezi instrucțiuni mai sus)")
        print("   Set OPENAI_API_KEY (see instructions above)")
        print("3. Rulează: python ator_rag_openai.py")
        print("\n" + "="*80)
    else:
        # Pornește modul interactiv
        demo_interactive_mode()
