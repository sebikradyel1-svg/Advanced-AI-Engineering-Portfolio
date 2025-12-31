
# Sistem RAG pentru Gestionarea Cunoștințelor Interne HR
# Importuri necesare

import os
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np


# LangChain componente (actualizate la versiunea nouă)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate


# Transformers pentru LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# Configurație sistem
@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    llm_model: str = "google/flan-t5-base"
    top_k_retrieval: int = 5
    docs_path: str = "./company_docs"
    vector_store_path: str = "./vector_store"
    vector_db_type: str = "faiss"


class DocumentProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def load_documents(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        print(f" Încărcat {len(documents)} document(e) din {file_path}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        print(f" Creat {len(chunks)} chunk-uri din documente")
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source_type"] = "hr_policy"
        return chunks


class DensePassageRetriever:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = None
        self.vector_store = None
        
    def initialize_embeddings(self):
        print(" Inițializare model embeddings...")
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embeddings_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print(f"Model embeddings încărcat: {self.config.embeddings_model}")
    
    def create_vector_store(self, chunks: List[Document]):
        print("🔄 Creare bază de date vectorială...")
        if self.config.vector_db_type == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            self.vector_store.save_local(self.config.vector_store_path)
            print(" Index FAISS creat și salvat")
        elif self.config.vector_db_type == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.config.vector_store_path
            )
            self.vector_store.persist()
            print(" Chroma DB creat și persistat")
    
    def load_vector_store(self):
        if os.path.exists(self.config.vector_store_path):
            if self.config.vector_db_type == "faiss":
                self.vector_store = FAISS.load_local(
                    self.config.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            elif self.config.vector_db_type == "chroma":
                self.vector_store = Chroma(
                    persist_directory=self.config.vector_store_path,
                    embedding_function=self.embeddings
                )
            print(" Index vectorial încărcat din stocare")
            return True
        return False


class ConversationalHRAgent:
    def __init__(self, config: RAGConfig, retriever: DensePassageRetriever):
        self.config = config
        self.retriever = retriever
        self.llm = None
        self.chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.prompt_template = """Ești un asistent HR specializat în politicile companiei. 
Folosește DOAR informațiile din contextul furnizat pentru a răspunde la întrebare.
Dacă informația nu se găsește în context, spune explicit "Nu am găsit această informație în documentele disponibile."
Răspunde **doar în limba română**.
Răspunde **scurt și la obiect**, extrăgând doar fraza relevantă din context.
Nu inventa sau presupune informații care nu sunt în context.

Context:
{context}

Istoric conversație:
{chat_history}

Întrebare: {question}

Răspuns:"""
        
    def initialize_llm(self):
        print(" Inițializare LLM...")
        model_name = self.config.llm_model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32, 
            device_map="auto"
        )
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            truncation=True,
            temperature=0.7,
            top_k=50, 
            top_p=0.95,
            do_sample=True
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        print(f" LLM încărcat: {model_name}")

    def create_qa_chain(self):
        print(" Creare lanț conversațional...")
        prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        base_retriever = self.retriever.vector_store.as_retriever(
            search_kwargs={"k": self.config.top_k_retrieval}
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=base_retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
        print("Lanț conversațional RAG creat")

    def query(self, question: str) -> Dict:
        result = self.chain({"question": question})
        return {
            "answer": result["answer"],
            "source_documents": [
                {"content": doc.page_content[:200] + "...", "metadata": doc.metadata}
                for doc in result.get("source_documents", [])[:3]
            ],
            "chat_history": self.get_conversation_history()
        }

    def get_conversation_history(self) -> List[Dict]:
        return [{"type": m.__class__.__name__, "content": m.content} for m in self.memory.chat_memory.messages[-10:]]

    def clear_memory(self):
        self.memory.clear()


class HRKnowledgeRAGSystem:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.document_processor = DocumentProcessor(self.config)
        self.retriever = DensePassageRetriever(self.config)
        self.agent = None
        
    def setup(self, documents_path: str):
        print("="*50)
        print("INIȚIALIZARE SISTEM RAG HR")
        print("="*50)
        print("\n PAS 1: PROCESARE DOCUMENTE")
        documents = self.document_processor.load_documents(documents_path)
        chunks = self.document_processor.split_documents(documents)
        print("\n PAS 2: INIȚIALIZARE EMBEDDINGS")
        self.retriever.initialize_embeddings()
        print("\n PAS 3: INDEXARE VECTORIALĂ")
        if not self.retriever.load_vector_store():
            self.retriever.create_vector_store(chunks)
        print("\n PAS 4: INIȚIALIZARE AGENT CONVERSAȚIONAL")
        self.agent = ConversationalHRAgent(self.config, self.retriever)
        self.agent.initialize_llm()
        self.agent.create_qa_chain()
        print("\n" + "="*50)
        print(" SISTEM RAG FUNCȚIONAL")
        print("="*50)
    
    def chat(self, question: str) -> str:
        response = self.agent.query(question)
        print("\n" + "="*50)
        print(f" ÎNTREBARE: {question}")
        print("-"*50)
        print(f" RĂSPUNS: {response['answer']}")
        if response['source_documents']:
            print("\n SURSE:")
            for i, doc in enumerate(response['source_documents'], 1):
                print(f"  {i}. {doc['content']}")
        print("="*50)
        return response['answer']


def create_sample_policies_file():
    sample_content = """MANUAL DE POLITICI INTERNE - COMPANIA XYZ

CAPITOLUL 1: PROGRAMUL DE LUCRU
1.1. Programul standard de lucru este de luni până vineri, între orele 09:00 și 18:00.
1.2. Pauza de prânz este între 13:00 și 14:00.
1.3. Munca de acasă este permisă maximum 2 zile pe săptămână, cu aprobarea managerului direct.
1.4. Orele suplimentare trebuie aprobate în prealabil de către manager.

CAPITOLUL 2: CONCEDII ȘI ZILE LIBERE
2.1. Angajații au dreptul la 21 de zile lucrătoare de concediu anual.
2.2. Concediul medical este plătit conform legislației în vigoare.
2.3. Concediul de maternitate este de 126 de zile calendaristice.
2.4. Zilele de sărbători legale sunt libere și plătite.
2.5. Concediul fără plată poate fi acordat pentru maximum 30 de zile pe an.

CAPITOLUL 3: BENEFICII ANGAJAȚI
3.1. Asigurare medicală privată pentru toți angajații cu normă întreagă.
3.2. Abonament la sală de fitness sau activități sportive.
3.3. Bonuri de masă în valoare de 30 RON/zi lucrată.
3.4. Budget anual de 2000 RON pentru training și dezvoltare profesională.
3.5. Bonus anual bazat pe performanță, între 10-25% din salariu.
"""
    with open("companyPolicies.txt", "w", encoding="utf-8") as f:
        f.write(sample_content)
    print(" Fișier companyPolicies.txt creat cu succes!")


if __name__ == "__main__":
    if not os.path.exists("companyPolicies.txt"):
        create_sample_policies_file()
    rag_system = HRKnowledgeRAGSystem()
    rag_system.setup("companyPolicies.txt")
    questions = [
        "Câte zile de concediu am pe an?",
        "Care este programul de lucru?",
        "Ce beneficii medicale oferă compania?"
    ]
    for question in questions:
        rag_system.chat(question)
    print("\n💬 MOD INTERACTIV (scrie 'exit' pentru ieșire)")
    while True:
        user_question = input("\nÎntrebare: ")
        if user_question.lower().strip() == "exit":
            break
        rag_system.chat(user_question)
