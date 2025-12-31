
# HR Internal Knowledge Management RAG System
# Required Imports

import os
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Transformers for LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# System Configuration
@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    # Optimized for English
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-base"
    top_k_retrieval: int = 3
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
            separators=["\n\n", "\n", ".", " "]
        )

    def process_file(self, file_path: str) -> List[Document]:
        print(f"📂 Loading document: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

class HRKnowledgeRAGSystem:
    def __init__(self, config: RAGConfig = RAGConfig()):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vector_db = None
        self.qa_chain = None
        print(f"🚀 Initializing RAG System on: {self.device}")

    def setup_llm(self):
        print(f"🧠 Loading LLM: {self.config.llm_model}...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.llm_model)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.1, # Low temperature for factual accuracy
            device=0 if self.device == "cuda" else -1
        )
        return HuggingFacePipeline(pipeline=pipe)

    def setup(self, file_path: str):
        # 1. Process Documents
        processor = DocumentProcessor(self.config)
        docs = processor.process_file(file_path)

        # 2. Setup Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=self.config.embeddings_model)

        # 3. Create Vector Store
        print("📁 Building Vector Database...")
        self.vector_db = FAISS.from_documents(docs, embeddings)

        # 4. Setup LLM and Chain
        llm = self.setup_llm()
        
        # Professional English Prompt
        template = """Answer the question based ONLY on the following context. 
If the answer is not in the context, politely state that you do not have that information.

Context: {context}
Chat History: {chat_history}
Question: {question}

Helpful Answer:"""

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
            retriever=self.vector_db.as_retriever(search_kwargs={"k": self.config.top_k_retrieval}),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        print("✅ System Ready!")

    def chat(self, question: str):
        if not self.qa_chain:
            return "System not initialized."
        
        result = self.qa_chain({"question": question})
        print(f"\n❓ Q: {question}")
        print(f"🤖 A: {result['answer']}")
        return result['answer']

def create_sample_policies_file():
    sample_content = """
CHAPTER 1: WORKING HOURS AND ATTENDANCE
1.1. Standard working hours are 9:00 AM to 6:00 PM, Monday through Friday.
1.2. Employees are entitled to a 60-minute lunch break.
1.3. Remote work is allowed up to 2 days per week with prior manager approval.

CHAPTER 2: LEAVE AND TIME OFF
2.1. Employees are entitled to 21 days of paid annual leave.
2.2. Sick leave is paid according to national legislation.
2.3. Maternity leave consists of 126 calendar days.
2.4. Public holidays are non-working paid days.

CHAPTER 3: EMPLOYEE BENEFITS
3.1. Private medical insurance is provided for all full-time employees.
3.2. Gym membership or fitness activity allowance.
3.3. Meal vouchers worth 30 RON per working day.
3.4. Annual budget of 2000 RON for training and professional development.
"""
    with open("companyPolicies.txt", "w", encoding="utf-8") as f:
        f.write(sample_content)
    print("✅ File companyPolicies.txt created successfully!")

if __name__ == "__main__":
    if not os.path.exists("companyPolicies.txt"):
        create_sample_policies_file()
    
    rag_system = HRKnowledgeRAGSystem()
    rag_system.setup("companyPolicies.txt")
    
    test_questions = [
        "How many vacation days do I have per year?",
        "What is the standard working schedule?",
        "What medical benefits are provided?",
        "Is remote work allowed?",
        "What is the budget for professional development?"
    ]
    
    for question in test_questions:
        rag_system.chat(question)
    
    print("\n💬 INTERACTIVE MODE (type 'exit' to quit)")
    while True:
        user_question = input("\nYour Question: ")
        if user_question.lower() == 'exit':
            break
        rag_system.chat(user_question)