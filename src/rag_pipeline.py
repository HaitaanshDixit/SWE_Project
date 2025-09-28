import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pathlib import Path

VECTORSTORE_DIR = Path("faiss_index")

class RAGPipeline:

    def __init__(self):
        # Initialize LLM and embedding model
        self.llm = ChatGroq(
            model_name="llama3-8b-8192",  # or "mixtral-8x7b-32768"
            temperature=0.9
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # Load or create vectorstore
        if VECTORSTORE_DIR.exists():
            self.vectorstore = FAISS.load_local(
                str(VECTORSTORE_DIR),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else :
            try :
                self.vectorstore = FAISS.from_texts([""], self.embeddings)
                self.vectorstore.save_local(str(VECTORSTORE_DIR))
            except Exception as e:
                raise RuntimeError("Failed to create a FAISS index: " + str(e))    


        # Build the QA chain
        retriever = self.vectorstore.as_retriever()
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)

    def get_chunks(self, documents, chunk_size=500, chunk_overlap=100):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)

    def add_documents(self, documents):
        chunks = self.get_chunks(documents)
        self.vectorstore.add_documents(chunks)
        self.vectorstore.save_local(str(VECTORSTORE_DIR))

    def get_answer(self, query):
        return self.qa_chain.run(query)
