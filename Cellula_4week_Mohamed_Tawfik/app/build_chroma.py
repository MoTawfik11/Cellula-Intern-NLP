import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from config import DOCS_DIR, PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from utils import load_documents, split_docs

def build_chroma():
    print("🔹 Building Chroma vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = load_documents(DOCS_DIR)
    chunks = split_docs(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(f"✅ Chroma DB saved to {PERSIST_DIR}")

if __name__ == "__main__":
    build_chroma()
