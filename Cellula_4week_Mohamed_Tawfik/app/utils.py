import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_documents(data_dir):
    """
    Load all .txt documents from the given directory.
    Each file becomes a LangChain Document with metadata.
    """
    docs = []

    if not os.path.exists(data_dir):
        print(f"⚠️ Data directory '{data_dir}' not found. Creating it...")
        os.makedirs(data_dir, exist_ok=True)
        return [Document(page_content="No documents found", metadata={"source": "none"})]

    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if not txt_files:
        print(f"⚠️ No .txt files found in '{data_dir}'. Please add text files to build the RAG database.")
        return [Document(page_content="No documents found", metadata={"source": "none"})]

    for fname in txt_files:
        file_path = os.path.join(data_dir, fname)
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            for d in loaded_docs:
                d.metadata["source"] = fname
                docs.append(d)
        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")

    print(f"✅ Loaded {len(docs)} documents from '{data_dir}'")
    return docs or [Document(page_content="No valid documents found", metadata={"source": "none"})]


def split_docs(docs, chunk_size, chunk_overlap):
    """
    Split documents into smaller chunks for embedding and retrieval.
    """
    if not docs or docs[0].page_content.strip() == "No documents found":
        print("⚠️ Skipping document splitting — no valid content.")
        return docs

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    print(f"🧩 Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks
