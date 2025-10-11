from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
import gradio as gr
import torch

# ====================================================
# --- 1. Load Vector Database ---
# ====================================================
PERSIST_DIR = "chroma_db"
print("🔹 Loading Chroma vector store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
print("✅ Chroma DB loaded successfully!")

# ====================================================
# --- 2. Load Local Model ---
# ====================================================
print("🔹 Loading local FLAN-T5 model... (this may take 20–40s the first time)")

model_name = "google/flan-t5-base"

device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.3,
    device=device
)

llm = HuggingFacePipeline(pipeline=pipe)
print("✅ Model loaded and ready!")

# ====================================================
# --- 3. Memory & Prompt ---
# ====================================================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

system_prompt = """You are a helpful RAG assistant built by Mohamed Tawfik.
Answer questions using retrieved context. 
If the context is not enough, say 'I’m not sure from the data.'
Be concise and accurate."""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"{system_prompt}\n\nContext:\n{{context}}\n\nQuestion: {{question}}\nAnswer:"
)

# ====================================================
# --- 4. Build RAG Chain ---
# ====================================================
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# ====================================================
# --- 5. Gradio Interface ---
# ====================================================
def chat(user_input, history):
    try:
        result = qa_chain.invoke({"question": user_input})
        return result["answer"]
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"

chat_interface = gr.ChatInterface(
    fn=chat,
    title="🔍 Mohamed Tawfik’s Offline RAG Assistant",
    description="A conversational RAG chatbot powered by a local FLAN-T5 model and Chroma DB.",
)

# ====================================================
# --- 6. Launch App ---
# ====================================================
if __name__ == "__main__":
    print("🚀 Starting Gradio app (offline)...")
    chat_interface.launch(server_name="127.0.0.1", server_port=7860, debug=True)
