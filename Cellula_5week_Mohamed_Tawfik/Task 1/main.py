import os
import sys
from typing import Optional, Dict, Any

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Ensure task modules are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

app = FastAPI(title="Unified NLP System", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Unified NLP System</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #0b1220; color: #e6edf3; }
    header { padding: 16px 20px; background: #0f172a; border-bottom: 1px solid #1f2a44; }
    h1 { margin: 0; font-size: 18px; }
    .tabs { display: flex; gap: 8px; padding: 12px 20px; border-bottom: 1px solid #1f2a44; }
    .tab { padding: 8px 12px; cursor: pointer; background: #111827; border: 1px solid #1f2a44; border-bottom: none; border-radius: 8px 8px 0 0; }
    .tab.active { background: #1f2937; }
    .panel { display: none; padding: 20px; }
    .panel.active { display: block; }
    .row { margin-bottom: 12px; display: flex; flex-direction: column; gap: 6px; }
    input, textarea, select { background: #0f172a; color: #e6edf3; border: 1px solid #334155; border-radius: 6px; padding: 10px; }
    button { background: #2563eb; color: white; border: none; padding: 10px 14px; border-radius: 6px; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .result { white-space: pre-wrap; background: #0f172a; border: 1px solid #1f2a44; padding: 12px; border-radius: 6px; }
    .hint { color: #9aa4b2; font-size: 12px; }
    footer { padding: 12px 20px; color: #94a3b8; border-top: 1px solid #1f2a44; font-size: 12px; }
  </style>
  <script>
    function setTab(id) {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      document.getElementById('tab-'+id).classList.add('active');
      document.getElementById('panel-'+id).classList.add('active');
    }
    async function postJson(path, body) {
      const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error('HTTP '+res.status);
      return await res.json();
    }
    async function runTask3() {
      const prompt = document.getElementById('t3-prompt').value;
      const k = parseInt(document.getElementById('t3-k').value || '3');
      const maxNew = parseInt(document.getElementById('t3-max').value || '256');
      const btn = document.getElementById('t3-btn'); const out = document.getElementById('t3-out');
      btn.disabled = true; out.textContent = 'Running...';
      try { const data = await postJson('/task3/codegen', { prompt: prompt, top_k: k, max_new_tokens: maxNew }); out.textContent = data.code || JSON.stringify(data, null, 2); }
      catch (e) { out.textContent = 'Error: '+e.message; }
      finally { btn.disabled = false; }
    }
    async function runTask4() {
      const q = document.getElementById('t4-q').value;
      const k = parseInt(document.getElementById('t4-k').value || '3');
      const btn = document.getElementById('t4-btn'); const out = document.getElementById('t4-out');
      btn.disabled = true; out.textContent = 'Running...';
      try { const data = await postJson('/task4/qa', { question: q, k: k }); out.textContent = data.answer || JSON.stringify(data, null, 2); }
      catch (e) { out.textContent = 'Error: '+e.message; }
      finally { btn.disabled = false; }
    }
    async function runTask5() {
      const sid = document.getElementById('t5-sid').value || 'default';
      const msg = document.getElementById('t5-msg').value;
      const btn = document.getElementById('t5-btn'); const out = document.getElementById('t5-out');
      btn.disabled = true; out.textContent = 'Running...';
      try { const data = await postJson('/task5/assistant', { session_id: sid, message: msg }); out.textContent = data.reply || JSON.stringify(data, null, 2); }
      catch (e) { out.textContent = 'Error: '+e.message; }
      finally { btn.disabled = false; }
    }
    window.addEventListener('DOMContentLoaded', () => setTab('t3'));
  </script>
  </head>
  <body>
    <header><h1>Unified NLP System</h1></header>
    <div class="tabs">
      <div id="tab-t3" class="tab" onclick="setTab('t3')">Task 3 – Codegen</div>
      <div id="tab-t4" class="tab" onclick="setTab('t4')">Task 4 – RAG QA</div>
      <div id="tab-t5" class="tab" onclick="setTab('t5')">Task 5 – Assistant</div>
    </div>

    <section id="panel-t3" class="panel">
      <div class="row"><label>Prompt</label><textarea id="t3-prompt" rows="5" placeholder="Write a function is_prime(n)"></textarea></div>
      <div class="row" style="display:flex; gap:12px; flex-direction:row;">
        <div class="row" style="flex:1"><label>Top K</label><input id="t3-k" type="number" value="3" /></div>
        <div class="row" style="flex:1"><label>Max New Tokens</label><input id="t3-max" type="number" value="256" /></div>
      </div>
      <button id="t3-btn" onclick="runTask3()">Run Codegen</button>
      <div class="hint">Model downloads may take several minutes on first run.</div>
      <pre id="t3-out" class="result"></pre>
    </section>

    <section id="panel-t4" class="panel">
      <div class="row"><label>Question</label><textarea id="t4-q" rows="3" placeholder="What is RAG?"></textarea></div>
      <div class="row" style="max-width:200px"><label>Top K</label><input id="t4-k" type="number" value="3" /></div>
      <button id="t4-btn" onclick="runTask4()">Ask</button>
      <div class="hint">Uses local Chroma DB from Task_4/chroma_db and FLAN-T5.</div>
      <pre id="t4-out" class="result"></pre>
    </section>

    <section id="panel-t5" class="panel">
      <div class="row" style="max-width:300px"><label>Session ID</label><input id="t5-sid" placeholder="default" /></div>
      <div class="row"><label>Message</label><textarea id="t5-msg" rows="3" placeholder="Explain list comprehensions"></textarea></div>
      <button id="t5-btn" onclick="runTask5()">Send</button>
      <div class="hint">Configure provider env vars for best results (see README).</div>
      <pre id="t5-out" class="result"></pre>
    </section>

    <footer>
      Try the API in Swagger UI: <a href="/docs" target="_blank" style="color:#93c5fd">/docs</a>
    </footer>
  </body>
</html>
        """
    )


@app.get("/routes")
def routes() -> Dict[str, Any]:
    return {
        "name": "Unified NLP System",
        "version": "1.0.0",
        "routes": {
            "health": "/health",
            "task3_codegen": "/task3/codegen",
            "task4_qa": "/task4/qa",
            "task5_assistant": "/task5/assistant",
        },
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


class CodegenRequest(BaseModel):
    prompt: str
    top_k: Optional[int] = 3
    max_new_tokens: Optional[int] = 256


@app.post("/task3/codegen")
def task3_codegen(req: CodegenRequest) -> Dict[str, Any]:
    sys.path.append(os.path.join(ROOT, "Task_3"))
    from Task_3.rag_pipeline import run_rag  # type: ignore

    code = run_rag(req.prompt, top_k=req.top_k or 3, max_new_tokens=req.max_new_tokens or 256)
    return {"code": code}


class RagQuery(BaseModel):
    question: str
    k: Optional[int] = 3


@app.post("/task4/qa")
def task4_qa(req: RagQuery) -> Dict[str, Any]:
    # Build once per process
    sys.path.append(os.path.join(ROOT, "Task_4", "app"))
    from langchain.memory import ConversationBufferMemory
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline as hf_pipeline
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
    import torch

    persist_dir = os.path.join(ROOT, "Task_4", "chroma_db")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": req.k or 3})

    model_name = "google/flan-t5-base"
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = hf_pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, temperature=0.3, device=device)
    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    system_prompt = (
        "You are a helpful RAG assistant built by Mohamed Tawfik. "
        "Answer questions using retrieved context. If the context is not enough, say 'I’m not sure from the data.' Be concise and accurate."
    )
    prompt = PromptTemplate(input_variables=["context", "question"], template=f"{system_prompt}\n\nContext:\n{{context}}\n\nQuestion: {{question}}\nAnswer:")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    result = qa_chain.invoke({"question": req.question})
    return {"answer": result.get("answer", "")}


class AssistantMessage(BaseModel):
    session_id: Optional[str] = "default"
    message: str


_graph_cache: Dict[str, Any] = {}
_state_cache: Dict[str, Any] = {}


def _get_graph():
    if "graph" in _graph_cache:
        return _graph_cache["graph"]
    sys.path.append(os.path.join(ROOT, "Task_5"))
    from Task_5.graph import build_graph  # type: ignore
    g = build_graph().compile()
    _graph_cache["graph"] = g
    return g


def _get_state(session_id: str):
    sys.path.append(os.path.join(ROOT, "Task_5"))
    from Task_5.state import initial_state  # type: ignore
    if session_id not in _state_cache:
        _state_cache[session_id] = initial_state()
    return _state_cache[session_id]


@app.post("/task5/assistant")
def task5_assistant(msg: AssistantMessage) -> Dict[str, Any]:
    graph = _get_graph()
    state = _get_state(msg.session_id or "default")
    state.setdefault("messages", []).append({"role": "user", "content": msg.message})
    state = graph.invoke(state)
    # Save back
    _state_cache[msg.session_id or "default"] = state
    last = next((m for m in reversed(state["messages"]) if m["role"] == "assistant"), None)
    return {"reply": last["content"] if last else ""}


