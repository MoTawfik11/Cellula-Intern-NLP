## Unified NLP System (FastAPI)

This service unifies three tasks behind a single API:

- Task 3: Code generation with lightweight RAG prompt priming
- Task 4: RAG QA over a local Chroma DB (FLAN-T5)
- Task 5: LangGraph-based Python assistant with simple sessioning

### 1) Install

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r unified_app/requirements.txt
```

Optional for Task 5 (pick one):

- set OPENROUTER_API_KEY and OPENAI_BASE_URL=https://openrouter.ai/api/v1
- set GROQ_API_KEY and OPENAI_BASE_URL=https://api.groq.com/openai/v1
- or set OPENAI_API_KEY (and optionally MODEL)

### 2) Run

```bash
uvicorn unified_app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 3) Endpoints

1) Task 3 – Codegen

```bash
curl -X POST http://127.0.0.1:8000/task3/codegen \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a function is_prime(n)", "top_k": 3, "max_new_tokens": 256}'
```

2) Task 4 – RAG QA (uses `Task_4/chroma_db`)

```bash
curl -X POST http://127.0.0.1:8000/task4/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "k": 3}'
```

3) Task 5 – Assistant (sessioned)

```bash
curl -X POST http://127.0.0.1:8000/task5/assistant \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "Write a Python function to reverse a list."}'
```

### Notes

- First run will download models: `Salesforce/codegen-350M-mono` (Task 3) and `google/flan-t5-base` (Task 4).
- Ensure `Task_4/chroma_db` exists (already provided). To rebuild, use the scripts inside Task_4.
- GPU is auto-used if available for generation; otherwise CPU.
- For Task 5 without API keys, the assistant will respond with provider setup instructions.


