# embed_index.py
import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

DATA_PATH = "data/humaneval.jsonl"
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.pkl"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_index(batch_size=128):
    model = SentenceTransformer(EMB_MODEL)
    texts = []
    metas = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["prompt"])
            metas.append({"task_id": item["task_id"], "canonical_solution": item.get("canonical_solution",""), "prompt": item["prompt"]})
    # compute embeddings in batches
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    # FAISS requires float32
    embeddings = embeddings.astype('float32')
    dim = embeddings.shape[1]
    # For small dataset, IndexFlatIP with normalized vectors (cosine sim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    # save metadata
    with open(META_PATH, "wb") as mf:
        pickle.dump(metas, mf)
    print(f"Saved index to {INDEX_PATH}, meta to {META_PATH}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    build_index()
