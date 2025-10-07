# retriever.py
import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.pkl"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_path=INDEX_PATH, meta_path=META_PATH, emb_model=EMB_MODEL):
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metas = pickle.load(f)
        self.model = SentenceTransformer(emb_model)
        # We used normalized embeddings during index build
    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            meta = self.metas[int(idx)]
            results.append({"score": float(score), **meta})
        return results

if __name__ == "__main__":
    r = Retriever()
    q = "Write a function that computes the nth Fibonacci number."
    for res in r.retrieve(q, top_k=3):
        print(res["task_id"], res["score"])
