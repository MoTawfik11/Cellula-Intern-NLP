import json
import os
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EXAMPLES_PATH = os.path.join("examples", "seed_examples.jsonl")
INDEX_DIR = "embed_index"
INDEX_FILE = os.path.join(INDEX_DIR, "faiss.index")
META_FILE = os.path.join(INDEX_DIR, "meta.json")
MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "3"))


class Retriever:
    def __init__(self) -> None:
        os.makedirs(INDEX_DIR, exist_ok=True)
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self._load()
        else:
            self._build()

    def _build(self) -> None:
        items = self._load_examples()
        texts = [self._example_text(it) for it in items]
        embeddings = self._embed(texts)
        embeddings = self._normalize(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, INDEX_FILE)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        self.index = index
        self.meta = items

    def _load(self) -> None:
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.index = index
        self.meta = items

    def _load_examples(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not os.path.exists(EXAMPLES_PATH):
            return items
        with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    def _embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return emb.astype(np.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def _example_text(self, it: Dict[str, Any]) -> str:
        title = it.get("title", "")
        desc = it.get("description", "")
        code = it.get("code", "")
        return f"{title}\n{desc}\n{code}"

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        if self.index is None or not self.meta:
            return []
        q_emb = self._embed([query])
        q_emb = self._normalize(q_emb)
        scores, idxs = self.index.search(q_emb, top_k)
        results: List[Dict[str, Any]] = []
        for rank, (i, score) in enumerate(zip(idxs[0], scores[0])):
            if i < 0 or i >= len(self.meta):
                continue
            item = dict(self.meta[i])
            item["_score"] = float(score)
            item["_rank"] = rank
            results.append(item)
        return results
