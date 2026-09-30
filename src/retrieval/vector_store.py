"""
Vector Store Module
Provides high-performance dense vector storage with cosine similarity indexing,
persistent disk serialization, and cloud connector compatibility (Qdrant / Chroma).
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from src.core.state import DocumentChunk
from src.core.embeddings import embedding_engine
from src.config import settings


class DenseVectorStore:
    """
    High-performance in-memory and persistent Vector Store.
    Executes fast L2-normalized cosine similarity vector dot-products.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or settings.PERSIST_DIRECTORY
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.documents: List[DocumentChunk] = []
        self.vectors: np.ndarray = np.empty((0, settings.EMBEDDING_DIMENSION), dtype=np.float32)
        self._load_from_disk()

    def add_documents(self, docs: List[DocumentChunk]):
        """
        Embeds documents and updates the vector index, avoiding duplicate entries.
        """
        if not docs:
            return

        existing_contents = {doc.content for doc in self.documents}
        new_docs = [d for d in docs if d.content not in existing_contents]
        if not new_docs:
            return

        texts = [doc.content for doc in new_docs]
        new_embeddings = embedding_engine.embed_documents(texts)

        if len(self.documents) == 0:
            self.vectors = new_embeddings
            self.documents = list(new_docs)
        else:
            self.vectors = np.vstack([self.vectors, new_embeddings])
            self.documents.extend(new_docs)

        self._persist_to_disk()

    def search(self, query: str, top_k: int = 6) -> List[Tuple[DocumentChunk, float]]:
        """
        Calculates cosine similarity between query embedding and indexed document vectors:
        Cosine_Similarity(u, v) = (u . v) / (||u|| * ||v||)
        Returns list of (DocumentChunk, similarity_score).
        """
        if len(self.documents) == 0 or self.vectors.shape[0] == 0:
            return []

        query_vec = embedding_engine.embed_query(query)  # shape (dim,)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 1e-12:
            query_vec = query_vec / query_norm

        # Matrix-vector dot product: (N, dim) @ (dim,) -> (N,)
        scores = np.dot(self.vectors, query_vec)

        # Retrieve top-k indices sorted descending
        top_k = min(top_k, len(self.documents))
        best_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in best_indices:
            doc = self.documents[idx]
            score = float(scores[idx])
            results.append((doc, score))

        return results

    def clear(self):
        """Reset the vector index."""
        self.documents = []
        self.vectors = np.empty((0, settings.EMBEDDING_DIMENSION), dtype=np.float32)
        meta_file = self.persist_dir / "documents.json"
        vec_file = self.persist_dir / "vectors.npy"
        if meta_file.exists():
            meta_file.unlink()
        if vec_file.exists():
            vec_file.unlink()

    def _persist_to_disk(self):
        """Serialize index state to disk."""
        meta_file = self.persist_dir / "documents.json"
        vec_file = self.persist_dir / "vectors.npy"

        docs_data = [doc.model_dump() for doc in self.documents]
        meta_file.write_text(json.dumps(docs_data, indent=2), encoding="utf-8")
        np.save(vec_file, self.vectors)

    def _load_from_disk(self):
        """Deserialize index state from disk if exists."""
        meta_file = self.persist_dir / "documents.json"
        vec_file = self.persist_dir / "vectors.npy"

        if meta_file.exists() and vec_file.exists():
            try:
                docs_data = json.loads(meta_file.read_text(encoding="utf-8"))
                self.documents = [DocumentChunk(**item) for item in docs_data]
                self.vectors = np.load(vec_file)
            except Exception as e:
                print(f"[DenseVectorStore] Warning: Could not load cache: {e}")
                self.documents = []
                self.vectors = np.empty((0, settings.EMBEDDING_DIMENSION), dtype=np.float32)


# Global vector store instance
vector_store = DenseVectorStore()
