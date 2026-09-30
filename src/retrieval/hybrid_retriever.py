"""
Hybrid Retriever Module
Implements Reciprocal Rank Fusion (RRF) to combine Dense Semantic Vector Search
and Sparse Lexical BM25 Search into a robust, scale-invariant ranking.
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
from src.core.state import DocumentChunk
from src.retrieval.vector_store import vector_store
from src.retrieval.bm25_retriever import bm25_retriever
from src.config import settings


class HybridRetriever:
    """
    State-of-the-art Hybrid Retriever.
    Fuses dense vector semantic similarity with sparse BM25 lexical precision using RRF:
    RRF_Score(d) = sum_{m in {dense, bm25}} (1 / (k + rank_m(d)))
    """

    def __init__(self, rrf_k: int = None, top_k: int = None):
        self.rrf_k = rrf_k or settings.RRF_K
        self.top_k = top_k or settings.RETRIEVAL_TOP_K

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[DocumentChunk, float, Dict[str, Any]]]:
        """
        Executes parallel dense vector search and sparse BM25 search,
        then fuses candidate lists via Reciprocal Rank Fusion (RRF).

        Returns:
            List of (DocumentChunk, rrf_score, debug_attribution_metadata)
        """
        k_val = top_k or self.top_k

        # 1. Dense Semantic Retrieval
        dense_results = vector_store.search(query, top_k=k_val * 2)

        # 2. Sparse Lexical Retrieval
        sparse_results = bm25_retriever.search(query, top_k=k_val * 2)

        # Track ranks for RRF fusion
        # Map: chunk_id -> { "chunk": DocumentChunk, "dense_rank": int, "sparse_rank": int, "dense_score": float, "sparse_score": float }
        candidates: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "chunk": None,
            "dense_rank": None,
            "sparse_rank": None,
            "dense_score": 0.0,
            "sparse_score": 0.0,
            "rrf_score": 0.0
        })

        for rank, (doc, score) in enumerate(dense_results, start=1):
            candidates[doc.id]["chunk"] = doc
            candidates[doc.id]["dense_rank"] = rank
            candidates[doc.id]["dense_score"] = score

        for rank, (doc, score) in enumerate(sparse_results, start=1):
            if candidates[doc.id]["chunk"] is None:
                candidates[doc.id]["chunk"] = doc
            candidates[doc.id]["sparse_rank"] = rank
            candidates[doc.id]["sparse_score"] = score

        # Compute RRF score for all candidates
        fused_list: List[Tuple[DocumentChunk, float, Dict[str, Any]]] = []

        for chunk_id, data in candidates.items():
            rrf_score = 0.0
            if data["dense_rank"] is not None:
                rrf_score += 1.0 / (self.rrf_k + data["dense_rank"])
            if data["sparse_rank"] is not None:
                rrf_score += 1.0 / (self.rrf_k + data["sparse_rank"])

            data["rrf_score"] = rrf_score
            fused_list.append((data["chunk"], rrf_score, {
                "dense_rank": data["dense_rank"],
                "sparse_rank": data["sparse_rank"],
                "dense_score": round(data["dense_score"], 4),
                "sparse_score": round(data["sparse_score"], 4),
                "rrf_score": round(rrf_score, 6),
            }))

        # Sort descending by fused RRF score
        fused_list.sort(key=lambda x: x[1], reverse=True)
        return fused_list[:k_val]


# Global hybrid retriever instance
hybrid_retriever = HybridRetriever()
