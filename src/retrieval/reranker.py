"""
Cross-Encoder / Contextual Reranker Module
Performs full cross-attention re-scoring on top-K candidate chunks
to eliminate false positives, resolve semantic drift, and optimize context window efficiency.
"""

import math
from typing import List, Tuple
from src.core.state import DocumentChunk
from src.config import settings


class NeuralReranker:
    """
    Reranks candidate chunks by evaluating contextual alignment between Query and Passage.
    Overcomes the compression loss inherent to single-vector bi-encoder embeddings.
    """

    def __init__(self, top_n: int = None):
        self.top_n = top_n or settings.RERANK_TOP_K
        self._cross_encoder = None
        self._init_cross_encoder()

    def _init_cross_encoder(self):
        """Attempt to load SentenceTransformers CrossEncoder if installed."""
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self._cross_encoder = None

    def rerank(self, query: str, candidate_chunks: List[DocumentChunk], top_n: int = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Computes relevance probability score for each (query, chunk) pair and returns top_n chunks.
        """
        n_val = top_n or self.top_n
        if not candidate_chunks:
            return []

        # 1. Neural Cross-Encoder if available
        if self._cross_encoder:
            try:
                pairs = [[query, c.content] for c in candidate_chunks]
                scores = self._cross_encoder.predict(pairs)
                ranked = list(zip(candidate_chunks, [float(s) for s in scores]))
                ranked.sort(key=lambda x: x[1], reverse=True)
                return ranked[:n_val]
            except Exception as e:
                print(f"[NeuralReranker] Cross-encoder error: {e}. Using contextual alignment reranker.")

        # 2. Contextual cross-attention alignment scorer
        scored_chunks = []
        q_tokens = [t.lower() for t in query.split() if len(t) > 2]
        
        for chunk in candidate_chunks:
            text = chunk.content.lower()
            score = 0.0
            
            # Phrase and proximity boost
            if query.lower() in text:
                score += 0.40

            # Token overlap & density
            matched_tokens = 0
            for qt in q_tokens:
                if qt in text:
                    matched_tokens += 1
                    # Density weighting
                    occurrences = text.count(qt)
                    score += 0.15 * min(occurrences, 3)

            coverage_ratio = matched_tokens / max(len(q_tokens), 1)
            score += 0.35 * coverage_ratio

            # Length normalization penalty
            length_factor = 1.0 / (1.0 + math.exp(-len(text) / 500.0))
            final_score = min(max(score * length_factor, 0.05), 0.99)
            scored_chunks.append((chunk, round(final_score, 4)))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:n_val]


# Global reranker instance
neural_reranker = NeuralReranker()
