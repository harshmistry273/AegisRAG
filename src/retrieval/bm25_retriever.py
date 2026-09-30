"""
BM25 Lexical Retriever Module
Implements the Okapi BM25 sparse keyword ranking algorithm.
Excels at exact keyword matches, code identifiers, acronyms, and proper nouns
where dense embeddings frequently suffer semantic drift.
"""

import re
import math
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.core.state import DocumentChunk
from src.config import settings


class BM25Retriever:
    """
    Okapi BM25 sparse lexical search engine.
    Calculates term-frequency / inverse-document-frequency (TF-IDF) relevance with
    document length normalization parameter 'b' and frequency saturation 'k1'.
    """

    def __init__(self, k1: float = None, b: float = None):
        self.k1 = k1 or settings.BM25_K1
        self.b = b or settings.BM25_B
        self.corpus_chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25_model: BM25Okapi = None

    def index_documents(self, documents: List[DocumentChunk]):
        """
        Tokenizes and builds the BM25 inverted index from document chunks.
        """
        self.corpus_chunks = list(documents)
        self.tokenized_corpus = [self._tokenize(doc.content) for doc in self.corpus_chunks]
        if self.tokenized_corpus:
            self.bm25_model = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self.bm25_model = None

    def search(self, query: str, top_k: int = 6) -> List[Tuple[DocumentChunk, float]]:
        """
        Ranks indexed documents against query terms using the BM25 scoring formula.
        Returns list of (DocumentChunk, bm25_score).
        """
        if not self.bm25_model or not self.corpus_chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25_model.get_scores(query_tokens)
        top_k = min(top_k, len(self.corpus_chunks))
        
        # Sort indices descending
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = indexed_scores[:top_k]

        results = []
        for idx, score in top_results:
            if score > 0:  # Only include chunks with positive term overlap
                results.append((self.corpus_chunks[idx], float(score)))

        return results

    def _tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into lowercase word tokens."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [tok for tok in cleaned.split() if len(tok) > 1]


# Global BM25 retriever instance
bm25_retriever = BM25Retriever()
