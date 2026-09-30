"""
Recursive Semantic Chunker Module
Splits documents using hierarchical boundary separators (paragraphs, headers, sentences)
with configurable overlap and rich metadata attachment to eliminate context fragmentation.
"""

import re
import uuid
from typing import List, Dict, Any
from src.core.state import DocumentChunk
from src.config import settings


class RecursiveSemanticChunker:
    """
    Production-grade recursive text chunker.
    Preserves semantic sentence boundaries and enriches each chunk with lineage metadata.
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        # Hierarchical boundary regex
        self.separators = ["\n\n## ", "\n\n# ", "\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        """
        Splits text into cohesive semantic chunks and returns DocumentChunk objects.
        """
        base_meta = metadata or {}
        raw_chunks = self._recursive_split(text, self.separators)
        
        # Merge tiny chunks and enforce overlap
        merged_chunks = self._merge_with_overlap(raw_chunks)
        
        document_chunks = []
        for idx, chunk_text in enumerate(merged_chunks):
            chunk_id = f"{base_meta.get('source', 'doc')}_{idx}_{uuid.uuid4().hex[:6]}"
            chunk_meta = {
                **base_meta,
                "chunk_index": idx,
                "total_chunks": len(merged_chunks),
                "char_length": len(chunk_text),
                "estimated_tokens": len(chunk_text.split()) * 4 // 3,
            }
            document_chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    content=chunk_text.strip(),
                    metadata=chunk_meta,
                    source=base_meta.get("source", "internal"),
                )
            )
        return document_chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Hierarchical recursive string splitting."""
        final_chunks: List[str] = []
        if not separators:
            return [text]

        separator = separators[0]
        new_separators = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        for s in splits:
            if not s.strip():
                continue
            if len(s) <= self.chunk_size:
                final_chunks.append(s.strip())
            else:
                if new_separators:
                    sub_chunks = self._recursive_split(s, new_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(s.strip())

        return final_chunks

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Merges adjacent pieces up to chunk_size and injects sliding overlap."""
        merged: List[str] = []
        current: List[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            if current_len + piece_len + 1 <= self.chunk_size:
                current.append(piece)
                current_len += piece_len + 1
            else:
                if current:
                    merged_text = " ".join(current)
                    merged.append(merged_text)
                    # Retain trailing portion for sliding window overlap
                    overlap_chars = 0
                    overlap_pieces = []
                    for p in reversed(current):
                        overlap_pieces.insert(0, p)
                        overlap_chars += len(p)
                        if overlap_chars >= self.chunk_overlap:
                            break
                    current = overlap_pieces
                    current_len = sum(len(p) for p in current) + len(current)
                current.append(piece)
                current_len += piece_len + 1

        if current:
            merged.append(" ".join(current))

        return merged


semantic_chunker = RecursiveSemanticChunker()
