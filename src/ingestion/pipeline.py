"""
Ingestion Pipeline Orchestrator
Coordinates document loading, semantic chunking, embedding generation,
and dual index population (BM25 + Dense Vector Store).
"""

from pathlib import Path
from typing import List, Union, Dict, Any
from src.ingestion.loader import document_loader
from src.ingestion.chunker import semantic_chunker
from src.core.state import DocumentChunk


class IngestionPipeline:
    """
    End-to-end document ingestion engine.
    Prepares documents for hybrid retrieval by generating both semantic vector embeddings
    and lexical token indices.
    """

    def __init__(self):
        self.loader = document_loader
        self.chunker = semantic_chunker

    def process_file(self, file_path: Union[str, Path]) -> List[DocumentChunk]:
        """Loads a file, parses content, and outputs structured semantic chunks."""
        doc_data = self.loader.load_file(file_path)
        metadata = {
            "source": doc_data.get("source"),
            "file_type": doc_data.get("file_type"),
            "file_size": doc_data.get("file_size_bytes"),
        }
        chunks = self.chunker.chunk_document(doc_data["text"], metadata=metadata)
        return chunks

    def process_raw_text(self, text: str, source_name: str = "user_input") -> List[DocumentChunk]:
        """Chunks arbitrary raw text provided via UI or API."""
        metadata = {"source": source_name, "file_type": "text"}
        return self.chunker.chunk_document(text, metadata=metadata)


ingestion_pipeline = IngestionPipeline()
