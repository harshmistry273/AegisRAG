"""Ingestion components for AegisRAG"""
from src.ingestion.loader import document_loader, DocumentLoader
from src.ingestion.chunker import semantic_chunker, RecursiveSemanticChunker
from src.ingestion.pipeline import ingestion_pipeline, IngestionPipeline

__all__ = [
    "document_loader",
    "DocumentLoader",
    "semantic_chunker",
    "RecursiveSemanticChunker",
    "ingestion_pipeline",
    "IngestionPipeline",
]
