"""
Pydantic API Schemas for FastAPI Endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw text or article content to ingest")
    source_name: str = Field(default="api_upload", description="Identifier or filename for source")


class IngestResponse(BaseModel):
    status: str
    chunks_created: int
    source: str
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User search query or question")
    mode: str = Field(default="advanced", description="'advanced' for Corrective RAG or 'naive' for baseline")
    top_k: Optional[int] = Field(default=4, description="Number of context passages to retain")


class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str
    model_used: str
    faithfulness_score: float
    relevance_score: float
    total_latency_ms: float
    retrieved_documents: List[Dict[str, Any]]
    trace_logs: List[Dict[str, Any]]


class BenchmarkResponse(BaseModel):
    queries_evaluated: int
    naive_rag: Dict[str, Any]
    advanced_crag: Dict[str, Any]
    improvements: Dict[str, Any]
    individual_comparisons: List[Dict[str, Any]]
