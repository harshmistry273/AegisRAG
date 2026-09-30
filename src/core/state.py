"""
Graph State definition for LangGraph Corrective RAG (CRAG) and Self-RAG.
Maintains state across query transformation, hybrid retrieval, grading, synthesis, and reflection.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a discrete context chunk retrieved from knowledge sources."""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    source: str = "internal"  # internal | web | fallback


class PipelineStepLog(BaseModel):
    """Telemetry log for individual nodes in the RAG execution graph."""
    node_name: str
    description: str
    status: str = "success"  # success | warning | skipped | error
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphState(TypedDict):
    """
    State dictionary passed between nodes in the LangGraph workflow.
    """
    question: str
    transformed_queries: List[str]
    documents: List[DocumentChunk]
    filtered_documents: List[DocumentChunk]
    generation: str
    web_search_needed: bool
    relevance_grade: str  # 'yes' | 'no'
    relevance_score: float
    hallucination_grade: str  # 'grounded' | 'hallucinated'
    answer_quality_grade: str  # 'useful' | 'not_useful'
    iterations: int
    trace_logs: List[Dict[str, Any]]
