"""
FastAPI REST API Application for AegisRAG
Exposes production-ready microservice endpoints for document ingestion,
hybrid retrieval, corrective generation, and system benchmarking.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.api.schemas import IngestTextRequest, IngestResponse, QueryRequest, QueryResponse, BenchmarkResponse
from src.ingestion.pipeline import ingestion_pipeline
from src.retrieval.vector_store import vector_store
from src.retrieval.bm25_retriever import bm25_retriever
from src.agentic.crag_workflow import crag_workflow
from src.evaluation.benchmark import rag_benchmark


app = FastAPI(
    title="AegisRAG REST API",
    description="Enterprise Multi-Modal Agentic & Corrective RAG (CRAG) System with Groq LPU Inference",
    version=settings.VERSION,
)

# Enable CORS for cross-origin web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
def health_check():
    """System health check and vector store status."""
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "indexed_chunks": len(vector_store.documents),
        "groq_model": settings.GROQ_MODEL,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
    }


@app.post("/ingest/text", response_model=IngestResponse, tags=["Ingestion"])
def ingest_raw_text(payload: IngestTextRequest):
    """Chunks and indexes arbitrary raw text into Dense and BM25 stores."""
    try:
        chunks = ingestion_pipeline.process_raw_text(payload.text, source_name=payload.source_name)
        vector_store.add_documents(chunks)
        bm25_retriever.index_documents(vector_store.documents)
        return IngestResponse(
            status="success",
            chunks_created=len(chunks),
            source=payload.source_name,
            message=f"Successfully indexed {len(chunks)} semantic chunks."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["RAG Engine"])
def query_rag_engine(payload: QueryRequest):
    """Executes full Corrective RAG or baseline Naive RAG pipeline."""
    try:
        result = crag_workflow.run(question=payload.question, mode=payload.mode)
        docs_repr = [
            {
                "id": doc.id,
                "content": doc.content,
                "source": doc.source,
                "metadata": doc.metadata,
                "score": round(getattr(doc, "score", 0.0), 4)
            }
            for doc in result["retrieved_documents"]
        ]
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            mode=result["mode"],
            model_used=result["model_used"],
            faithfulness_score=result["faithfulness_score"],
            relevance_score=result["relevance_score"],
            total_latency_ms=result["total_latency_ms"],
            retrieved_documents=docs_repr,
            trace_logs=result["trace_logs"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["Evaluation"])
def trigger_benchmark():
    """Runs automated comparative benchmark between Naive RAG and AegisRAG."""
    try:
        data = rag_benchmark.run_comparative_benchmark()
        return BenchmarkResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
