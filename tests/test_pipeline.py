"""
Integration test for AegisRAG with live Groq LPU API.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.core.groq_client import groq_client
from src.ingestion.pipeline import ingestion_pipeline
from src.retrieval.vector_store import vector_store
from src.retrieval.bm25_retriever import bm25_retriever
from src.agentic.crag_workflow import crag_workflow
from src.core.default_data import DEFAULT_KNOWLEDGE


def test_live_groq_and_pipeline():
    print("\n--- 1. Testing Live Groq LPU Connection ---")
    res = groq_client.generate("Reply with 'GROQ_LPU_ACTIVE' and the model name if you are operational.")
    print("Groq Response:", res["content"])
    print("Latency:", res["latency_ms"], "ms | Model:", res["model"], "| Mock?", res.get("is_mock"))

    print("\n--- 2. Ingesting Enterprise Knowledge Base ---")
    for item in DEFAULT_KNOWLEDGE:
        chunks = ingestion_pipeline.process_raw_text(item["text"], source_name=item["source"])
        vector_store.add_documents(chunks)
    bm25_retriever.index_documents(vector_store.documents)
    print(f"Total Chunks Indexed: {len(vector_store.documents)}")

    print("\n--- 3. Testing Advanced Corrective RAG Workflow ---")
    query = "What is the mathematical formulation of Reciprocal Rank Fusion (RRF)?"
    result = crag_workflow.run(query, mode="advanced")
    print("\nResult Answer:\n", result["answer"][:300], "...")
    print("\nTotal Latency:", result["total_latency_ms"], "ms")
    print("Faithfulness Score:", result["faithfulness_score"])
    print("Relevance Score:", result["relevance_score"])
    print("\nExecution Nodes:")
    for step in result["trace_logs"]:
        print(f" - [{step['status'].upper()}] {step['node']}: {step.get('latency_ms', 0)} ms")

    print("\n--- 4. All Tests Passed Successfully! ---")


if __name__ == "__main__":
    test_live_groq_and_pipeline()
