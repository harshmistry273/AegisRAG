"""
LangGraph Native StateGraph Implementation of Corrective RAG (CRAG)
Demonstrates production agentic state machine construction using LangGraph.
"""

from typing import Dict, Any, List
from src.core.state import GraphState, DocumentChunk
from src.agentic.crag_workflow import crag_workflow


def build_crag_graph():
    """
    Constructs and compiles the LangGraph StateGraph workflow for Corrective RAG.
    Nodes:
      - transform_query
      - hybrid_retrieve
      - neural_rerank
      - grade_documents
      - web_search_fallback
      - generate_answer
      - check_faithfulness
    """
    try:
        from langgraph.graph import StateGraph, START, END

        workflow = StateGraph(GraphState)

        # Node definitions
        def transform_query_node(state: GraphState):
            q = state["question"]
            expanded = crag_workflow.query_transformer.multi_query_expansion(q, num_queries=2)
            return {"transformed_queries": expanded}

        def hybrid_retrieve_node(state: GraphState):
            queries = state.get("transformed_queries") or [state["question"]]
            candidates = []
            for q in queries:
                fused = crag_workflow.hybrid_retriever.retrieve(q)
                for doc, score, _ in fused:
                    if not any(d.id == doc.id for d in candidates):
                        doc.score = score
                        candidates.append(doc)
            return {"documents": candidates}

        def rerank_node(state: GraphState):
            docs = state.get("documents", [])
            q = state["question"]
            reranked = crag_workflow.reranker.rerank(q, docs)
            return {"filtered_documents": [t[0] for t in reranked]}

        def grade_documents_node(state: GraphState):
            docs = state.get("filtered_documents", [])
            q = state["question"]
            grade = crag_workflow.graders.grade_retrieval_relevance(q, docs)
            is_rel = grade["is_relevant"]
            return {
                "relevance_grade": "yes" if is_rel else "no",
                "relevance_score": grade["relevance_score"],
                "web_search_needed": not is_rel
            }

        def web_search_node(state: GraphState):
            q = state["question"]
            web_docs = crag_workflow.web_search.search(q, max_results=3)
            return {"filtered_documents": web_docs}

        def generate_answer_node(state: GraphState):
            docs = state.get("filtered_documents", [])
            q = state["question"]
            context = "\n\n".join([f"[{d.id}]: {d.content}" for d in docs])
            prompt = f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer with source citations:"
            res = crag_workflow.groq.generate(prompt)
            return {"generation": res["content"]}

        def check_faithfulness_node(state: GraphState):
            docs = state.get("filtered_documents", [])
            ans = state.get("generation", "")
            faith = crag_workflow.graders.grade_hallucination(ans, docs)
            return {
                "hallucination_grade": "grounded" if faith["is_grounded"] else "hallucinated",
                "faithfulness_score": faith["faithfulness_score"]
            }

        # Routing conditional logic
        def decide_to_search(state: GraphState):
            if state.get("web_search_needed", False):
                return "web_search"
            return "generate"

        # Register nodes
        workflow.add_node("transform_query", transform_query_node)
        workflow.add_node("hybrid_retrieve", hybrid_retrieve_node)
        workflow.add_node("rerank_documents", rerank_node)
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("web_search", web_search_node)
        workflow.add_node("generate_answer", generate_answer_node)
        workflow.add_node("check_faithfulness", check_faithfulness_node)

        # Wire graph transitions
        workflow.add_edge(START, "transform_query")
        workflow.add_edge("transform_query", "hybrid_retrieve")
        workflow.add_edge("hybrid_retrieve", "rerank_documents")
        workflow.add_edge("rerank_documents", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_search,
            {
                "web_search": "web_search",
                "generate": "generate_answer"
            }
        )
        workflow.add_edge("web_search", "generate_answer")
        workflow.add_edge("generate_answer", "check_faithfulness")
        workflow.add_edge("check_faithfulness", END)

        return workflow.compile()
    except Exception as e:
        # Graceful fallback if langgraph is not installed
        return None
