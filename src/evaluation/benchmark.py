"""
Evaluation and Benchmarking Module
Provides automated evaluation comparing Naive RAG vs Advanced Corrective RAG (CRAG).
Calculates RAG Triad Metrics:
1. Context Relevance
2. Answer Faithfulness (Hallucination detection)
3. Answer Relevance
4. Latency & Token Efficiency
"""

import time
from typing import List, Dict, Any
from src.agentic.crag_workflow import crag_workflow


class RAGBenchmarkSuite:
    """
    Automated experimental benchmark suite for evaluating RAG architectures.
    """

    BENCHMARK_QUERIES = [
        "What is the mathematical formulation of Reciprocal Rank Fusion (RRF)?",
        "How does the Okapi BM25 algorithm normalize document length?",
        "Explain how Corrective RAG detects and handles irrelevant retrieved documents.",
        "What are the latency and architectural advantages of Groq LPU inference over standard GPUs?",
        "How does cross-encoder reranking resolve the bi-encoder information bottleneck?"
    ]

    def run_comparative_benchmark(self, queries: List[str] = None) -> Dict[str, Any]:
        """
        Executes head-to-head evaluation across test queries.
        Returns aggregated performance metrics for Naive RAG vs Advanced AegisRAG.
        """
        test_set = queries or self.BENCHMARK_QUERIES
        results = {
            "queries_evaluated": len(test_set),
            "naive_rag": {
                "avg_latency_ms": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "total_runs": 0,
            },
            "advanced_crag": {
                "avg_latency_ms": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "total_runs": 0,
            },
            "individual_comparisons": []
        }

        naive_latencies, naive_faiths, naive_rels = [], [], []
        adv_latencies, adv_faiths, adv_rels = [], [], []

        for q in test_set:
            # 1. Run Naive
            naive_run = crag_workflow.run(q, mode="naive")
            naive_latencies.append(naive_run["total_latency_ms"])
            naive_faiths.append(naive_run["faithfulness_score"])
            naive_rels.append(naive_run["relevance_score"])

            # 2. Run Advanced
            adv_run = crag_workflow.run(q, mode="advanced")
            adv_latencies.append(adv_run["total_latency_ms"])
            adv_faiths.append(adv_run["faithfulness_score"])
            adv_rels.append(adv_run["relevance_score"])

            results["individual_comparisons"].append({
                "query": q,
                "naive": {
                    "latency_ms": naive_run["total_latency_ms"],
                    "faithfulness": naive_run["faithfulness_score"],
                    "relevance": naive_run["relevance_score"],
                    "answer_excerpt": naive_run["answer"][:120] + "..."
                },
                "advanced": {
                    "latency_ms": adv_run["total_latency_ms"],
                    "faithfulness": adv_run["faithfulness_score"],
                    "relevance": adv_run["relevance_score"],
                    "answer_excerpt": adv_run["answer"][:120] + "..."
                }
            })

        n = len(test_set)
        results["naive_rag"]["avg_latency_ms"] = round(sum(naive_latencies) / n, 2)
        results["naive_rag"]["avg_faithfulness"] = round(sum(naive_faiths) / n, 3)
        results["naive_rag"]["avg_relevance"] = round(sum(naive_rels) / n, 3)
        results["naive_rag"]["total_runs"] = n

        results["advanced_crag"]["avg_latency_ms"] = round(sum(adv_latencies) / n, 2)
        results["advanced_crag"]["avg_faithfulness"] = round(sum(adv_faiths) / n, 3)
        results["advanced_crag"]["avg_relevance"] = round(sum(adv_rels) / n, 3)
        results["advanced_crag"]["total_runs"] = n

        # Percentage Improvements
        results["improvements"] = {
            "faithfulness_gain": round((results["advanced_crag"]["avg_faithfulness"] - results["naive_rag"]["avg_faithfulness"]) / results["naive_rag"]["avg_faithfulness"] * 100, 1),
            "relevance_gain": round((results["advanced_crag"]["avg_relevance"] - results["naive_rag"]["avg_relevance"]) / results["naive_rag"]["avg_relevance"] * 100, 1),
        }

        return results


# Global benchmark instance
rag_benchmark = RAGBenchmarkSuite()
