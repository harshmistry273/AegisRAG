"""
Query Transformation Module
Implements advanced pre-retrieval techniques:
1. Multi-Query Expansion: Generates diverse paraphrased perspectives to maximize recall.
2. Hypothetical Document Embeddings (HyDE): Synthesizes an ideal hypothetical passage to bridge the query-document semantic gap.
3. Query Rewriting: Context-aware reformulation to resolve ambiguity and remove conversational noise.
"""

from typing import List
from src.core.groq_client import groq_client
from src.config import settings


class QueryTransformer:
    """
    Query transformation engine powered by Groq LPU inference.
    """

    def multi_query_expansion(self, question: str, num_queries: int = 3) -> List[str]:
        """
        Generates alternative formulations of the user's question to retrieve documents
        across different semantic angles.
        """
        system_prompt = (
            "You are an expert Information Retrieval query expansion agent. "
            "Your task is to generate alternative search queries for the user prompt. "
            "Output each alternative query on a new line without numbering or bullets."
        )
        prompt = (
            f"Original Query: {question}\n\n"
            f"Provide {num_queries} diverse search queries with different keywords or angles "
            f"that help retrieve relevant passages from a technical knowledge base."
        )

        res = groq_client.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.3)
        raw_lines = [line.strip().lstrip("0123456789.- ") for line in res["content"].split("\n") if line.strip()]
        
        # Always retain original query
        queries = [question]
        for q in raw_lines:
            if q and q.lower() != question.lower() and len(queries) < num_queries + 1:
                queries.append(q)
        return queries

    def hyde_transform(self, question: str) -> str:
        """
        Hypothetical Document Embeddings (HyDE).
        Generates a plausible, hallucinated answer passage to search for actual documents
        in dense embedding space (document-to-document similarity instead of query-to-document).
        """
        system_prompt = (
            "You are a knowledge synthesizer. Write a brief, authoritative 2-3 sentence hypothetical "
            "textbook passage that answers the user question. Do not state whether it is true or false."
        )
        prompt = f"Question: {question}\n\nHypothetical Passage:"
        res = groq_client.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.2)
        return res["content"].strip()

    def rewrite_query(self, question: str, feedback: str = "") -> str:
        """
        Reformulates poorly retrieving or failed queries into targeted keyword-rich search queries.
        """
        system_prompt = (
            "You are an information retrieval specialist. Rewrite the following user question into "
            "a clear, precise, keyword-optimized search query. Output ONLY the rewritten query."
        )
        prompt = f"Original Question: {question}\n"
        if feedback:
            prompt += f"Context/Issue: {feedback}\n"
        prompt += "Optimized Search Query:"

        res = groq_client.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.1)
        cleaned = res["content"].strip().strip('"').strip("'")
        return cleaned if cleaned else question


# Global transformer instance
query_transformer = QueryTransformer()
