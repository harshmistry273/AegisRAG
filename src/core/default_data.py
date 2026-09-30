"""
Default Enterprise Knowledge Base
Provides rich technical documentation on RAG, Okapi BM25, Reciprocal Rank Fusion,
and Groq LPU hardware architecture for instant demonstrations.
"""

DEFAULT_KNOWLEDGE = [
    {
        "source": "AegisRAG_Architecture_Doc.md",
        "text": """# AegisRAG System Architecture & Design
AegisRAG is a production-grade Corrective Retrieval-Augmented Generation (CRAG) system engineered to overcome the limitations of Naive RAG.
Traditional Naive RAG relies strictly on dense vector similarity (Bi-Encoder embeddings), which suffers from three fundamental flaws:
1. Lost in the Middle: Long context windows dilute critical information located in intermediate positions.
2. Semantic Drift: Dense embeddings fail on exact keyword lookups, acronyms, and alphanumeric entity identifiers.
3. Blind Context Injection: Naive RAG unconditionally feeds retrieved documents to the LLM even if they are irrelevant, resulting in hallucinated responses.

AegisRAG resolves these limitations through a multi-stage cyclic architecture:
First, Query Transformation utilizes Multi-Query Expansion and Hypothetical Document Embeddings (HyDE) to reformulate user intent.
Second, Hybrid Retrieval executes parallel Dense Cosine Search and Sparse Okapi BM25 keyword search, fused via Reciprocal Rank Fusion (RRF) with constant k=60.
Third, a Neural Contextual Reranker re-orders candidates via cross-attention scoring.
Fourth, an automated Document Relevance Grader evaluates retrieved excerpts. If relevance is insufficient, the system triggers automated query rewriting or external Web Search Fallback.
Finally, generation is executed using Groq's high-throughput LPU Inference Engine, followed by a Self-Reflective Faithfulness and Hallucination Guardrail."""
    },
    {
        "source": "Groq_LPU_Inference_Specifications.md",
        "text": """# Groq Language Processing Unit (LPU) Specifications
The Groq LPU is a purpose-built Tensor Streaming Processor designed specifically for sequential deep learning inference.
Unlike conventional GPUs (e.g., NVIDIA H100) which utilize dynamic thread scheduling and high-bandwidth memory (HBM) architectures that incur memory-bandwidth bottlenecks, Groq employs a Deterministic Tensor Architecture:
1. Static Compiler Scheduling: Execution schedules and data movement are computed statically at compile-time, eliminating runtime execution jitter.
2. Ultra-Low Time-to-First-Token (TTFT): Groq achieves TTFT under 200 milliseconds and sustained throughput exceeding 300 tokens per second on large open models.
3. Deterministic Latency: Linear scaling with predictable SLA guarantees, making Groq optimal for multi-step agentic loops where cumulative multi-call latency would otherwise degrade user experience."""
    },
    {
        "source": "Information_Retrieval_Mathematics.md",
        "text": """# Mathematical Foundations of Hybrid Retrieval
1. Okapi BM25 Lexical Ranking:
Given a query Q with terms q_1, ..., q_n and document D, the BM25 score is defined as:
Score(D, Q) = sum_{i=1}^{n} IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl)))
Where:
- f(q_i, D) is the term frequency of q_i in D.
- |D| is document length and avgdl is average document length across the corpus.
- k1 = 1.5 regulates term frequency saturation.
- b = 0.75 regulates document length normalization.

2. Reciprocal Rank Fusion (RRF):
To fuse rankings from disparate scoring domains (BM25 in [0, inf) and Cosine Similarity in [-1, 1]) without calibration bias, RRF scores each document d across retrieval methods M as:
RRF_Score(d) = sum_{m in M} 1 / (k + rank_m(d))
Where k = 60 is the smoothing constant that prevents high-ranking outliers from dominating the fused distribution."""
    }
]
