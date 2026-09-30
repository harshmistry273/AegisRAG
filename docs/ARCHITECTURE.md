# Deep Architectural Specification: AegisRAG

## 1. Information Retrieval Bottlenecks in Naive RAG

Naive RAG relies upon a simplistic three-stage pipeline:
1. **Embedding generation** using a bi-encoder (e.g. `text-embedding-ada-002` or `all-MiniLM-L6-v2`).
2. **K-Nearest Neighbors (KNN)** or Approximate Nearest Neighbors (ANN, e.g. HNSW) retrieval based on cosine similarity.
3. **Unchecked Prompt Concatenation**: Appending the top-$K$ passages into the prompt context window.

### Primary Vulnerabilities:
- **Bi-Encoder Compression Bottleneck:** Bi-encoders map an entire passage of text $\mathcal{D}$ into an isolated vector $\mathbf{v} \in \mathbb{R}^d$. This independent projection prevents cross-attention between token pairs in $\mathcal{Q}$ and $\mathcal{D}$. While computationally efficient ($O(1)$ lookup via index), it forfeits fine-grained interaction.
- **Lost in the Middle:** Attention mechanism studies (Liu et al., Stanford) show LLM retrieval recall drops significantly when target facts reside in the middle of long contexts.
- **Semantic Drift on Lexical Identifiers:** Dense embeddings place words with similar conceptual roles close together, frequently confusing distinct alphanumeric tokens (e.g., error codes `HTTP 404` vs `HTTP 502`, or hardware specs `L40S` vs `H100`).

---

## 2. Mathematical Formalization of AegisRAG Solutions

### 2.1 Hybrid Retrieval: Dense + Okapi BM25
AegisRAG utilizes dual indexing. When a query is received:
1. Dense Retrieval produces ranking $R_{\text{dense}} = [d_1, d_2, \dots]$.
2. Sparse Okapi BM25 produces ranking $R_{\text{bm25}} = [d'_1, d'_2, \dots]$.

### 2.2 Reciprocal Rank Fusion (RRF)
Linear combination of scores:
$$S(d) = \alpha \cdot S_{\text{dense}}(d) + (1-\alpha) \cdot S_{\text{bm25}}(d)$$
is mathematically problematic because BM25 scores depend on corpus length and term frequencies, with no fixed upper bound, whereas cosine similarity is constrained to $[-1, 1]$.

Reciprocal Rank Fusion sidesteps calibration by utilizing ordinal rank positions:
$$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
where $k = 60$. If a document is ranked #1 in BM25 ($r=1$) and #5 in dense ($r=5$), its score is:
$$\text{RRF}(d) = \frac{1}{60 + 1} + \frac{1}{60 + 5} = 0.01639 + 0.01538 = 0.03177$$

### 2.3 Cross-Encoder Neural Reranking
After candidate pooling via RRF, the top candidate passages are fed into a cross-encoder:
$$\text{Score}_{\text{cross}}(Q, D) = \sigma\left(\mathbf{W} \cdot \text{Transformer}([CLS] \circ Q \circ [SEP] \circ D \circ [SEP])\right)$$
Because all tokens in $Q$ attend to all tokens in $D$ simultaneously across all transformer layers, false positives with surface-level semantic similarity are effectively discarded.

---

## 3. The Corrective RAG (CRAG) Cyclic State Machine

```
                   +-------------------+
                   |    Start: Query   |
                   +---------+---------+
                             |
                             v
               +-------------+-------------+
               |  Query Transformation     |
               |  (Multi-Query Expansion)  |
               +-------------+-------------+
                             |
                             v
               +-------------+-------------+
               |     Hybrid Retrieval      |
               | (Dense + BM25 via RRF)    |
               +-------------+-------------+
                             |
                             v
               +-------------+-------------+
               |  Neural Context Reranker  |
               +-------------+-------------+
                             |
                             v
               +-------------+-------------+
               |   CRAG Relevance Grader   |
               +-------------+-------------+
                             |
             +---------------+---------------+
             |                               |
       [Score >= 0.60]                 [Score < 0.60]
             |                               |
             |                               v
             |                 +-------------+-------------+
             |                 |    Web Search Fallback    |
             |                 +-------------+-------------+
             |                               |
             +---------------+---------------+
                             |
                             v
               +-------------+-------------+
               |    Groq LPU Synthesis     |
               |    (Llama-3.3 / Qwen)     |
               +-------------+-------------+
                             |
                             v
               +-------------+-------------+
               |  Self-RAG Groundedness    |
               |    Faithfulness Check     |
               +-------------+-------------+
                             |
                             v
                   +---------+---------+
                   |  Final Response   |
                   +-------------------+
```

---

## 4. Groq LPU Hardware Acceleration Rationale

In a standard agentic RAG loop, multi-call LLM latency compounds linearly:
$$T_{\text{total}} = T_{\text{query\_expand}} + T_{\text{grade\_docs}} + T_{\text{synthesis}} + T_{\text{hallucination\_check}}$$

- **Traditional GPU Cloud (NVIDIA A100 / H100):** Each generation call takes $1.5\text{s} - 2.5\text{s}$. Cumulative latency: **$6\text{s} - 10\text{s}$**, rendering the application unusable for interactive search.
- **Groq LPU (Language Processing Unit):** With static tensor execution schedules and on-chip SRAM bandwidth of $80\text{ TB/s}$, Groq achieves $300+\text{ tokens/sec}$ and sub-200ms TTFT. Cumulative latency: **$< 1.5\text{ seconds}$** for all 4 LLM invocations combined.
