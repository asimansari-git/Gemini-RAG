# Advanced RAG: Hybrid Search with Reciprocal Rank Fusion (RRF)

This document explains the concept of Hybrid Search and Reciprocal Rank Fusion (RRF), a state-of-the-art retrieval strategy that combines the precision of keyword search with the conceptual understanding of semantic vector search.

---

### The Problem: The Blind Spots of Vector Search

Dense vector search (using embedding models like `gemini-embedding-001` or `text-embedding-004`) is exceptional at understanding concepts, synonyms, and natural phrasing. However, it suffers from notorious blind spots:

1.  **Exact Keywords & Identifiers:** Product model numbers (`SKU-9821`), error codes (`0x80070005`), chemical formulas, or function names (`reciprocal_rank_fusion`) often lack rich conceptual semantics in embedding space. Dense vectors can easily overlook or misrank them.
2.  **Statutory & Legal Citations:** Precise section numbers and legal acts require exact character matching rather than conceptual approximations.
3.  **Out-of-Vocabulary Terms:** Newly coined terms, acronyms, or proper names might not map cleanly to neighboring vector clusters.

Conversely, traditional lexical search (like **BM25**) excels at finding exact token occurrences and frequency weighting, but is completely blind to synonyms, semantic meaning, or paraphrased intent.

---

### The Solution: Hybrid Search (Sparse + Dense)

**Hybrid Search** unifies both paradigms into a single, complementary retrieval pipeline:

*   **Dense Retrieval (ChromaDB):** Retrieves candidate chunks based on cosine distance in semantic embedding space.
*   **Sparse Retrieval (BM25):** Retrieves candidate chunks based on exact keyword occurrences and term frequency-inverse document frequency weighting.

By querying both indices simultaneously, we ensure that both high-level semantic intent and precise keyword hits are captured in the candidate pool.

---

### Merging Results: Reciprocal Rank Fusion (RRF)

Once we obtain two independent ranked lists of documents (one from ChromaDB and one from BM25), we cannot simply add their raw scores together because:
*   ChromaDB scores are distances or cosine similarities (ranging from 0 to 1).
*   BM25 scores are unbounded positive numbers based on term statistics.

**Reciprocal Rank Fusion (RRF)** solves this without requiring complex score normalization. Instead of using raw scores, RRF evaluates the **rank position** of each document across both lists:

$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

Where:
*   $M$ represents the set of retrieval systems (ChromaDB dense search and BM25 sparse search).
*   $r_m(d)$ is the rank position (1st, 2nd, 3rd, etc.) of document $d$ in system $m$.
*   $k$ is a smoothing constant (standard default is $60$) that prevents top-ranked outliers from completely dominating the combined score.

Documents appearing near the top of both lists receive the highest cumulative RRF scores, while documents present in only one list still receive fair representation if they ranked near the top.

---

### The Analogy: Two Expert Investigators

Imagine you are solving a complex mystery and consult two different investigators:

*   **Investigator A (The Semantic Profiler - Dense Vector):** Understands human intent, motives, and broad psychological patterns. They find suspects whose behavior feels conceptually aligned with the crime.
*   **Investigator B (The Forensic Archivist - BM25):** Searches fingerprint databases, badge serial numbers, and exact license plate logs. They find direct physical matches.
*   **The Lead Detective (RRF):** Takes the ranked suspect lists from both investigators. Suspects flagged highly by both investigators immediately rise to the top of the priority list, followed by those with decisive individual evidence.

---

### Implementation Overview

In this branch, the pipeline works as follows:

1.  **Data Ingestion (`load_data.py`):**
    *   Chunks documents and inserts them into ChromaDB with embeddings.
    *   Simultaneously tokenizes all chunks and serializes a `BM25Okapi` index (`bm25_index.pkl`) in `chroma_storage`.
2.  **Candidate Retrieval (`main.py`):**
    *   Fetches the top-$k$ dense candidates from ChromaDB.
    *   Fetches the top-$k$ sparse candidates from BM25.
3.  **Score Fusion:**
    *   Applies `reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_n=5)` to merge and order the documents.
4.  **Final Generation:**
    *   Injects the fused context into `gemini-2.5-flash` to construct the final factual answer.
