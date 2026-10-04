# Advanced RAG Techniques

This document explains several advanced techniques for Retrieval-Augmented Generation (RAG) that can be used to improve the quality and accuracy of a Q&A system.

---

### 1. Similarity Search (The Default)

*   **What it is:** The foundational RAG technique. It converts a user's query into a vector (a numerical representation of its meaning) and then searches a vector database for document chunks whose vectors are mathematically closest to the query's vector.
*   **The Goal:** To find the most relevant information based purely on the semantic meaning of the query.
*   **Analogy:** You ask a librarian for books about "space travel." They go to the catalog and find the 5 books whose descriptions are the most similar to the words "space travel." It's simple, direct, and effective.
*   **In our code:** The line `collection.query(query_texts=[query], n_results=5)` performs a similarity search.

---

### 2. Maximal Marginal Relevance (MMR) Search

*   **What it is:** A more advanced search method that balances two factors:
    1.  **Relevance:** How similar is the chunk to the user's query?
    2.  **Diversity:** How different is the chunk from the *other* chunks that have already been selected?
*   **The Goal:** To avoid retrieving a set of chunks that are all very similar and redundant. MMR aims to provide a diverse yet relevant set of results.
*   **Analogy:** You ask the librarian for books on "space travel." Instead of giving you 5 introductory books that all say the same thing, an MMR-powered librarian gives you one book on history, one on physics, one on future plans, etc. You get a much broader and more useful set of information.

---

### 3. Adding a Filter in Search (Metadata Filtering)

*   **What it is:** This technique allows you to pre-filter documents based on their metadata *before* the similarity search even happens. You instruct the database to only search within a subset of documents that match specific criteria.
*   **The Goal:** To dramatically narrow down the search space, making the search faster and more accurate, especially when the user's query implies a specific source.
*   **Analogy:** You say, "Find me information on 'the economy,' but *only* from the '2023 newspapers' section." The librarian completely ignores all other sections and only searches in the one you specified.
*   **Example:** If a user asks, "What did the president say in the 2023 State of the Union?", you could add a filter to only search where the metadata `filename` is `'state_of_the_union_2023.txt'`.

---

### 4. LLM-Aided Search (Self-Query)

*   **What it is:** A sophisticated technique where you use the LLM itself to help you search. You provide the LLM with the user's query and a description of your available metadata. The LLM then translates the natural language question into a structured query that includes both the search text and the appropriate metadata filters.
*   **The Goal:** To automatically apply filters based on the user's intent, without the user needing to know what filters are available.
*   **Analogy:** You say, "I'm trying to remember what that politician said about technology in last year's big speech." The super-smart librarian translates this into a precise plan: "I need to search for 'technology' but only in the document named 'big_speech_2024.txt'."

---

### 5. Compression

*   **What it is:** A post-processing step that refines the retrieved documents. First, you retrieve a larger number of documents (e.g., 20 instead of 5). Then, you use an LLM to quickly "compress" these documents by either:
    1.  **Filtering:** Discarding any documents that aren't truly relevant to the specific question.
    2.  **Summarizing/Extracting:** Pulling out only the single most relevant sentence or two from each document.
*   **The Goal:** To clean up the retrieved context, removing all noise and fluff. This provides the final LLM with a more potent, concentrated set of information, leading to a more accurate and concise final answer.
*   **Analogy:** The librarian brings you a stack of 10 books. But instead of making you read them all, they quickly flip through and put sticky notes on the single most important paragraph in each book that relates to your question. You only need to read the 10 paragraphs with sticky notes.

---

### 6. Hybrid Search (Sparse + Dense with Reciprocal Rank Fusion)

*   **What it is:** A dual-retrieval pipeline that combines lexical keyword search (BM25) with semantic vector search (ChromaDB) and blends their rankings using Reciprocal Rank Fusion (RRF).
*   **The Goal:** To eliminate the blind spots of pure semantic embeddings (which struggle with exact product serials, error codes, and statutory citations) while preserving semantic understanding for natural language queries.
*   **Analogy:** You ask for a specific vintage watch part. One librarian searches strictly by the exact catalog number (BM25), while another searches by descriptive style and time period (vector search). A lead coordinator (RRF) prioritizes the items found at the top of both searches.
*   **In our code:** `query_chroma_dense` and `query_bm25` retrieve candidate pools, which are fused via `reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_n=5)`.

