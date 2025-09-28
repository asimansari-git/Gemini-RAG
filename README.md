# RAG: A Hands-On Guide

This repository provides a hands-on guide to understanding and implementing various techniques for Retrieval-Augmented Generation (RAG). Each branch demonstrates a specific RAG concept, building from the simplest to the most advanced.

---

### How to Use This Repository

Each branch in this repository contains a complete, working example of a specific RAG technique. To get started, you'll want to bring the code to your local machine.

**1. Clone the Repository**

First, clone this repository to your computer using the following command:

```bash
git clone git@github.com:asimibnakhlaque/Gemini-RAG.git
```

**2. List All Available Branches**

Once cloned, navigate into the directory and see all the available branches. Each one is a different lesson.

```bash
git branch -a
```

**3. Switch to a Branch**

You can switch to any branch to see the code and documentation for that specific technique. For example, to start at the beginning, you would check out the `similarity-search` branch:

```bash
git checkout similarity-search
```

---

### The Learning Path (Index)

I recommend exploring the branches in the following order to get a comprehensive understanding of RAG, moving from fundamental to more advanced techniques.

1.  **[`similarity-search`](https://github.com/asimibnakhlaque/Gemini-RAG/tree/similarity-search) (The Foundation)**
    *   **Concept:** This branch implements the most basic form of RAG, using a simple similarity search to retrieve documents.
    *   **Analogy:** A librarian who finds books based on the similarity of their core ideas to your request.
    *   **Read:** [document](https://github.com/asimibnakhlaque/Gemini-RAG/tree/similarity-search?tab=readme-ov-file#rag-the-foundation-of-retrieval---similarity-search)

2.  **[`mmr`](https://github.com/asimibnakhlaque/Gemini-RAG/tree/mmr) (The Nutritionist)**
    *   **Concept:** This branch introduces Maximal Marginal Relevance (MMR) to improve the diversity of retrieved documents and avoid redundancy.
    *   **Analogy:** A nutritionist who builds a balanced meal, ensuring variety instead of just one type of food.
    *   **Read:** [document](https://github.com/asimibnakhlaque/Gemini-RAG/tree/mmr?tab=readme-ov-file#rag-searching-with-maximal-marginal-relevance-mmr)

3.  **[`metadata-filtering`](https://github.com/asimibnakhlaque/Gemini-RAG/tree/metadata-filtering) (The Librarian with an Index)**
    *   **Concept:** This branch demonstrates how to pre-filter documents based on their metadata, narrowing the search space for more precise results.
    *   **Analogy:** A librarian who can jump directly to the correct aisle and section before starting the search.
    *   **Read:** [document](https://github.com/asimibnakhlaque/Gemini-RAG/tree/metadata-filtering?tab=readme-ov-file#rag-precise-searching-with-metadata-filtering)

4.  **[`self-query`](https://github.com/asimibnakhlaque/Gemini-RAG/tree/self-query) (The Super-Smart Librarian)**
    *   **Concept:** This branch demonstrates how to use a Self-Query Retriever to let the LLM itself translate a natural language question into a structured, filtered query.
    *   **Analogy:** A librarian who understands the user's intent and automatically knows which section and topic to search for.
    *   **Read:** [document](https://github.com/asimibnakhlaque/Gemini-RAG/tree/self-query?tab=readme-ov-file#llm-aided-search-the-self-query-retriever)

5.  **[`compression`](https://github.com/asimibnakhlaque/Gemini-RAG/tree/compression) (The Research Assistant)**
    *   **Concept:** This branch shows how to use a model to compress retrieved documents down to only the most relevant information before generating a final answer.
    *   **Analogy:** An assistant who reads ten books and returns a single page of highlighted, relevant paragraphs.
    *   **Read:** [document](https://github.com/asimibnakhlaque/Gemini-RAG/tree/compression?tab=readme-ov-file#advanced-rag-contextual-compression)

6.  **[`compression-filter`](https://github.com/asimibnakhlaque/Gemini-RAG/tree/compression-filter) (The Relevancy Scanner)**
    *   **Concept:** This branch uses a faster, more direct compression method that filters documents based on their embedding similarity to the query.
    *   **Analogy:** A high-tech scanner that instantly rejects books that don't match the core theme of your request.
    *   **Read:** [document](https://github.com/asimibnakhlaque/Gemini-RAG/tree/compression-filter?tab=readme-ov-file#rag-compressing-with-embedding-filters)

### Additional Documentation

*   `RAG_Techniques.md`: A high-level overview of all the concepts covered.
*   `ChromaDB_vs_LangChain.md`: Explains the roles of the different libraries used in these examples.
*   `Embeddings_and_Vectorization.md`: Explains the core concepts of how text is converted into numbers for the computer to understand.

---

### Acknowledgements

It is important to give credit where it is due. The foundational `similarity-search` branch is based on the official ChromaDB documentation and examples.

The implementation of `load_data.py` and the core concepts for using Gemini embeddings with ChromaDB were heavily inspired by the official ChromaDB Gemini example.

Additionally, the `get_gemini_response` and `build_prompt` functions used across the `main.py` files in the various branches are also derived from this excellent resource.

You can find the original source code here:
[ChromaDB Gemini Examples](https://github.com/chroma-core/chroma/tree/main/examples/gemini)
