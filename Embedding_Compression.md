# RAG: Compressing with Embedding Filters

This document explains the concept of an Embedding Filter, a specific and highly efficient method for contextual compression in a RAG system.

---

### The Concept: A Mathematical Gatekeeper

In our previous discussion on compression, we talked about using a language model to read retrieved documents and extract the relevant parts. That is a powerful technique, but it requires an extra LLM call, which adds time and cost.

An **Embedding Filter** is a different, more direct form of contextual compression. It doesn't need a separate LLM to decide what's relevant. Instead, it uses the power of mathematics—specifically, the vector embeddings we already have.

The process is simple but very effective:

1.  **Retrieve:** First, the system does a normal vector search to get a list of potentially relevant documents (e.g., the top 10 most similar documents).
2.  **Compare:** It then takes the original user's query and compares its vector embedding directly against the vector embedding of *each retrieved document*.
3.  **Filter:** It calculates a similarity score for each document. If a document's similarity to the query is below a certain **similarity_threshold** (e.g., 0.79), it is discarded.
4.  **Pass-Through:** Only the documents that pass this strict similarity test are sent to the final LLM to generate the answer.

This method acts as a mathematical gatekeeper, ensuring that only the documents that are most semantically aligned with the query make it through.

### The Analogy: The Relevancy Scanner

Imagine you ask your research assistant to find information on "corporate tax law."

*   **Standard Retriever:** The assistant brings you 10 books. Some are about tax law in general, one is about family tax, and a few are specifically about corporate tax.
*   **Embedding Filter:** The assistant gets the same 10 books. Before giving them to you, it puts each book through a "Relevancy Scanner" that is precisely tuned to "corporate tax law." The scanner immediately rejects the books on general and family tax, leaving you with only the most relevant ones.

This scanner doesn't need to *read* the books; it just checks their core theme (their embedding) against your request.

### Why Use an Embedding Filter?

*   **Efficiency:** It is extremely fast. It avoids the overhead of a second LLM call by reusing the embeddings that have already been calculated. This makes it one of the most performant ways to implement compression.
*   **Cost-Effective:** Because it doesn't use another LLM for the compression step, it saves on token costs.
*   **Tunable Precision:** You can easily adjust the `similarity_threshold`. A higher threshold (e.g., 0.85) makes the filter stricter, allowing only highly similar documents through. A lower threshold (e.g., 0.75) is more permissive.

While an LLM-based compressor can be more nuanced—extracting specific sentences—an Embedding Filter is a powerful and lightweight tool for quickly removing irrelevant documents from the context, leading to faster, cheaper, and more focused answers.
