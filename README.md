# Advanced RAG: Contextual Compression

This document explains the concept of Contextual Compression in a RAG pipeline, focusing on the `LLMChainExtractor` method and the critical tuning required to make it work effectively.

---

### The Goal: Compressing the Context

In a RAG pipeline, "compression" does not mean creating a ZIP file. It means **"compressing the context"** that is sent to the final, answering LLM. The goal is to refine the retrieved documents, removing noise, fluff, and irrelevant information to provide the final LLM with a smaller, more potent, and highly relevant set of information. This leads to faster, cheaper, and often more accurate answers.

### The `LLMChainExtractor` (The "Editor")

The `LLMChainExtractor` is a powerful but aggressive compression technique. It works by iterating through each document retrieved by the base retriever and using an LLM to "edit" it.

Here's the process for each document:

1.  The document's content is passed to an LLM (e.g., `gemini-2.0-flash-lite`).
2.  The LLM is given a prompt that essentially asks, **"Given the user's original question, are there any sentences in this document that are directly relevant? If so, return only those sentences. If not, return nothing."**
3.  The output of the LLM (either the extracted sentences or an empty string) replaces the original document.

### The Challenge: The Overly Aggressive Editor

A common problem with this method is that the compressor can be *too* aggressive. If the LLM decides that none of the retrieved documents contain a "perfect" answer, it might return an empty string for all of them. This leaves the final answering LLM with no context, leading to poor results.

The number of documents returned by the `LLMChainExtractor` is a *result* of its process, not a parameter we can directly control.

### The Solution: A Well-Behaved Pipeline

While it's possible for the `LLMChainExtractor` to be too aggressive, it often works well with the default settings of the base retriever, especially when the base retriever itself is powerful (like the `SelfQueryRetriever`). The key is that the base retriever provides a high-quality, relevant set of documents for the compressor to work with.

If results are poor, one potential tuning method is to configure the base retriever to fetch more documents (e.g., using `search_kwargs={'k': 10}`). However, starting with the defaults is recommended.

**Example Implementation:**

```python
# In main.py

# 1. Create the base retriever with default settings
base_retriever = SelfQueryRetriever.from_llm(
    llm,
    vector_store,
    document_content_description,
    metadata_field_info,
)

# 2. Create the compressor
compressor = LLMChainExtractor.from_llm(llm)

# 3. Create the final compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 4. Invoke the retriever
# The pipeline will fetch documents, then compress them.
retrieved_docs = compression_retriever.invoke(query)
```

This pipeline effectively uses an LLM to first understand the query and apply filters, and then uses an LLM again to refine the retrieved content, providing a high-quality, focused context for the final answer.
