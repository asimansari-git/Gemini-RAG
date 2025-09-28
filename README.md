# RAG: Precise Searching with Metadata Filtering

This document explains the concept of Metadata Filtering, a powerful technique for dramatically improving the accuracy and efficiency of your RAG system.

---

### The Concept: Searching in the Right Aisle

So far, our search methods have looked through the *entire* library of documents to find the most relevant results. This is effective, but can be inefficient if we already know something about the document we're looking for.

**Metadata** is extra information we store alongside our document chunks. It's data *about* the data. Common examples include:
*   The `filename` the chunk came from.
*   The `date` the document was created.
*   The `author` or `source`.
*   A `category` or `tag`.

**Metadata Filtering** is the process of telling the database to *only* search within documents that match specific metadata criteria. It allows us to narrow the search space *before* the vector similarity search even begins.

### The Analogy: The Librarian with an Index

Imagine you go to a massive library and ask the librarian for information on "the economy."

*   **Without Filtering:** The librarian has to wander through the entire library—fiction, history, science—to find books that match your query.
*   **With Metadata Filtering:** You can be more specific. You say, "Find me information on 'the economy,' but only look in the 'Newspapers from 2023' section." The librarian doesn't waste a second. They go directly to the correct aisle (the one labeled "Newspapers, 2023") and *then* start looking for books about the economy. The search is faster and the results are guaranteed to be from the correct source.

Metadata filtering is like giving your librarian a perfect index to the library, allowing them to instantly jump to the right section before they even start reading titles.

### How It Works

In `main.py` of this branch, we implement this directly. We first ask the user if they want to apply a filter:

```python
filename_filter = input("Filter by filename (optional, press enter to skip): ").strip()
```

If the user provides a filename, we pass it directly into the search call within a `filter` dictionary:

```python
retrieved_docs = vector_store.max_marginal_relevance_search(
    query=query,
    k=5,
    filter={"filename": filename_filter}
)
```

This `filter` parameter is a powerful instruction to the vector database. It says, "Ignore everything else first. Only consider documents where the `filename` metadata is an exact match for what the user provided. *Then*, within that small subset, run your vector search."

This technique is a crucial step in building more advanced and user-responsive RAG systems, as it allows you to combine the power of semantic vector search with the precision of structured database queries.

Continue with [self-query](https://github.com/asimibnakhlaque/Gemini-RAG/tree/self-query)
