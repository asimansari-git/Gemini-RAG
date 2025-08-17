# Learning RAG: A Journey Through the Branches

Welcome, traveler, to a hands-on guide for understanding Retrieval-Augmented Generation (RAG). Think of this repository as a living textbook where each branch is a chapter, demonstrating a specific concept that builds on the last. We'll go from the simple to the sophisticated, one branch at a time.

---

### How to Use This Repository

This repository is designed to be explored. Each branch contains a complete, working example of a RAG technique. To get started, you'll want to bring the code to your local machine.

**1. Clone the Repository**

First, clone this repository to your computer using the following command:

```bash
git clone <repository-url>
```

**2. See All the Chapters (Branches)**

Once cloned, navigate into the directory and see all the available branches. Each one is a different lesson.

```bash
git branch -a
```

**3. Travel to a Chapter**

You can switch to any branch to see the code and documentation for that specific technique. For example, to start at the beginning, you would check out the `similarity-search` branch:

```bash
git checkout similarity-search
```

---

### The Learning Path: Your Itinerary

We recommend exploring the branches in the following order. It's like a guided tour from the fundamentals to more advanced, powerful techniques.

1.  **Start Here: `similarity-search` (The Foundation)**
    *   **The Idea:** This is the bedrock of RAG. We'll explore how to find documents based on the similarity of their meaning to a user's question.
    *   **Analogy:** The magical librarian who instantly finds books based on the *ideas* you're thinking of, not just the words you say.
    *   **Read:** `Similarity_Search.md`

2.  **Next Stop: `mmr` (The Nutritionist)**
    *   **The Idea:** Avoid getting the same information over and over. MMR helps us find documents that are not only relevant but also *different* from each other.
    *   **Analogy:** Building a balanced meal instead of just a plate full of chicken. We want variety!
    *   **Read:** `MMR.md`

3.  **Level Up: `self-query` (The Super-Smart Librarian)**
    *   **The Idea:** Let the LLM itself figure out how to best search the database. It translates a natural question into a structured, filtered query automatically.
    *   **Analogy:** The librarian who hears you mumble about "that speech from last year" and knows exactly which document you mean.
    *   **Read:** `Self_Query_Explained.md` and `Self_Query_Under_The_Hood.md`

4.  **Get Efficient: `compression` (The Research Assistant)**
    *   **The Idea:** Retrieve a broad set of documents, then use a model to "compress" them down to only the most potent, relevant sentences before the final answer is generated.
    *   **Analogy:** The assistant who reads 10 books for you and returns with a single page of highlighted, relevant paragraphs.
    *   **Read:** `Contextual_Compression.md`

5.  **Get *Really* Efficient: `compression-filter` (The Relevancy Scanner)**
    *   **The Idea:** A faster, more direct way to compress. This method uses math (embedding similarity) to instantly discard retrieved documents that aren't relevant enough.
    *   **Analogy:** A high-tech scanner that instantly rejects books that don't match the core theme of your request.
    *   **Read:** `Embedding_Compression.md`

### Additional Wisdom

*   `RAG_Techniques.md`: A high-level map of all the concepts covered.
*   `ChromaDB_vs_LangChain.md`: Explains why we use different tools for different jobs (the engine vs. the chassis).