# RAG: Searching with Maximal Marginal Relevance (MMR)

This document explains the concept of Maximal Marginal Relevance (MMR), a sophisticated search strategy to improve the quality of documents retrieved in a RAG system.

---

### The Problem: The Echo Chamber

A standard vector search is very good at finding documents that are similar to a query. In fact, it's so good that it often finds documents that are *very* similar to each other. This can create an "echo chamber" effect.

Imagine you ask, "What are the main themes of the president's speech?" A simple similarity search might return five document chunks that all discuss the exact same point about the economy. While highly relevant, this context is also highly redundant. You are showing the LLM the same piece of information five times, wasting valuable context space and potentially missing out on other important themes.

### The Solution: Balancing Relevance and Diversity

**Maximal Marginal Relevance (MMR)** is a search algorithm designed to solve this exact problem. It optimizes for two things simultaneously:

1.  **Relevance:** Finding documents that are closely related to the user's query.
2.  **Diversity:** Finding documents that are different from each other, providing a broader perspective.

MMR forces the search results to be more varied. Instead of just grabbing the top 5 most similar documents, it selects a mix of documents that are both relevant to the query and novel compared to the other selected documents.

### The Analogy: Building a Balanced Meal

Think of building a meal from a buffet.

*   **Standard Search (The Picky Eater):** You love chicken. You go to the buffet and fill your plate with five different kinds of chicken—fried chicken, grilled chicken, chicken nuggets, etc. Your plate is full, but you only have one type of food.
*   **MMR Search (The Nutritionist):** A nutritionist builds your plate. They start with a piece of grilled chicken because it's a great match for your preference. For the next item, they look for something that is also appealing but *different* from the chicken. They add some roasted vegetables. Then a salad. Then a piece of bread. The final plate has a variety of items that are all desirable but cover a wider range of food groups.

MMR acts like that nutritionist for your data, ensuring the context you feed the LLM is not just relevant, but also well-rounded and diverse.

### How It Works Under the Hood

MMR operates in a two-step process controlled by two key parameters:

1.  **`fetch_k`**: First, the system fetches a large number of documents that are relevant to the query (e.g., `fetch_k: 20`). This creates the initial pool of candidates.
2.  **`k`**: Then, it iterates through this pool to select a smaller, final set of documents (e.g., `k: 5`). It starts by picking the document most similar to the query. Then, for each subsequent pick, it calculates a score that is a combination of the document's similarity to the query and its dissimilarity to the documents already selected.

By tuning these parameters, you can control the trade-off between relevance and diversity, creating a context that is both informative and non-repetitive, leading to more comprehensive and accurate answers from the LLM.
