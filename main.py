import argparse
import os
import pickle
from typing import Dict, List, Tuple
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from google import genai

load_dotenv()


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    k: int = 60,
    top_n: int = 5,
) -> List[Dict]:
    """Combines dense vector ranks and BM25 sparse ranks using Reciprocal Rank Fusion."""
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Dict] = {}

    # Rank dense results
    for rank, doc in enumerate(dense_results, start=1):
        content = doc["text"]
        doc_lookup[content] = doc
        scores[content] = scores.get(content, 0.0) + (1.0 / (k + rank))

    # Rank sparse results
    for rank, doc in enumerate(sparse_results, start=1):
        content = doc["text"]
        doc_lookup[content] = doc
        scores[content] = scores.get(content, 0.0) + (1.0 / (k + rank))

    # Sort documents by fused RRF score
    ranked_content = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return [doc_lookup[c] for c in ranked_content[:top_n]]


def query_bm25(bm25_data: dict, query: str, top_k: int = 10) -> List[Dict]:
    """Executes sparse keyword retrieval over the serialized BM25 corpus."""
    tokenized_query = query.lower().split()
    bm25 = bm25_data["bm25"]
    corpus_docs = bm25_data["documents"]
    corpus_metas = bm25_data["metadatas"]

    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Only retain actual keyword matches
            results.append({
                "text": corpus_docs[idx],
                "metadata": corpus_metas[idx],
                "score": scores[idx],
            })
    return results


def query_chroma_dense(collection, query: str, top_k: int = 10) -> List[Dict]:
    """Executes dense semantic retrieval over ChromaDB."""
    res = collection.query(query_texts=[query], n_results=top_k)
    results = []
    if res["documents"] and res["documents"][0]:
        for doc_text, meta in zip(res["documents"][0], res["metadatas"][0]):
            results.append({
                "text": doc_text,
                "metadata": meta,
            })
    return results


def build_prompt(query: str, context: List[str]) -> str:
    context_str = "\n\n---\n\n".join(context)
    return (
        f"You are a helpful assistant. Answer the user's question accurately using ONLY "
        f"the provided context excerpts. If the information is not present, say 'I am not sure'.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def main(
    collection_name: str = "documents_collection",
    persist_directory: str = "chroma_storage",
) -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not found.")

    genai_client = genai.Client(api_key=api_key)

    # 1. Align embedding model identically to load_data.py
    embedding_function = embedding_functions.GoogleGeminiEmbeddingFunction(
 model_name="gemini-embedding-001"
    )

    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    # 2. Load serialized BM25 index payload
    bm25_file_path = os.path.join(persist_directory, "bm25_index.pkl")
    with open(bm25_file_path, "rb") as f:
        bm25_data = pickle.load(f)

    print("Hybrid RRF Retriever initialized. Ready for queries.\n")

    while True:
        query = input("Query: ").strip()
        if not query:
            continue

        print("\nThinking...\n")

        # 3. Retrieve dense and sparse candidates
        dense_hits = query_chroma_dense(collection, query, top_k=10)
        sparse_hits = query_bm25(bm25_data, query, top_k=10)

        # 4. Fuse candidate lists via RRF
        fused_docs = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_n=5)

        context_texts = [d["text"] for d in fused_docs]
        source_files = list({d["metadata"].get("filename", "unknown") for d in fused_docs})

        # 5. Generate final response
        prompt = build_prompt(query, context_texts)
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        print(response.text or "No response.")
        print("\n" + "=" * 50)
        print("Source files:", source_files)
        print("Retrieved chunks:", len(context_texts))
        print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid Search (BM25 + Chroma) with RRF")
    parser.add_argument("--persist_directory", type=str, default="chroma_storage")
    parser.add_argument("--collection_name", type=str, default="documents_collection")
    args = parser.parse_args()

    main(
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )