import argparse
import os
from typing import List
from dotenv import load_dotenv
from google import genai
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()


def build_prompt(query: str, context: List[str]) -> str:
    """
    Builds a prompt for the LLM.

    This function builds a prompt for the LLM. It takes the original query,
    and the returned context, and asks the model to answer the question based only
    on what's in the context, not what's in its weights.

    Args:
        query (str): The original query.
        context (List[str]): The context of the query, returned by embedding search.

    Returns:
        A prompt for the LLM (str).
    """
    base_prompt = {
        "content": (
            "I am going to ask you a question, which I would like you to answer"
            " based only on the provided context, and not any other information."
            " If there is not enough information in the context to answer the question,"
            ' say "I am not sure", then try to make a guess.'
            " Break your answer up into nicely readable paragraphs."
        )
    }
    user_prompt = {
        "content": f" The question is '{query}'. Here is all the context you have: {' '.join(context)}"
    }

    # combine the prompts to output a single prompt string
    return f"{base_prompt['content']} {user_prompt['content']}"


def get_gemini_response(
    client: genai.Client, query: str, context: List[str], model: str = "gemini-2.5-pro"
) -> str:
    """
    Queries the Gemini API to get a response to the question using the google-genai SDK.

    Args:
        client (genai.Client): Initialized Google GenAI client.
        query (str): The original query.
        context (List[str]): The context of the query, returned by embedding search.
        model (str): Gemini model identifier.

    Returns:
        A response to the question.
    """
    response = client.models.generate_content(
        model=model,
        contents=build_prompt(query, context),
    )
    return response.text or ""


def main(
    collection_name: str = "documents_collection", persist_directory: str = "chroma_storage"
) -> None:
    # Resolve API Key for Google Gen AI
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not found.")

    # Initialize Google Gen AI client
    genai_client = genai.Client(api_key=api_key)

    # Instantiate a persistent chroma client in the persist_directory.
    client = chromadb.PersistentClient(path=persist_directory)

    # Modern Google Gen AI embedding function using google-genai SDK
    embedding_function = embedding_functions.GoogleGeminiEmbeddingFunction(
        api_key=api_key, model_name="text-embedding-004"
    )

    # Get the collection.
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    # Simple interactive query loop.
    while True:
        query = input("Query: ")
        if len(query.strip()) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue
        print("\nThinking...\n")

        # Query the collection to get the 5 most relevant results
        results = collection.query(
            query_texts=[query], n_results=5, include=["documents", "metadatas"]
        )

        if "chunks" in collection_name:
            sources = "\n".join(
                set(result.get("filename", "unknown") for result in results["metadatas"][0])
            )
        else:
            sources = "\n".join(
                [
                    f"{result.get('filename', 'unknown')}: line {result.get('line_number', 'N/A')}"
                    for result in results["metadatas"][0]  # type: ignore
                ]
            )

        # Get the response from Gemini using google-genai Client
        response = get_gemini_response(genai_client, query, results["documents"][0])  # type: ignore

        # Output, with sources
        print(response)
        print("\n")
        print(f"Source documents:\n{sources}")
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run similarity search RAG with Chroma and Gemini"
    )

    parser.add_argument(
        "--persist_directory",
        type=str,
        default="chroma_storage",
        help="The directory where you want to store the Chroma collection",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="documents_collection",
        help="The name of the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )
