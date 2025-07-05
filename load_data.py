import os
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


def main(
    documents_directory: str = "documents",
    collection_name: str = "documents_collection",
    persist_directory: str = ".",
) -> None:
    # Read all files in the data directory
    documents = []
    metadatas = []
    files = os.listdir(documents_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for filename in files:
        with open(f"{documents_directory}/{filename}", "r", encoding="latin-1") as file:
            lines = file.read()

        docs = text_splitter.split_text(lines)
        for doc in docs:
            documents.append(doc)
            metadatas.append({"filename":filename})
    # Instantiate a persistent chroma client in the persist_directory.
    # Learn more at docs.trychroma.com
    client = chromadb.PersistentClient(path=persist_directory)

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not found.")

    genai.configure(api_key=google_api_key)

    # create embedding function
    embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=google_api_key, model_name="text-embedding-004"
    )

    # If the collection already exists, we just return it. This allows us to add more
    # data to an existing collection.
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    # Create ids from the current count
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # Load the documents in batches of 100
    for i in tqdm(
        range(0, len(documents), 100), desc="Adding documents", unit_scale=100
    ):
        collection.add(
            ids=ids[i : i + 100],
            documents=documents[i : i + 100],
            metadatas=metadatas[i : i + 100],  # type: ignore
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")


if __name__ == "__main__":
    # Read the data directory, collection name, and persist directory
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
    )

    # Add arguments
    parser.add_argument(
        "--data_directory",
        type=str,
        default="documents",
        help="The directory where your text files are stored",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="documents_collection",
        help="The name of the Chroma collection",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        default="chroma_storage",
        help="The directory where you want to store the Chroma collection",
    )

    # Parse arguments
    args = parser.parse_args()

    main(
        documents_directory=args.data_directory,
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )