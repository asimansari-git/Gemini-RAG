import argparse
import os
from typing import List
from dotenv import load_dotenv

import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.globals import set_debug
set_debug(True)

load_dotenv()

model = genai.GenerativeModel("gemini-2.5-pro")


def build_prompt(query: str, context: List[str]) -> str:
    """
    Builds a prompt for the LLM. #

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
        "content": "I am going to ask you a question, which I would like you to answer"
        " based only on the provided context, and not any other information."
        " If there is not enough information in the context to answer the question,"
        ' say "I am not sure", then try to make a guess.'
        " Break your answer up into nicely readable paragraphs.",
    }
    user_prompt = {
        "content": f" The question is '{query}'. Here is all the context you have:"
        f'{(" ").join(context)}',
    }

    # combine the prompts to output a single prompt string
    system = f"{base_prompt['content']} {user_prompt['content']}"

    return system


def get_gemini_response(query: str, context: List[str]) -> str:
    """
    Queries the Gemini API to get a response to the question.

    Args:
    query (str): The original query.
    context (List[str]): The context of the query, returned by embedding search.

    Returns:
    A response to the question.
    """

    response = model.generate_content(build_prompt(query, context))

    return response.text


def main(
        collection_name: str = "documents", persist_directory: str = "chroma_storage"
) -> None:
    # Check if the GOOGLE_API_KEY environment variable is set. Prompt the user to set it if not.
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not found.")

    genai.configure(api_key=google_api_key)

    # 1. Instantiate the Google Generative AI embedding function
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=google_api_key
    )

    # 2. Instantiate the Chroma vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name
    )

    # 3. Define the metadate fields that the Self-Query Retriever can use
    metadata_field_info = [
        AttributeInfo(
            name="filename",
            description="The name of the file the chunk text is from. For example, `state_of_the_union_2022.txt`",
            type="string"
        )
    ]

    # 4. Define the LLM that will power the self-querying logic
    llm = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it", google_api_key=google_api_key)

    # 5. Create the Self-Query retriever
    document_content_description = "This content of a State of the Union address"
    retriever = SelfQueryRetriever.from_llm(
        llm,
        vector_store,
        document_content_description,
        metadata_field_info,
        verbose=True #To see the generated queries
    )
    # We use a simple input loop.
    while True:
        # Get the user's query
        query = input("Query: ")
        if len(query) == 0:
            print("Please enter a question. Ctrl+C to Quit.\n")
            continue

        print("\nThinking...\n")

        # The retriever now does all the work of parsing the query
        retrieved_docs = retriever.invoke(query)

        # Extracting the context out of docs
        context = [doc.page_content for doc in retrieved_docs]

        sources = "\n".join(
            set(doc.metadata['filename'] for doc in retrieved_docs)
        )

        # Get the response from Gemini
        response = get_gemini_response(query, context)  # type: ignore

        # Output, with sources
        print(response)
        print("\n")
        print(f"Source documents:\n{sources}")
        print(f"Context documents:\n{context}")
        print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load documents from a directory into a Chroma collection"
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
