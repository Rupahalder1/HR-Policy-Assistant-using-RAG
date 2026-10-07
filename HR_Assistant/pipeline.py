"""Wires all the components together into one ready-to-use agent.

This is the single entry point that main.py (CLI) and app.py (Streamlit)
both call. Each step is handled by its own small module.
"""


import config

from agent import create_hr_agent
from document_loader import load_document
from llm import get_llm
from splitter import split_into_chunks
from tools import create_search_tool

from vector_store import (
    build_vector_store,
    save_vector_store,
    get_retriever,
    load_vector_store,
    vector_store_exists,
)


# ============================================================
# DATA INGESTION
# ============================================================

def build_vector_store_for_document(
    file_path: str = config.DATA_FILE_PATH
):
    """
    Load + split + embed the document.

    If a FAISS vector store already exists on disk,
    load it instead of rebuilding it.
    """

    if vector_store_exists():

        print(
            "Found a saved vector store on disk, "
            "loading it (fast, no re-embedding)."
        )

        return load_vector_store()

    print(
        "No saved vector store, "
        "building one from scratch..."
    )

    documents = load_document(file_path)

    chunks = split_into_chunks(documents)

    print(
        f"Loaded '{file_path}' and split it "
        f"into {len(chunks)} chunks."
    )

    vector_store = build_vector_store(chunks)

    save_vector_store(vector_store)

    print(
        "Vector store built and saved to disk "
        "for next time."
    )

    return vector_store


# ============================================================
# BUILD HR ASSISTANT
# ============================================================

def build_HR_Assistant(
    file_path: str = config.DATA_FILE_PATH
):
    """
    Build the full RAG agent, ready to answer questions.
    """

    config.check_api_keys()

    # 1. Load/build vector store
    vector_store = build_vector_store_for_document(
        file_path
    )

    # 2. Create retriever
    retriever = get_retriever(vector_store)

    # 3. Create search tool
    search_tool = create_search_tool(retriever)

    # 4. Load LLM
    llm = get_llm()

    # 5. Create agent
    agent = create_hr_agent(
        llm,
        [search_tool]
    )

    return agent


# ============================================================
# ASK THE ASSISTANT
# ============================================================

def ask(
    agent,
    question: str
) -> str:
    """
    Ask the agent a question and return
    its final answer as plain text.
    """

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    return response["messages"][-1].content
