"""
retriever.py
Wraps a FAISS vectorstore as a retriever for use in the QA chain.
"""

import logging

from langchain_community.vectorstores import FAISS

from src.retrieval.vectorstore import load_vectorstore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4


def get_retriever(vectorstore: FAISS = None, k: int = DEFAULT_TOP_K):
    """
    Return a retriever from the vectorstore.
    If no vectorstore is passed, loads the saved index from disk.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_docs(query: str, vectorstore: FAISS = None, k: int = DEFAULT_TOP_K):
    """Run a query directly and return matching documents (no LLM call)."""
    if not query or not query.strip():
        raise ValueError("Query must not be empty")

    retriever = get_retriever(vectorstore, k=k)
    results = retriever.invoke(query)
    logger.info(f"Retrieved {len(results)} chunk(s) for query: '{query}'")
    return results


if __name__ == "__main__":
    test_query = input("Enter a test query: ")
    results = retrieve_docs(test_query)
    print(f"\nTop {len(results)} results:\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(doc.page_content[:300], "...")
        print(f"Source: {doc.metadata.get('source', 'unknown')}\n")