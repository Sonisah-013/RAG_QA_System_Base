"""
qa_chain.py
Builds the Retrieval QA chain using LCEL and exposes ask_question(), 
which returns both the answer and the source chunks used to generate it.
"""

import logging
import os
import warnings
from typing import Any, List, Tuple

# Suppress deprecation warnings from community packages
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from src.generation.prompts import qa_prompt
from src.retrieval.retriever import get_retriever
from src.retrieval.vectorstore import load_vectorstore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2


def build_qa_chain(vectorstore: FAISS) -> Any:
    """Build a modern Retrieval QA chain using LCEL."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Add it to your .env file, e.g. OPENAI_API_KEY=sk-..."
        )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=api_key)
    retriever = get_retriever(vectorstore)

    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    logger.info(f"Built QA chain using model: {LLM_MODEL}")
    return retrieval_chain


def ask_question(chain: Any, question: str) -> Tuple[str, List[str]]:
    """
    Run a question through the QA chain.

    Returns:
        answer: the generated answer string
        sources: list of short source snippets used to ground the answer
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty")

    try:
        result = chain.invoke({"input": question})
    except Exception as e:
        logger.error(f"QA chain failed: {e}")
        raise

    answer = result["answer"]
    source_docs = result.get("context", [])

    sources = []
    for doc in source_docs:
        page = doc.metadata.get("page", "?")
        source_name = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:200].replace("\n", " ").strip()
        sources.append(f"[{source_name}, page {page}] {snippet}...")

    logger.info(f"Answered question using {len(sources)} source chunk(s)")
    return answer, sources


if __name__ == "__main__":
    vs = load_vectorstore()
    qa_chain = build_qa_chain(vs)

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break
        ans, srcs = ask_question(qa_chain, query)
        print(f"\nAnswer: {ans}")
        print("\nSources:")
        for s in srcs:
            print(f"  - {s}")