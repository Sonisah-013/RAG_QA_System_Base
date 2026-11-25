"""
qa_chain.py
Builds the Retrieval QA chain (LCEL) with Groq and exposes ask_question(),
which returns the answer and the source chunks used.
"""

import logging
import os
from typing import Any, List, Tuple

from dotenv import load_dotenv

load_dotenv()

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq

from src.generation.prompts import qa_prompt
from src.retrieval.retriever import get_retriever
from src.retrieval.vectorstore import load_vectorstore

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
NO_ANSWER_PREFIX = "I don't have enough information"


def build_qa_chain(vectorstore) -> Any:
    """Build the retrieval QA chain using Groq."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment. "
            "Add it to your .env file, e.g. GROQ_API_KEY=gsk_..."
        )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1500,
        api_key=api_key,
    )

    retriever = get_retriever(vectorstore)
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    logger.info(f"QA chain ready (Groq model: {GROQ_MODEL})")
    return chain


def ask_question(chain: Any, question: str) -> Tuple[str, List[str]]:
    """Run a question through the chain. Returns (answer, sources)."""
    if not question or not question.strip():
        raise ValueError("Question must not be empty")

    try:
        result = chain.invoke({"input": question})
    except Exception as e:
        logger.error(f"QA chain failed: {e}")
        raise

    answer = result["answer"].strip()

    # No sources when the model says it couldn't find the answer
    if answer.startswith(NO_ANSWER_PREFIX):
        return answer, []

    sources: List[str] = []
    seen = set()
    for doc in result.get("context", []):
        name = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{name}, page {page + 1}" if isinstance(page, int) else name
        snippet = doc.page_content[:200].replace("\n", " ").strip()
        key = (label, snippet)
        if key in seen:
            continue
        seen.add(key)
        sources.append(f"[{label}] {snippet}...")

    logger.info(f"Answered using {len(sources)} source chunk(s)")
    return answer, sources


if __name__ == "__main__":
    vs = load_vectorstore()
    qa_chain = build_qa_chain(vs)

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break
        ans, _ = ask_question(qa_chain, query)
        print(f"\nAnswer: {ans}")