"""
vectorstore.py
Builds, saves, and loads the FAISS vector index.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Package import fix
from src.retrieval.embeddings import get_embeddings

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INDEX_PATH = "faiss_index"


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Embed chunks and build a fresh in-memory FAISS index."""
    if not chunks:
        raise ValueError("Cannot build a vectorstore from an empty list of chunks")

    embeddings = get_embeddings()
    logger.info(f"Building FAISS index from {len(chunks)} chunk(s)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    logger.info("FAISS index built successfully")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, index_path: str | Path = DEFAULT_INDEX_PATH) -> None:
    """Persist a FAISS index to disk."""
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    logger.info(f"Saved FAISS index to {index_path}")


def load_vectorstore(index_path: str | Path = DEFAULT_INDEX_PATH) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {path}. Build one first with build_vectorstore()."
        )

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,  # required for local FAISS pickle loading
    )
    logger.info(f"Loaded FAISS index from {path}")
    return vectorstore


if __name__ == "__main__":
    from src.ingestion.ingestion import load_and_chunk  # Adjust module name to match your file structure

    chunks = load_and_chunk("data")
    if chunks:
        vs = build_vectorstore(chunks)
        save_vectorstore(vs)