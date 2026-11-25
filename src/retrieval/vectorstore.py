"""
vectorstore.py
Creates and loads the local FAISS vector store using local HuggingFace embeddings.
"""

import logging
from pathlib import Path
import warnings

# Suppress DeprecationWarnings from langchain_community
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Fix import: reference src.embedding.embedder instead of src.retrieval.embeddings
from src.embedding.embedder import get_embeddings

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INDEX_PATH = Path("faiss_index")


def create_vectorstore(documents: list[Document]) -> FAISS:
    """Builds and saves a FAISS vector store locally."""
    embeddings = get_embeddings()
    logger.info(f"Creating FAISS index with {len(documents)} documents...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    vectorstore.save_local(str(INDEX_PATH))
    logger.info(f"FAISS index saved locally at: {INDEX_PATH.resolve()}")
    return vectorstore


def load_vectorstore() -> FAISS:
    """Loads an existing local FAISS vector store."""
    embeddings = get_embeddings()
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}. Run ingestion first."
        )
    
    logger.info(f"Loading FAISS index from: {INDEX_PATH.resolve()}")
    return FAISS.load_local(
        str(INDEX_PATH), 
        embeddings, 
        allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    sample_docs = [
        Document(page_content="RAG stands for Retrieval-Augmented Generation."),
        Document(page_content="FAISS is a library for efficient similarity search.")
    ]
    create_vectorstore(sample_docs)