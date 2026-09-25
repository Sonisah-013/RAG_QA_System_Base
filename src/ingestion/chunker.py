"""
chunker.py
Splits loaded documents into overlapping chunks ready for embedding.
"""

import logging
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split a list of Document objects into smaller overlapping chunks."""
    if not documents:
        logger.warning("No documents to split — returning empty list")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s)")
    return chunks


if __name__ == "__main__":
    from loader import load_documents

    docs = load_documents("data")
    chunks = split_documents(docs)
    if chunks:
        print(f"\nSample chunk:\n{'-' * 40}")
        print(chunks[0].page_content[:300])
        print(f"{'-' * 40}\nMetadata: {chunks[0].metadata}")