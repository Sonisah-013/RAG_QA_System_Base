"""
ingest.py
Reads every PDF/TXT in data/, splits it into chunks, and builds the FAISS index.

Run from the project root:
    python ingest.py
"""

import logging

from src.ingestion.loader import load_documents
from src.ingestion.chunker import split_documents
from src.retrieval.vectorstore import create_vectorstore

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main():
    docs = load_documents("data")
    if not docs:
        print("No documents found in data/. Add a PDF or TXT file and try again.")
        return

    chunks = split_documents(docs)
    if not chunks:
        print("Documents were loaded but produced no text chunks (scanned PDF?).")
        return

    create_vectorstore(chunks)
    print(f"\nDone: indexed {len(chunks)} chunks from {len(docs)} page(s)/document(s).")


if __name__ == "__main__":
    main()