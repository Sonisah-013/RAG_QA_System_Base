"""
loader.py
Loads documents (PDF/TXT) from a directory or a single file path.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_documents(data_dir: str | Path) -> List[Document]:
    """Load all PDF and TXT files from a directory (recursively)."""
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    documents: List[Document] = []

    for pdf_file in data_path.glob("**/*.pdf"):
        try:
            documents.extend(PyPDFLoader(str(pdf_file)).load())
        except Exception as e:
            logger.error(f"Failed to load PDF {pdf_file}: {e}")

    for txt_file in data_path.glob("**/*.txt"):
        try:
            documents.extend(TextLoader(str(txt_file)).load())
        except Exception as e:
            logger.error(f"Failed to load TXT {txt_file}: {e}")

    if not documents:
        logger.warning(f"No documents found in {data_path}")
    else:
        logger.info(f"Loaded {len(documents)} document(s) from {data_path}")

    return documents


def load_single_pdf(file_path: str | Path) -> List[Document]:
    """Load one PDF file directly — used by the Streamlit upload flow."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    docs = PyPDFLoader(str(path)).load()
    logger.info(f"Loaded {len(docs)} page(s) from {path.name}")
    return docs


if __name__ == "__main__":
    docs = load_documents("data")
    print(f"Loaded {len(docs)} documents.")
    if docs:
        print("Example document:", docs[0])