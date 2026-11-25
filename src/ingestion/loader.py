"""
loader.py
Loads documents (PDF/TXT) from a directory or a single file path.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def load_file(file_path: str | Path, display_name: Optional[str] = None) -> List[Document]:
    """Load one PDF or TXT file. Pass display_name for uploads saved to temp files."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    name = display_name or path.name
    if suffix == ".pdf":
        docs = PyPDFLoader(str(path)).load()
    else:
        docs = TextLoader(str(path), encoding="utf-8").load()

    kept = []
    for doc in docs:
        # Collapse the line break between every word that some PDFs produce
        doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
        # Optional: remove the repeated header from this specific PDF
        doc.page_content = doc.page_content.replace("Prepared by: Binilraj Adhikari", "").strip()
        if not doc.page_content:
            continue  # blank page (often a scanned PDF)
        doc.metadata["source"] = name
        kept.append(doc)

    if not kept:
        logger.warning(f"{name}: no extractable text (scanned PDF or empty file?)")
    return kept

def load_documents(data_dir: str | Path) -> List[Document]:
    """Load all supported files under a directory (recursively)."""
    data_path = Path(data_dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    documents: List[Document] = []
    for file in sorted(data_path.rglob("*")):
        if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                documents.extend(load_file(file))
            except Exception as e:
                logger.error(f"Failed to load {file.name}: {e}")

    if not documents:
        logger.warning(f"No readable documents found in {data_path}")
    else:
        logger.info(f"Loaded {len(documents)} page(s)/document(s) from {data_path}")
    return documents


if __name__ == "__main__":
    docs = load_documents("data")
    print(f"Loaded {len(docs)} documents.")
    if docs:
        print("Example document:", docs[0])