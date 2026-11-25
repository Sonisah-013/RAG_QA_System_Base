"""
embedder.py
Initializes local HuggingFace embeddings for free vector operations.
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    logger.info(f"Loading local embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)


def embed_text(text: str) -> list[float]:
    embeddings = get_embeddings()
    vector = embeddings.embed_query(text)
    logger.info(f"Embedded text into a {len(vector)}-dim vector")
    return vector


if __name__ == "__main__":
    sample = "This is a test sentence for embedding."
    vec = embed_text(sample)
    print(f"Sample embedding (first 5 dims): {vec[:5]}")
    print(f"Total dimensions: {len(vec)}")