"""
embedder.py
Initializes the embedding model used to build/query the vector index.
"""

import logging
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> OpenAIEmbeddings:
    """
    Initialize the OpenAI embedding model.

    Raises:
        ValueError: if OPENAI_API_KEY is not set in the environment.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Add it to your .env file, e.g. OPENAI_API_KEY=sk-..."
        )

    logger.info(f"Loading embedding model: {model_name}")
    return OpenAIEmbeddings(model=model_name, api_key=api_key)


def embed_text(text: str, model_name: str = EMBEDDING_MODEL) -> list[float]:
    """Embed a single string directly. Useful for quick debugging/tests."""
    embeddings = get_embeddings(model_name)
    vector = embeddings.embed_query(text)
    logger.info(f"Embedded text into a {len(vector)}-dim vector")
    return vector


if __name__ == "__main__":
    sample = "This is a test sentence for embedding."
    vec = embed_text(sample)
    print(f"Sample embedding (first 5 dims): {vec[:5]}")
    print(f"Total dimensions: {len(vec)}")