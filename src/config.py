"""
config.py
Central place for pipeline constants. Adjust these instead of hunting
through every module when you want to tune chunking, models, or paths.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_PATH = BASE_DIR / "faiss_index"

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Models ---
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2

# --- Retrieval ---
TOP_K = 4

# --- API key ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print(
        "[WARNING] OPENAI_API_KEY not found in environment. "
        "Add it to your .env file before running the pipeline."
    )