"""config.py - single source of truth for pipeline settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths (project root = one level above src/) ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_PATH = BASE_DIR / "faiss_index"

# --- Chunking (characters) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Models ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"   # if you switch to Groq; check current model names
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 512

# --- Retrieval ---
TOP_K = 4
SCORE_THRESHOLD = 0.3                # tune on your own documents

# --- API keys (read lazily where needed, don't warn at import) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")