import json
import chromadb # Base vectorielle open-source Chroma
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer # SBERT Embedding
from pathlib import Path

# Path to every pages of the source document
PAGES_PATH = Path("data/processed/pages.jsonl")
CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "airliquide_pages"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_pages():
    pages = []
    with open(PAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages
