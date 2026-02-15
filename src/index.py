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

print("Initialization OK")

def load_pages():
    pages = []
    with open(PAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages

def main():

    if not PAGES_PATH.exists():
        raise FileNotFoundError("pages.jsonl not found. Run ingest.py first.")

    print("Loading pages...")
    pages = load_pages()
    print(f"{len(pages)} pages loaded.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Initializing Chroma DB...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # Si la collection existe déjà, on la supprime (MVP simple)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    print("Creating embeddings and indexing...")

    batch_size = 64

    for i in range(0, len(pages), batch_size):

        batch = pages[i:i+batch_size]

        texts = [p["text"] for p in batch]
        embeddings = model.encode(texts).tolist()

        ids = [f'{p["doc"]}::p{p["page"]}' for p in batch]
        metadatas = [{"doc": p["doc"], "page": p["page"]} for p in batch]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Indexed {min(i+batch_size, len(pages))}/{len(pages)}")

    print("Indexing complete.")

if __name__ == "__main__":
    main()
