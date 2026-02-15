import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path("data/processed/chroma")
COLLECTION_NAME = "airliquide_pages"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def make_snippet(text: str, n: int = 350) -> str:
    """Réduit le texte pour afficher un extrait lisible."""
    one_line = " ".join(text.split())
    return (one_line[:n] + "…") if len(one_line) > n else one_line


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/query.py "dette nette 2024"')

    query = sys.argv[1]
    top_k = 5

    if not CHROMA_DIR.exists():
        raise SystemExit(f"Chroma DB introuvable: {CHROMA_DIR}. Lance d'abord: python src/index.py")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Opening Chroma DB...")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=COLLECTION_NAME)

    print("Embedding query...")
    q_emb = model.encode([query]).tolist()[0]

    print("Searching...")
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    print(f"\nQuery: {query}\n")
    for rank, (doc_text, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        # distance: plus petit = plus proche (selon métrique)
        print(f"{rank}) {meta['doc']} — p.{meta['page']} (distance={dist:.4f})")
        print(f"   {make_snippet(doc_text)}\n")


if __name__ == "__main__":
    main()