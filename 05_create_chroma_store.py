"""
05_create_chroma_store.py
---------------------------
Stage 5 of the RAG pipeline: VECTOR STORE.

Takes the chunks produced by 03_chunking.py, embeds them with the BGE
model from 04_vector_representation.py, and persists them to a local
ChromaDB collection on disk.

Run standalone (builds/rebuilds the store):
    python 05_create_chroma_store.py
"""

import json
import importlib
from pathlib import Path

from langchain_community.vectorstores import Chroma

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

vector_rep = importlib.import_module("04_vector_representation")
get_embedding_model = vector_rep.get_embedding_model

CHUNKS_PATH = Path(__file__).parent / "artifacts" / "03_chunks.json"
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "chemical_safety_cards"


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunks_to_documents(chunks: list[dict]) -> list[Document]:
    docs = []
    for chunk in chunks:
        meta = {k: v for k, v in chunk["metadata"].items() if v is not None}
        meta["chunk_id"] = chunk["chunk_id"]
        docs.append(Document(page_content=chunk["text"], metadata=meta))
    return docs


def build_vector_store(
    docs: list[Document],
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    persist_dir.mkdir(parents=True, exist_ok=True)
    embedding_model = get_embedding_model()

    # Chroma writes in batches internally; for a few hundred/thousand chunks
    # a single call is fine for a lightweight lab-scale project.
    store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    print(f"[05_create_chroma_store] Persisted {len(docs)} chunks -> {persist_dir}")
    return store


def load_vector_store(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Re-open an already-built store without re-embedding anything."""
    embedding_model = get_embedding_model()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(persist_dir),
    )


if __name__ == "__main__":
    chunks = load_chunks()
    documents = chunks_to_documents(chunks)
    build_vector_store(documents)
