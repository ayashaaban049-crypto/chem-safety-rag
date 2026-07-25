"""
04_vector_representation.py
-----------------------------
Stage 4 of the RAG pipeline: VECTOR REPRESENTATION.

Uses BAAI/bge-small-en-v1.5 (lightweight, reliable on constrained hosting).
Arabic query support is handled upstream by translating the question to
English before retrieval (see 07_prompting.py), so this embedding model
does not need to be multilingual itself.

Run standalone (quick sanity check / smoke test):
    python 04_vector_representation.py
"""

import os

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

BGE_QUERY_INSTRUCTION = "Represent this question for retrieving relevant chemical safety information:"

_embedding_model = None  # simple module-level cache


def get_embedding_model() -> HuggingFaceBgeEmbeddings:
    """Return a cached BGE embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction=BGE_QUERY_INSTRUCTION,
        )
        print(f"[04_vector_representation] Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    return _embedding_model


if __name__ == "__main__":
    model = get_embedding_model()
    sample_query = "How do I handle a sulfuric acid spill?"
    sample_doc = "Sulfuric acid spills should be absorbed with dry sand or an inert absorbent."
    q_vec = model.embed_query(sample_query)
    d_vec = model.embed_documents([sample_doc])[0]
    print(f"[04_vector_representation] Query embedding dim: {len(q_vec)}")
    print(f"[04_vector_representation] Doc embedding dim:   {len(d_vec)}")
