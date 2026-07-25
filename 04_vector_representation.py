"""
04_vector_representation.py
-----------------------------
Stage 4 of the RAG pipeline: VECTOR REPRESENTATION.

Wraps a Hugging Face multilingual embedding model (intfloat/multilingual-e5-small
by default) behind a single get_embedding_model() function so every other
stage (store creation + retrieval) shares one consistent embedding function.
This model supports both Arabic and English queries.

Run standalone (quick sanity check / smoke test):
    python 04_vector_representation.py
"""

import os

from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")

_embedding_model = None  # simple module-level cache


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a cached multilingual embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
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
