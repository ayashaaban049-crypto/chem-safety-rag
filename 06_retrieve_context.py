"""
06_retrieve_context.py
------------------------
Stage 6 of the RAG pipeline: CONTEXT RETRIEVAL.

Implements a lightweight hybrid retriever:
  1. Dense retrieval  - BGE embedding similarity search against Chroma.
  2. Keyword boost    - if the query mentions a chemical name that also
                         appears in a chunk's metadata, that chunk's score
                         is boosted so the exact chemical's own card is
                         prioritized over a merely "semantically similar"
                         chemical.
  3. Section awareness - queries containing words like "spill", "PPE",
                         "first aid", "storage", "dispose" are nudged
                         towards chunks tagged with the matching ICSC
                         section, since these are the emergency-relevant
                         sections lab users search for.

Run standalone (interactive smoke test):
    python 06_retrieve_context.py
"""

import importlib
import re

store_module = importlib.import_module("05_create_chroma_store")
load_vector_store = store_module.load_vector_store

TOP_K = 5
FETCH_K = 20  # over-fetch, then re-rank
import os

# Minimum combined score (similarity + boosts) a chunk must reach to be
# considered relevant enough to answer from. Chunks below this are rejected
# rather than being returned just because they were in the Top-K.
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.35"))
SECTION_KEYWORDS = {
    "spill": ["Spillage Disposal", "Spillage"],
    "leak": ["Spillage Disposal", "Spillage"],
    "ppe": ["Prevention"],
    "protective": ["Prevention"],
    "wear": ["Prevention"],
    "first aid": ["First Aid"],
    "eyes": ["First Aid", "Eyes"],
    "skin": ["First Aid", "Skin"],
    "inhal": ["First Aid", "Inhalation"],
    "swallow": ["First Aid", "Ingestion"],
    "fire": ["Fire", "Fire Fighting", "Firefighting"],
    "storage": ["Storage"],
    "store": ["Storage"],
    "dispos": ["Spillage Disposal", "Storage"],
    "waste": ["Spillage Disposal", "Storage"],
    "label": ["Packaging & Labelling", "Packaging And Labelling"],
    "exposure": ["Types Of Hazard/Exposure", "Exposure", "Acute Hazards/Symptoms"],
    "symptom": ["Acute Hazards/Symptoms"],
    "long-term": ["Effects Of Long-Term Or Repeated Exposure"],
    "chronic": ["Effects Of Long-Term Or Repeated Exposure"],
}


def _matched_sections(query: str) -> set[str]:
    q = query.lower()
    hits = set()
    for keyword, sections in SECTION_KEYWORDS.items():
        if keyword in q:
            hits.update(sections)
    return hits


def _score_boost(doc, query_lower: str, target_sections: set[str]) -> float:
    boost = 0.0
    chemical = (doc.metadata.get("chemical_name") or "").lower()
    source_file = (doc.metadata.get("source_file") or "").lower()

    # Generic multi-chemical reference guides (NIOSH, WHO compendium) cover
    # hundreds of substances, so their chunks must NOT compete on equal
    # footing with a chunk from a chemical-specific ICSC/MSDS card when the
    # user named a specific chemical. Detect them by filename.
    is_generic_reference = "niosh" in source_file or "who_compendium" in source_file

    if chemical and chemical in query_lower:
        boost += 0.55  # strong boost: user named this exact chemical
    elif chemical and any(word in query_lower for word in chemical.split() if len(word) > 3):
        boost += 0.20  # partial name match

    if target_sections and doc.metadata.get("section") in target_sections:
        boost += 0.08
        if is_generic_reference:
            # Section match alone shouldn't let a generic guide outrank a
            # chemical-specific card — cut its section boost further.
            boost -= 0.05

    return boost


def retrieve_context(
    query: str, k: int = TOP_K, fetch_k: int = FETCH_K, threshold: float = RELEVANCE_THRESHOLD
) -> list[dict]:
    """Hybrid top-k retrieval. Returns a list of dicts with text + metadata + score.

    Chunks scoring below `threshold` are rejected outright rather than being
    returned just because they made the Top-K cut. If nothing clears the
    threshold, this returns an empty list — callers must treat that as
    "no relevant context found", not as "return whatever we have."
    """
    store = load_vector_store()

    # similarity_search_with_relevance_scores gives higher = more similar (0-1 ish)
    raw_results = store.similarity_search_with_relevance_scores(query, k=fetch_k)

    query_lower = query.lower()
    target_sections = _matched_sections(query)

    reranked = []
    for doc, score in raw_results:
        final_score = score + _score_boost(doc, query_lower, target_sections)
        reranked.append((doc, final_score))

    reranked.sort(key=lambda pair: pair[1], reverse=True)

    # Reject low-confidence chunks instead of always returning Top-K.
    relevant = [(doc, score) for doc, score in reranked if score >= threshold]
    top = relevant[:k]

    results = []
    for doc, score in top:
        results.append(
            {
                "text": doc.page_content,
                "score": round(float(score), 4),
                "chemical_name": doc.metadata.get("chemical_name"),
                "icsc_number": doc.metadata.get("icsc_number"),
                "section": doc.metadata.get("section"),
                "source_file": doc.metadata.get("source_file"),
                "page": doc.metadata.get("page"),
            }
        )
    return results


def format_context_for_prompt(results: list[dict]) -> str:
    """Turn retrieved chunks into a numbered context block with citation tags."""
    blocks = []
    for i, r in enumerate(results, start=1):
        citation = f"[{i}] {r['chemical_name']} (ICSC {r['icsc_number']}) - {r['section']}"
        blocks.append(f"{citation}\n{r['text']}")
    return "\n\n".join(blocks)
def retrieve_context_hybrid(
    query: str,
    uploaded_retriever=None,
    k: int = TOP_K,
    fetch_k: int = FETCH_K,
) -> list[dict]:
    """Same as retrieve_context(), but optionally also searches a temporary,
    session-scoped uploaded-PDF retriever (see 08_uploaded_pdf.py) and merges
    the results in.

    When a file is uploaded, slots are reserved for each source (roughly
    half from the uploaded document, half from the permanent Chroma
    knowledge base) so neither source can crowd the other out entirely.
    Any unused uploaded slots are backfilled from Chroma.

    If uploaded_retriever is None, this behaves EXACTLY like retrieve_context()
    — no change for users who haven't uploaded anything.
    """
    if uploaded_retriever is None:
        return retrieve_context(query, k=k, fetch_k=fetch_k)

    uploaded_slots = max(1, k // 2)
    chroma_slots = k - uploaded_slots

    try:
        uploaded_docs = uploaded_retriever.invoke(query)
    except Exception as exc:
        print(f"[06_retrieve_context] Uploaded retriever failed, ignoring it: {exc}")
        return retrieve_context(query, k=k, fetch_k=fetch_k)

    uploaded_results = []
    for doc in uploaded_docs[:uploaded_slots]:
        uploaded_results.append(
            {
                "text": doc.page_content,
                # Slightly favor the user's own uploaded document: it's the
                # one thing they explicitly asked to be searched.
                "score": 0.9,
                "chemical_name": doc.metadata.get("chemical_name"),
                "icsc_number": doc.metadata.get("icsc_number"),
                "section": doc.metadata.get("section", "Uploaded Document"),
                "source_file": doc.metadata.get("source_file"),
                "page": doc.metadata.get("page"),
            }
        )

    # Backfill: if the uploaded PDF didn't have enough relevant chunks,
    # let Chroma fill the remaining slots instead of returning fewer than k.
    remaining_slots = k - len(uploaded_results)
    chroma_results = retrieve_context(query, k=remaining_slots, fetch_k=fetch_k)

    merged = uploaded_results + chroma_results
    return merged[:k]

if __name__ == "__main__":
    test_queries = [
        "How should a concentrated sulfuric acid spill be handled in the lab?",
        "What PPE is required when handling arsenic powder?",
    ]
    for q in test_queries:
        print(f"\n=== Query: {q} ===")
        results = retrieve_context(q)
        for r in results:
            print(f"  score={r['score']:.3f}  {r['chemical_name']} / {r['section']}")
