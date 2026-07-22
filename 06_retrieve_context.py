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
    if chemical and chemical in query_lower:
        boost += 0.25  # strong boost: user named this exact chemical
    elif chemical and any(word in query_lower for word in chemical.split()):
        boost += 0.10  # partial name match

    if target_sections and doc.metadata.get("section") in target_sections:
        boost += 0.15
    return boost


def retrieve_context(query: str, k: int = TOP_K, fetch_k: int = FETCH_K) -> list[dict]:
    """Hybrid top-k retrieval. Returns a list of dicts with text + metadata + score."""
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
    top = reranked[:k]

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
