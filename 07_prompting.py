"""
07_prompting.py
------------------
Stage 7 of the RAG pipeline: PROMPTING / GENERATION.

Wires the retriever from 06_retrieve_context.py to Groq's hosted
Llama 3 70B model (llama3-70b-8192) with a strict, grounded system
prompt: the assistant must answer ONLY from retrieved ICSC/MSDS
context, must say so explicitly when the context doesn't cover the
question, and must cite the chemical + ICSC section for every claim.

API key handling (per project rules):
  - GROQ_API_KEY is never hard-coded here.
  - It is read from the environment first, then (when running inside
    Streamlit) from st.secrets — see streamlit_app.py for the exact
    fallback logic.

Run standalone (requires GROQ_API_KEY to be set in your environment):
    export GROQ_API_KEY=your_key_here
    python 07_prompting.py
"""

import os
import importlib

from groq import Groq

retrieve_module = importlib.import_module("06_retrieve_context")
retrieve_context = retrieve_module.retrieve_context
retrieve_context_hybrid = retrieve_module.retrieve_context_hybrid
format_context_for_prompt = retrieve_module.format_context_for_prompt

# ---------------------------------------------------------------------------
# Config — never put a real key here. Left blank; filled at runtime from
# the environment or Streamlit secrets.
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a Chemical Safety & MSDS Assistant (مستشار السلامة الكيميائية) \
for a chemistry/biochemistry lab. Your job is to give fast, accurate emergency and \
safety guidance (spills, first aid, PPE, storage, waste disposal, firefighting) for \
hazardous chemicals.

STRICT RULES — follow these exactly:
1. Answer ONLY using the "RETRIEVED CONTEXT" provided below. Do not use outside \
   knowledge, do not guess, and do not fill gaps with general chemistry knowledge.
2. If the retrieved context does not contain enough information to answer safely, \
   your response MUST start with the exact tag [NO_MATCH] on its own, followed by \
   a brief explanation that the available safety cards do not cover this, and a \
   recommendation to consult the physical MSDS/ICSC or a safety officer. Never \
   invent a safety procedure. Use [NO_MATCH] ONLY when you truly cannot answer — \
   not when you have partial but usable information.
3. Every factual claim must be followed by a citation marker like [1], [2] that \
   matches the numbered context blocks you were given.
4. Prioritize immediate, actionable safety steps first (what to do right now), then \
   supporting detail.
5. If the user's question involves a life-threatening emergency (e.g. major exposure, \
   fire, large spill), begin the answer with a short "IMMEDIATE ACTION" line before \
   any other detail.
6. Keep answers concise, structured (use short bullet points), and written for \
   someone standing in a lab reading this quickly, not writing an essay.
7. Never contradict the retrieved context. If two retrieved chunks conflict, say so.

You may answer in English or Arabic depending on the language the user asked in.
"""

USER_PROMPT_TEMPLATE = """RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

Answer the question using only the retrieved context above, with citation markers \
like [1], [2] referencing the numbered blocks. End your answer with a "Sources" line \
listing which chemicals/sections you cited.
"""


def get_groq_client(api_key: str = None) -> Groq:
    key = api_key or GROQ_API_KEY
    if not key:
        raise RuntimeError(
            "No GROQ_API_KEY found. Set it as an environment variable or, when "
            "deployed on Streamlit Cloud, add it under Manage app -> Secrets."
        )
    return Groq(api_key=key)


def _translate_to_english(question: str, api_key: str = None, model: str = None) -> str:
    """Translate the question to English for retrieval only. If it's already
    English, the model just returns it unchanged. Falls back to the original
    question if translation fails for any reason."""
    try:
        client = get_groq_client(api_key)
        response = client.chat.completions.create(
            model=model or GROQ_MODEL,
            temperature=0,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the user's message to English. If it is "
                        "already in English, return it unchanged. Reply with "
                        "ONLY the translated text, nothing else."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return question  # if translation fails, just retrieve with the original


NO_MATCH_TAG = "[NO_MATCH]"


def _strip_no_match_tag(answer_text: str) -> tuple[str, bool]:
    """Detect the model's explicit [NO_MATCH] marker and strip it from the
    displayed text. Returns (cleaned_text, is_fallback)."""
    stripped = answer_text.strip()
    if stripped.startswith(NO_MATCH_TAG):
        cleaned = stripped[len(NO_MATCH_TAG):].strip()
        return cleaned, True
    return answer_text, False


def generate_answer(
    question: str,
    k: int = 5,
    api_key: str = None,
    model: str = None,
    temperature: float = 0.1,
    uploaded_retriever=None,
) -> dict:
    """Full RAG call: translate -> retrieve context -> build grounded prompt -> call Groq.

    If uploaded_retriever is provided (a temporary, session-scoped FAISS
    retriever built from a user-uploaded PDF — see 08_uploaded_pdf.py), its
    results are merged in with the permanent Chroma knowledge base for this
    call only. If uploaded_retriever is None (the default), behavior is
    unchanged from before.

    Sources are returned ONLY when they were actually used to produce a real
    answer. If retrieval finds nothing above the relevance threshold, or the
    model itself signals [NO_MATCH] (insufficient context), sources is always
    an empty list — never populated just because chunks were in the Top-K.
    """
    search_query = _translate_to_english(question, api_key, model)
    results = retrieve_context_hybrid(search_query, uploaded_retriever=uploaded_retriever, k=k)

    if not results:
        return {
            "answer": (
                "I couldn't find any relevant chemical safety information for this "
                "question in the current knowledge base. Please consult the physical "
                "MSDS/ICSC card or your lab safety officer."
            ),
            "sources": [],
        }

    context_block = format_context_for_prompt(results)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context_block, question=question)

    client = get_groq_client(api_key)
    response = client.chat.completions.create(
        model=model or GROQ_MODEL,
        temperature=temperature,
        max_tokens=900,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_answer = response.choices[0].message.content
    answer_text, is_fallback = _strip_no_match_tag(raw_answer)

    sources = [] if is_fallback else [
        {
            "chemical_name": r["chemical_name"],
            "icsc_number": r["icsc_number"],
            "section": r["section"],
            "source_file": r["source_file"],
            "score": r["score"],
        }
        for r in results
    ]

    return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    q = "How should a concentrated sulfuric acid spill be handled in the lab?"
    result = generate_answer(q)
    print("ANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(" -", s)
