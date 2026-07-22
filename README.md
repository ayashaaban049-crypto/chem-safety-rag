# Chemical Safety & MSDS Assistant
### مستشار السلامة الكيميائية وتصنيف المخاطر

A lightweight, production-ready RAG assistant for chemistry/biochemistry lab
safety: spill protocols, PPE, first aid, storage, and waste disposal —
grounded strictly in International Chemical Safety Cards (ICSC).

---

## 1. Architecture & Pipeline

```
data/*.pdf (35 ICSC cards)
      │
      ▼
01_documents.py           extract raw text per PDF page + parse chemical
                           name / ICSC number from filename
      │
      ▼
02_preprocessing.py       strip header/footer boilerplate, rejoin broken
                           hyphenation, normalize whitespace
      │
      ▼
03_chunking.py            SEMANTIC chunking: split on known ICSC section
                           headers (Prevention, First Aid, Fire, Spillage
                           Disposal, Storage, Packaging & Labelling, ...)
                           so a "spill" question retrieves the *whole*
                           spillage section, not a random slice of it.
                           Oversized sections get a secondary recursive
                           character split (800 chars, 120 overlap).
      │
      ▼
04_vector_representation.py   BAAI/bge-small-en-v1.5 (Hugging Face) via
                               LangChain's HuggingFaceBgeEmbeddings, with
                               the BGE query-instruction prefix applied
                               only to queries (per BGE's training recipe).
      │
      ▼
05_create_chroma_store.py     Embeds every chunk and persists it to a
                               local ChromaDB collection on disk
                               (./chroma_db).
      │
      ▼
06_retrieve_context.py        Hybrid top-k retrieval:
                                 - dense similarity search (BGE + Chroma)
                                 - + boost if the query names a chemical
                                   that matches a chunk's metadata
                                 - + boost if the query's intent (spill /
                                   PPE / first aid / storage / disposal)
                                   matches the chunk's ICSC section tag
      │
      ▼
07_prompting.py               Groq-hosted Llama 3 70B (llama3-70b-8192).
                               Strict system prompt: answer ONLY from
                               retrieved context, cite every claim with
                               [1]/[2] markers, explicitly say when the
                               knowledge base doesn't cover the question,
                               lead with "IMMEDIATE ACTION" for
                               emergencies.
      │
      ▼
streamlit_app.py              Chat UI + 4 one-click emergency queries +
                               expandable "Sources" panel under every
                               answer (chemical, ICSC #, section, file).
```

**Why semantic (section-aware) chunking matters here:** ICSC cards are
highly structured. A naive fixed-size split can cut a spill procedure in
half or merge "First Aid" with "Storage." Splitting on the cards' own
section headers first — and only falling back to size-based splitting
for oversized sections — keeps each retrieved chunk topically complete
and directly attributable to a section, which is what makes the citation
mechanism ("First Aid" for Sulfuric Acid [1]) reliable.

## 2. Project Structure

```
chem_safety_rag/
├── 01_documents.py
├── 02_preprocessing.py
├── 03_chunking.py
├── 04_vector_representation.py
├── 05_create_chroma_store.py
├── 06_retrieve_context.py
├── 07_prompting.py
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .streamlit/secrets.toml.example
├── .gitignore
└── data/                  # 35 ICSC safety cards (PDF)
```

`artifacts/` (intermediate JSON from stages 1-3) and `chroma_db/` (the
persisted vector store) are generated at run time and git-ignored — they
are not required in the repo, though you may commit a pre-built
`chroma_db/` to skip the first-run embedding step in production.

## 3. Setup

```bash
pip install -r requirements.txt

# Local run: put your Groq key in .env (copy from .env.example)
cp .env.example .env
# edit .env -> GROQ_API_KEY=...

# Run the full pipeline once to build the vector store (optional — the
# Streamlit app also builds it automatically on first launch)
python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 05_create_chroma_store.py   # imports 04 internally

streamlit run streamlit_app.py
```

## 4. API Key Handling (per submission rules)

- No real key is written in any `.py` file — `GROQ_API_KEY` is read from
  `os.environ` and, when running under Streamlit, from `st.secrets` as a
  fallback (see `streamlit_app.py` / `07_prompting.py`).
- `.env` and `.streamlit/secrets.toml` are both git-ignored.
- **Streamlit Cloud deployment:** open your app → **Manage app** →
  **Secrets**, and paste:

  ```toml
  GROQ_API_KEY = "your_groq_key_here"
  GROQ_MODEL = "llama3-70b-8192"
  ```

## 5. Sample Output (expected response shape)

> **Q: How should a concentrated sulfuric acid (H₂SO₄) spill be handled in the lab?**
>
> **IMMEDIATE ACTION:** Evacuate non-essential personnel, ventilate the area, and do not let the spill enter drains [1].
> - Wear acid-resistant gloves, splash goggles, and a face shield before approaching the spill [2].
> - Absorb with dry sand or inert absorbent — do NOT use combustible materials or attempt to neutralize with water directly onto concentrated acid [1].
> - Collect absorbed material in a suitable, labeled closed container for disposal as hazardous waste [1].
>
> **Sources:** [1] Sulfuric Acid — Spillage Disposal (ICSC 0362); [2] Sulfuric Acid — Prevention (ICSC 0362)

> **Q: What PPE is required when handling arsenic powder?**
>
> - Use a respirator with a particulate filter matched to arsenic dust concentration; local exhaust ventilation is preferred [1].
> - Wear protective gloves and protective clothing to prevent skin contact [1].
> - Wash hands before eating, drinking, or smoking; do not eat, drink, or smoke while handling [2].
>
> **Sources:** [1] Arsenic — Prevention (ICSC 0013); [2] Arsenic — Prevention (ICSC 0013)

*(Exact wording depends on live Groq generation; the assistant will
refuse to answer, or flag a gap, if the retrieved ICSC context doesn't
actually cover a sub-question — e.g. it will not invent an antidote that
isn't in the card.)*

## 6. Notes on This Build

- **Knowledge base actually used:** the 35 ICSC cards provided in
  `Data.zip` (Arsenic, Sulfuric Acid, Acetic Acid, Ethyl Acetate,
  Ammonia, Benzyl Alcohol, Titanium Dioxide (incl. nanoform), Benzene,
  Cadmium, and 26 others). No WHO Compendium PDF was included in the
  uploaded data, so it is not part of this build — drop it into `data/`
  and rerun the pipeline to add it.
- **Model choices are swappable:** `EMBEDDING_MODEL_NAME` in
  `04_vector_representation.py` and `GROQ_MODEL` in `07_prompting.py`
  are both env-var overridable without touching code.
- This environment could not reach `huggingface.co` or the Groq API
  directly (sandboxed network), so stages 1–3 were executed end-to-end
  against the real ICSC PDFs (82 pages → chunks verified), while the
  embedding/LLM stages were verified for syntax and import-compatibility
  only. Run `streamlit run streamlit_app.py` locally or on Streamlit
  Cloud, where those hosts are reachable, for the live first run.
