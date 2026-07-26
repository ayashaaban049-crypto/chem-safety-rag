"""
streamlit_app.py
-------------------
Chemical Safety & MSDS Assistant | مستشار السلامة الكيميائية وتصنيف المخاطر

Streamlit front-end for the RAG pipeline built across 01_documents.py ->
07_prompting.py. On first load it builds the Chroma vector store from the
PDFs in data/ (cached across reruns), then serves a chat interface backed
by BGE embeddings + Groq Llama 3 70B, with quick-access buttons for the
most common lab emergencies and a "Sources" panel under every answer.
"""

import importlib
from pathlib import Path

import streamlit as st

docs_mod = importlib.import_module("01_documents")
prep_mod = importlib.import_module("02_preprocessing")
chunk_mod = importlib.import_module("03_chunking")
store_mod = importlib.import_module("05_create_chroma_store")
prompting = importlib.import_module("07_prompting")

st.set_page_config(
    page_title="Chemical Safety & MSDS Assistant",
    page_icon="🧪",
    layout="wide",
)

CHROMA_DIR = Path(__file__).parent / "chroma_db"


def get_available_chemicals() -> list[str]:
    """List unique chemical names parsed from the PDF filenames in data/."""
    data_dir = Path(__file__).parent / "data"
    names = set()
    for pdf_path in data_dir.glob("*.pdf"):
        meta = docs_mod.parse_filename_metadata(pdf_path.name)
        names.add(meta["chemical_name"])
    return sorted(names)

QUICK_QUERIES = [
    "How should a concentrated sulfuric acid (H2SO4) spill be handled in the lab?",
    "What PPE is required when handling acrylamide or arsenic powder?",
    "What are the first-aid steps if a chemical gets into the eyes or skin?",
    "How should arsenic or volatile solvents be safely stored and disposed of?",
]

# ---------------------------------------------------------------------------
# API key resolution: env var first, then Streamlit secrets (never hard-coded)
# ---------------------------------------------------------------------------
try:
    if not prompting.GROQ_API_KEY:
        prompting.GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
        prompting.GROQ_MODEL = st.secrets.get("GROQ_MODEL", prompting.GROQ_MODEL)
except Exception:
    pass


@st.cache_resource(show_spinner=False)
def ensure_vector_store():
    """Build the Chroma store once (from data/) and cache it for the session.
    If chroma_db/ was shipped pre-built in the repo, this just loads it."""
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        return store_mod.load_vector_store()

    with st.spinner("Building knowledge base from ICSC safety cards (first run only)..."):
        pages = docs_mod.load_documents()
        clean_pages = prep_mod.preprocess_documents(pages)
        chunks = chunk_mod.chunk_documents(clean_pages)
        documents = store_mod.chunks_to_documents(chunks)
        store_mod.build_vector_store(documents)
    return store_mod.load_vector_store()


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})", expanded=False):
        for i, s in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {s['chemical_name']}** (ICSC {s['icsc_number']}) "
                f"— *{s['section']}*  \n"
                f"`{s['source_file']}` · relevance {s['score']:.2f}"
            )


def ask(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not prompting.GROQ_API_KEY:
            error_msg = (
                "⚠️ No Groq API key configured. Add `GROQ_API_KEY` under "
                "**Manage app → Secrets** (Streamlit Cloud) or set it as an "
                "environment variable locally."
            )
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            return

        with st.spinner("Retrieving safety context and generating answer..."):
            try:
                result = prompting.generate_answer(question, k=5)
            except Exception as exc:
                result = {"answer": f"⚠️ Error generating answer: {exc}", "sources": []}

        st.markdown(result["answer"])
        render_sources(result["sources"])
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧪 Chemical Safety & MSDS Assistant")
    st.caption("👩‍💻 Developed by: Aya Shaaban & Iman Moustafa")
    st.divider()

    st.markdown("#### ⚙️ System")

    if st.button(
        "🔄 Rebuild Knowledge Base",
        use_container_width=True,
    ):
        st.cache_resource.clear()
        st.rerun()

    st.divider()

    st.markdown("### 🧪 Available Chemicals")

    sidebar_clicked = None

    with st.expander(
        f"📚 Browse Chemicals ({len(get_available_chemicals())})",
        expanded=False,
    ):
        for chem in get_available_chemicals():
            if st.button(
                chem,
                key=f"chem_{chem}",
                use_container_width=True,
            ):
                sidebar_clicked = chem

    st.divider()

    st.caption(
        "⚠️ This assistant answers strictly from the loaded ICSC/MSDS cards. "
        "It is a decision-support tool, not a replacement for your lab's "
        "official safety procedures or emergency services."
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🧪 Chemical Safety & MSDS Assistant")
st.caption("Ask about spills, PPE, first aid, storage, and waste disposal for lab chemicals.")

ensure_vector_store()

st.markdown("**Quick emergency queries:**")
cols = st.columns(len(QUICK_QUERIES))
quick_clicked = None
for col, q in zip(cols, QUICK_QUERIES):
    label = q if len(q) < 45 else q[:42] + "..."
    if col.button(label, help=q, use_container_width=True):
        quick_clicked = q

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

typed_question = st.chat_input("Ask about a chemical spill, PPE, first aid, storage...")

if quick_clicked:
    ask(quick_clicked)
elif sidebar_clicked:
    ask(
        f"What are the safety precautions, PPE, first-aid steps, storage "
        f"requirements, and spill handling instructions for {sidebar_clicked}?"
    )
elif typed_question:
    ask(typed_question)
