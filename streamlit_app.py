"""
streamlit_app.py
-------------------
Chemical Safety & MSDS Assistant | Developed by Aya Shaaban & Iman Moustafa

Streamlit front-end for the RAG pipeline built across 01_documents.py ->
08_uploaded_pdf.py. All backend logic (Chroma retrieval, session-scoped
uploaded-PDF FAISS retrieval, Groq generation) is unchanged — this file only
adds visual styling, layout, and a Dark/Light theme switcher.

Styling note: custom CSS is deliberately kept minimal and only targets
elements we render ourselves (header banner, dev badge, prompt buttons) plus
the top-level app/sidebar/header background so there are no white gaps.
Native Streamlit components (file_uploader, radio, text inputs) are left
untouched so their built-in rendering and contrast stay intact.
"""

import base64
import importlib
from pathlib import Path

import streamlit as st

docs_mod = importlib.import_module("01_documents")
prep_mod = importlib.import_module("02_preprocessing")
chunk_mod = importlib.import_module("03_chunking")
store_mod = importlib.import_module("05_create_chroma_store")
prompting = importlib.import_module("07_prompting")
uploaded_pdf_mod = importlib.import_module("08_uploaded_pdf")

st.set_page_config(
    page_title="Chemical Safety & MSDS Assistant",
    page_icon="🧪",
    layout="wide",
)

CHROMA_DIR = Path(__file__).parent / "chroma_db"
ASSETS_DIR = Path(__file__).parent / "assets"

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


# ---------------------------------------------------------------------------
# Theme state — defaults to Dark Mode on first load
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"


def inject_css(theme: str):
    """Minimal, targeted CSS. Only touches:
    - overall app / header / sidebar background (so there are no white gaps)
    - the custom header banner we render
    - the developer badge / avatar we render
    - button hover styling (used by the quick-query prompt buttons)

    Deliberately does NOT touch st.file_uploader, st.radio, text inputs, or
    any other native Streamlit component internals — those keep their
    built-in Streamlit styling and contrast in both themes.
    """
    if theme == "dark":
        bg_app = "#0e1117"
        bg_sidebar = "#12181c"
        text_main = "#ffffff"
        text_muted = "#b8ffd9"
        accent = "#00e676"
        accent_glow = "rgba(0, 230, 118, 0.55)"
        border_soft = "rgba(0, 230, 118, 0.35)"
    else:  # light
        bg_app = "#f8f9fa"
        bg_sidebar = "#ffffff"
        text_main = "#212529"
        text_muted = "#2e6b4f"
        accent = "#1b8a4f"
        accent_glow = "rgba(27, 138, 79, 0.35)"
        border_soft = "rgba(27, 138, 79, 0.30)"

    st.markdown(
        f"""
        <style>
        /* Consistent app-wide background so there are no white gaps between
           the header bar, main container, and sidebar. */
        .stApp,
        header[data-testid="stHeader"] {{
            background-color: {bg_app};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {bg_sidebar};
        }}

        /* Custom header banner */
        .lab-header {{
            background: linear-gradient(135deg, {bg_app} 0%, {bg_sidebar} 100%);
            border: 1px solid {accent};
            border-radius: 14px;
            padding: 28px 32px;
            margin-bottom: 24px;
            box-shadow: 0 0 24px {accent_glow};
        }}
        .lab-header h1 {{
            color: {text_main};
            font-size: 2rem;
            margin: 0 0 6px 0;
        }}
        .lab-header p {{
            color: {text_muted};
            font-size: 0.95rem;
            margin: 0;
        }}

        /* Developer badge + avatar */
        .avatar-wrap {{
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }}
        .avatar-wrap img {{
            width: 120px;
            height: 120px;
            object-fit: cover;
            border-radius: 50%;
            border: 3px solid {accent};
            box-shadow: 0 0 18px {accent_glow};
        }}
        .dev-badge {{
            text-align: center;
            color: {text_muted};
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 18px;
        }}

        /* Prompt / action buttons only — hover glow effect */
        div[data-testid="stButton"] > button {{
            border-radius: 10px !important;
            border: 1px solid {border_soft} !important;
            transition: all 0.25s ease-in-out;
        }}
        div[data-testid="stButton"] > button:hover {{
            border: 1px solid {accent} !important;
            box-shadow: 0 0 14px {accent_glow};
            color: {accent} !important;
        }}

        /* Answer callout boxes */
        .alert-box {{
            border-radius: 10px;
            padding: 16px 18px;
            margin: 10px 0;
            border-left: 5px solid;
        }}
        .alert-danger {{
            background: rgba(255, 87, 34, 0.10);
            border-left-color: #ff5722;
        }}
        .alert-safe {{
            background: rgba(0, 230, 118, 0.08);
            border-left-color: {accent};
        }}
        .alert-neutral {{
            background: rgba(128, 128, 128, 0.08);
            border-left-color: #808a8f;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <div class="lab-header">
            <h1>🧪 Chemical Safety & MSDS Assistant</h1>
            <p>Real-time emergency guidance for lab spills, first aid, PPE, and waste
            management — grounded strictly in ICSC / MSDS safety data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_avatar():
    image_path = ASSETS_DIR / "download.jpg"
    if image_path.exists():
        encoded = base64.b64encode(image_path.read_bytes()).decode()
        st.markdown(
            f"""
            <div class="avatar-wrap">
                <img src="data:image/jpeg;base64,{encoded}" />
            </div>
            <div class="dev-badge">👨‍🔬 Developed by: Aya Shaaban &amp; Iman Moustafa</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="dev-badge">👨‍🔬 Developed by: Aya Shaaban &amp; Iman Moustafa</div>',
            unsafe_allow_html=True,
        )


def get_alert_class(question: str, answer: str) -> str:
    text = f"{question} {answer}".lower()
    danger_words = ["spill", "first aid", "fire", "emergency", "exposure", "inhal", "burn"]
    safe_words = ["storage", "store", "ppe", "protective equipment", "dispos"]
    if any(w in text for w in danger_words):
        return "alert-danger"
    if any(w in text for w in safe_words):
        return "alert-safe"
    return "alert-neutral"


@st.cache_resource(show_spinner=False)
def ensure_vector_store():
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        return store_mod.load_vector_store()

    with st.spinner("Building knowledge base from safety documents (first run only)..."):
        pages = docs_mod.load_documents()
        clean_pages = prep_mod.preprocess_documents(pages)
        chunks = chunk_mod.chunk_documents(clean_pages)
        documents = store_mod.chunks_to_documents(chunks)
        store_mod.build_vector_store(documents)
    return store_mod.load_vector_store()


def get_available_chemicals() -> list[str]:
    data_dir = Path(__file__).parent / "data"
    names = set()
    for pdf_path in data_dir.glob("*.pdf"):
        meta = docs_mod.parse_filename_metadata(pdf_path.name)
        names.add(meta["chemical_name"])
    return sorted(names)


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
                result = prompting.generate_answer(
                    question,
                    k=5,
                    uploaded_retriever=st.session_state.get("uploaded_retriever"),
                )
            except Exception as exc:
                result = {"answer": f"⚠️ Error generating answer: {exc}", "sources": []}

        alert_class = get_alert_class(question, result["answer"])
        st.markdown(
            f'<div class="alert-box {alert_class}">{result["answer"]}</div>',
            unsafe_allow_html=True,
        )
        render_sources(result["sources"])
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )


# ---------------------------------------------------------------------------
# Sidebar (plain Streamlit containers — no custom card CSS, avoids the
# rendering conflicts from before)
# ---------------------------------------------------------------------------
with st.sidebar:
    render_sidebar_avatar()

    st.markdown("#### 🎨 Theme ")
    theme_choice = st.radio(
        "Theme",
        options=["🌙 Dark Mode", "☀️ Light Mode"],
        index=0 if st.session_state["theme"] == "dark" else 1,
        label_visibility="collapsed",
        key="theme_radio",
    )
    st.session_state["theme"] = "dark" if theme_choice == "🌙 Dark Mode" else "light"
    st.divider()

    st.markdown("#### ⚙️ System")
    if st.button("🔄 Rebuild knowledge base"):
        st.cache_resource.clear()
        st.rerun()
    st.divider()

    st.markdown("#### 📄 Upload a Document (this session only)")
    uploaded_pdf = st.file_uploader(
        "Upload one MSDS / SDS / SOP / safety guide PDF",
        type=["pdf"],
        accept_multiple_files=False,
        key="session_pdf_uploader",
        help=(
            "This file is used only for your current session. It is never "
            "saved to disk and never added to the permanent knowledge base."
        ),
    )

    if uploaded_pdf is not None:
        if st.session_state.get("uploaded_pdf_name") != uploaded_pdf.name:
            st.session_state["uploaded_pdf_bytes"] = uploaded_pdf.getvalue()
            st.session_state["uploaded_pdf_name"] = uploaded_pdf.name
            st.session_state.pop("uploaded_retriever", None)

            with st.spinner(f"Indexing {uploaded_pdf.name} for this session..."):
                try:
                    retriever = uploaded_pdf_mod.build_uploaded_retriever(
                        st.session_state["uploaded_pdf_bytes"],
                        file_name=uploaded_pdf.name,
                    )
                    st.session_state["uploaded_retriever"] = retriever
                except Exception as exc:
                    st.session_state["uploaded_retriever"] = None
                    st.error(f"⚠️ Could not process this PDF: {exc}")

        if st.session_state.get("uploaded_retriever") is not None:
            st.success(f"✅ Loaded for this session: {uploaded_pdf.name}")
        else:
            st.warning(f"⚠️ No searchable text found in: {uploaded_pdf.name}")
    else:
        st.session_state.pop("uploaded_pdf_bytes", None)
        st.session_state.pop("uploaded_pdf_name", None)
        st.session_state.pop("uploaded_retriever", None)
    st.divider()

    st.markdown("#### 🧪 Available Chemicals")
    sidebar_clicked = None
    with st.expander(f"Browse list ({len(get_available_chemicals())})", expanded=False):
        for chem in get_available_chemicals():
            if st.button(chem, key=f"chem_{chem}", use_container_width=True):
                sidebar_clicked = chem
    st.divider()

    st.caption(
        "⚠️ This assistant answers strictly from the loaded ICSC/MSDS cards. "
        "It is a decision-support tool, not a replacement for your lab's "
        "official safety procedures or emergency services."
    )

# ---------------------------------------------------------------------------
# Apply CSS for the selected theme (must run after the sidebar sets it)
# ---------------------------------------------------------------------------
inject_css(st.session_state["theme"])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
render_header()

ensure_vector_store()

st.markdown("**Quick emergency queries:**")
quick_clicked = None
row1 = st.columns(2)
row2 = st.columns(2)
for col, q in zip(row1 + row2, QUICK_QUERIES):
    if col.button(q, use_container_width=True, key=f"quick_{q}"):
        quick_clicked = q

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            alert_class = get_alert_class("", msg["content"])
            st.markdown(
                f'<div class="alert-box {alert_class}">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
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
