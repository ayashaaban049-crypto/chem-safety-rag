"""
08_uploaded_pdf.py
---------------------
Optional stage: SESSION-SCOPED UPLOADED DOCUMENT.

Builds a temporary, in-memory FAISS retriever from a single PDF that the
user uploads through the Streamlit UI (see streamlit_app.py). This is
completely separate from the permanent Chroma knowledge base:

  - Nothing here is written to disk.
  - Nothing here is added to the persistent Chroma store.
  - The FAISS index lives only in memory for the current Streamlit session
    and is discarded automatically when the session ends or a new file is
    uploaded.

Reuses the SAME embedding model as the rest of the pipeline
(04_vector_representation.get_embedding_model()) so no second model is ever
loaded into memory.
"""

import importlib
import io

from langchain_community.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

from pypdf import PdfReader

vector_rep = importlib.import_module("04_vector_representation")
get_embedding_model = vector_rep.get_embedding_model

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def extract_text_from_uploaded_pdf(file_bytes: bytes) -> list[dict]:
    """Extract text page-by-page from an in-memory PDF (no disk write)."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": page_num, "text": text})
    return pages


def build_uploaded_retriever(file_bytes: bytes, file_name: str, k: int = 5):
    """Build a temporary, in-memory FAISS retriever from an uploaded PDF.

    Returns a LangChain retriever object (via FAISS.as_retriever), or None
    if the PDF contained no extractable text.
    """
    pages = extract_text_from_uploaded_pdf(file_bytes)
    if not pages:
        return None

    documents = []
    for page in pages:
        chunks = _splitter.split_text(page["text"])
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 15:
                continue
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "chemical_name": file_name,
                        "icsc_number": None,
                        "section": "Uploaded Document",
                        "source_file": file_name,
                        "page": page["page"],
                    },
                )
            )

    if not documents:
        return None

    embedding_model = get_embedding_model()  # reuse the shared model, never a new one
    faiss_store = FAISS.from_documents(documents, embedding_model)
    print(f"[08_uploaded_pdf] Built in-memory FAISS index: {len(documents)} chunks from '{file_name}'")
    return faiss_store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    # Smoke test: build a retriever from a local PDF and run a sample query.
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 08_uploaded_pdf.py <path_to_pdf>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        raw_bytes = f.read()

    retriever = build_uploaded_retriever(raw_bytes, file_name=sys.argv[1])
    if retriever is None:
        print("No extractable text found in the PDF.")
    else:
        results = retriever.invoke("What are the safety precautions mentioned?")
        for doc in results:
            print("---")
            print(doc.page_content[:200])
