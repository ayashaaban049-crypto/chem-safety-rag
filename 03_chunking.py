"""
03_chunking.py
----------------
Stage 3 of the RAG pipeline: SEMANTIC CHUNKING.

ICSC cards follow a very consistent structural template:

    TYPES OF HAZARD/EXPOSURE | ACUTE HAZARDS/SYMPTOMS | PREVENTION | FIRST AID
    FIRE
    EXPLOSION
    EXPOSURE
      Inhalation / Skin / Eyes / Ingestion
    SPILLAGE DISPOSAL
    PACKAGING & LABELLING
    EMERGENCY RESPONSE
    STORAGE

Rather than blindly splitting by character count, we first split on these
known section headers so a "Spillage" question retrieves the *whole*
spillage section for a chemical, not an arbitrary half of it. Any section
that is still too large for the embedding model gets a secondary
recursive character split with overlap.

Run standalone:
    python 03_chunking.py
"""

import json
import re
from pathlib import Path

try:
    # Modern langchain (0.2+) ships the splitter in this standalone package.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Older langchain versions expose it directly under langchain.text_splitter.
    from langchain.text_splitter import RecursiveCharacterTextSplitter

CLEAN_DOCS_PATH = Path(__file__).parent / "artifacts" / "02_clean_documents.json"
CHUNKS_PATH = Path(__file__).parent / "artifacts" / "03_chunks.json"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Section headers that ICSC cards consistently use (order doesn't matter here,
# we just need to detect boundaries).
SECTION_HEADERS = [
    "TYPES OF HAZARD/EXPOSURE",
    "ACUTE HAZARDS/SYMPTOMS",
    "PREVENTION",
    "FIRST AID",
    "FIRE FIGHTING",
    "FIREFIGHTING",
    "FIRE",
    "EXPLOSION",
    "EXPOSURE",
    "INHALATION",
    "SKIN",
    "EYES",
    "INGESTION",
    "SPILLAGE DISPOSAL",
    "SPILLAGE",
    "PACKAGING & LABELLING",
    "PACKAGING AND LABELLING",
    "EMERGENCY RESPONSE",
    "STORAGE",
    "PHYSICAL STATE",
    "PHYSICAL AND CHEMICAL PROPERTIES",
    "ROUTES OF EXPOSURE",
    "EFFECTS OF SHORT-TERM EXPOSURE",
    "EFFECTS OF LONG-TERM OR REPEATED EXPOSURE",
    "OCCUPATIONAL EXPOSURE LIMITS",
    "ENVIRONMENTAL DATA",
    "NOTES",
]

_SECTION_SPLIT_RE = re.compile(
    r"(?=^(?:" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\b)",
    re.MULTILINE | re.IGNORECASE,
)

_fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def split_into_sections(text: str) -> list[str]:
    """Split page text on known ICSC section headers; fall back to the whole
    text as one section if no headers are detected."""
    parts = [p.strip() for p in _SECTION_SPLIT_RE.split(text) if p.strip()]
    return parts if parts else [text]


def detect_section_label(section_text: str) -> str:
    first_line = section_text.strip().split("\n", 1)[0][:60]
    for header in SECTION_HEADERS:
        if first_line.upper().startswith(header):
            return header.title()
    return "General"


def chunk_documents(pages: list[dict]) -> list[dict]:
    chunks = []
    chunk_id = 0
    for page in pages:
        sections = split_into_sections(page["text"])
        for section in sections:
            label = detect_section_label(section)
            if len(section) <= CHUNK_SIZE:
                sub_chunks = [section]
            else:
                sub_chunks = _fallback_splitter.split_text(section)

            for sub in sub_chunks:
                sub = sub.strip()
                if len(sub) < 15:
                    continue
                chunk_id += 1
                chunks.append(
                    {
                        "chunk_id": f"chunk_{chunk_id:05d}",
                        "text": sub,
                        "metadata": {
                            "source_file": page["source_file"],
                            "chemical_name": page["chemical_name"],
                            "icsc_number": page["icsc_number"],
                            "page": page["page"],
                            "section": label,
                        },
                    }
                )
    print(f"[03_chunking] Produced {len(chunks)} chunks from {len(pages)} pages.")
    return chunks


def load_clean_documents(path: Path = CLEAN_DOCS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_chunks(chunks: list[dict], out_path: Path = CHUNKS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[03_chunking] Saved chunks -> {out_path}")


if __name__ == "__main__":
    pages = load_clean_documents()
    chunks = chunk_documents(pages)
    save_chunks(chunks)
