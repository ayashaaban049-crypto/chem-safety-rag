"""
02_preprocessing.py
--------------------
Stage 2 of the RAG pipeline: PREPROCESSING / CLEANING.

ICSC PDF text extraction tends to produce:
  - repeated header/footer boilerplate ("International Chemical Safety Cards")
  - broken hyphenation across line breaks
  - stray page numbers
  - inconsistent whitespace

This module normalizes that noise while preserving the ICSC section
headings (TYPES OF HAZARD/EXPOSURE, PREVENTION, FIRST AID, FIRE, SPILLAGE,
STORAGE, PACKAGING & LABELLING) since 03_chunking.py relies on them.

Run standalone:
    python 02_preprocessing.py
"""

import json
import re
from pathlib import Path

RAW_DOCS_PATH = Path(__file__).parent / "artifacts" / "01_raw_documents.json"
CLEAN_DOCS_PATH = Path(__file__).parent / "artifacts" / "02_clean_documents.json"

BOILERPLATE_PATTERNS = [
    r"International Chemical Safety Cards?",
    r"Prepared in the context of cooperation.*?(?=\n|$)",
    r"©\s*I?LO.*?(?=\n|$)",
    r"ICSC\s*:?\s*\d{3,4}",
    r"Page\s*\d+\s*(of\s*\d+)?",
]

BOILERPLATE_RE = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    text = HYPHEN_BREAK_RE.sub(r"\1\2", text)          # rejoin hyphenated line breaks
    text = BOILERPLATE_RE.sub(" ", text)                 # strip repeated boilerplate
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def preprocess_documents(pages: list[dict]) -> list[dict]:
    cleaned = []
    for page in pages:
        text = clean_text(page["text"])
        if len(text) < 20:  # drop near-empty pages (blank cover pages etc.)
            continue
        new_page = dict(page)
        new_page["text"] = text
        cleaned.append(new_page)
    print(f"[02_preprocessing] Cleaned {len(cleaned)}/{len(pages)} pages "
          f"(dropped near-empty pages).")
    return cleaned


def load_raw_documents(path: Path = RAW_DOCS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_clean_documents(pages: list[dict], out_path: Path = CLEAN_DOCS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"[02_preprocessing] Saved clean documents -> {out_path}")


if __name__ == "__main__":
    raw = load_raw_documents()
    clean = preprocess_documents(raw)
    save_clean_documents(clean)
