"""
01_documents.py
----------------
Stage 1 of the RAG pipeline: DOCUMENT INGESTION.

Loads all International Chemical Safety Cards (ICSC PDFs) and any other
MSDS/reference documents placed in the `data/` folder, extracts raw text
per page, and attaches lightweight metadata (chemical name, ICSC number,
source filename) parsed straight from each filename.

This file has no dependency on the rest of the pipeline so it can be run
and inspected on its own:

    python 01_documents.py
"""

import os
import re
import json
from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data"
RAW_DOCS_PATH = Path(__file__).parent / "artifacts" / "01_raw_documents.json"

ICSC_FILENAME_PATTERN = re.compile(
    r"ICSC\s*(?P<number>\d+)\s*-\s*(?P<name>.+)\.pdf", re.IGNORECASE
)


def parse_filename_metadata(filename: str) -> dict:
    """Pull a clean chemical name + ICSC card number out of the filename."""
    match = ICSC_FILENAME_PATTERN.search(filename)
    if match:
        return {
            "icsc_number": match.group("number").strip(),
            "chemical_name": match.group("name").strip().title(),
        }
    return {
        "icsc_number": None,
        "chemical_name": Path(filename).stem.title(),
    }


def load_pdf(path: Path) -> list[dict]:
    """Extract text page-by-page from a single PDF."""
    reader = PdfReader(str(path))
    meta = parse_filename_metadata(path.name)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "source_file": path.name,
                    "chemical_name": meta["chemical_name"],
                    "icsc_number": meta["icsc_number"],
                    "page": page_num,
                    "text": text,
                }
            )
    return pages


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load every PDF (and .txt, if present) found in data_dir."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. Place your ICSC/MSDS "
            f"PDFs there before running the pipeline."
        )

    all_pages = []
    pdf_files = sorted(data_dir.glob("*.pdf"))
    txt_files = sorted(data_dir.glob("*.txt"))

    for pdf_path in pdf_files:
        try:
            all_pages.extend(load_pdf(pdf_path))
        except Exception as exc:  # keep ingestion resilient to one bad file
            print(f"[01_documents] WARNING: failed to read {pdf_path.name}: {exc}")

    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        meta = parse_filename_metadata(txt_path.name)
        all_pages.append(
            {
                "source_file": txt_path.name,
                "chemical_name": meta["chemical_name"],
                "icsc_number": meta["icsc_number"],
                "page": 1,
                "text": text,
            }
        )

    print(f"[01_documents] Loaded {len(pdf_files)} PDFs + {len(txt_files)} TXT "
          f"files -> {len(all_pages)} pages total.")
    return all_pages


def save_documents(pages: list[dict], out_path: Path = RAW_DOCS_PATH) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"[01_documents] Saved raw documents -> {out_path}")


if __name__ == "__main__":
    docs = load_documents()
    save_documents(docs)
