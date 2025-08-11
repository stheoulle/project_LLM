"""PDF ingestion utilities

Extract text from PDFs and save as .txt files under the project's docs/ directory so the
local retriever can index them.

Usage:
    from utils.pdf_ingest import ingest_pdf_file, ingest_folder

    ingest_pdf_file('/path/to/DSM5.pdf', docs_dir='docs')
    ingest_folder('/path/to/pdfs/', docs_dir='docs')

Also usable from CLI:
    python -m utils.pdf_ingest --input /path/to/pdf_or_folder --docs docs/ --recursive

This module prefers `pdfplumber` for extraction and falls back to `PyPDF2` if needed.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from datetime import datetime

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False


def _extract_text_pdfplumber(path: str) -> str:
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        raise RuntimeError(f"pdfplumber extraction failed for {path}: {e}")
    return "\n\n".join(text_parts)


def _extract_text_pypdf2(path: str) -> str:
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            except Exception:
                continue
    except Exception as e:
        raise RuntimeError(f"PyPDF2 extraction failed for {path}: {e}")
    return "\n\n".join(text_parts)


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF using available libraries. Raises on complete failure."""
    if PDFPLUMBER_AVAILABLE:
        try:
            return _extract_text_pdfplumber(path)
        except Exception:
            # fallback
            pass

    if PYPDF2_AVAILABLE:
        return _extract_text_pypdf2(path)

    raise RuntimeError("No PDF extraction library available (install pdfplumber or PyPDF2).")


def _clean_text(text: str) -> str:
    # Simple whitespace normalization
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])


def ingest_pdf_file(pdf_path: str, docs_dir: str = "docs", overwrite: bool = False) -> str:
    """Extract text from a single PDF and save into docs_dir with same base name and .txt extension.

    Returns the path to the produced text file.
    """
    pdf_path = str(pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    os.makedirs(docs_dir, exist_ok=True)
    base = Path(pdf_path).stem
    out_path = os.path.join(docs_dir, base + ".txt")

    if os.path.exists(out_path) and not overwrite:
        return out_path

    text = extract_text_from_pdf(pdf_path)
    cleaned = _clean_text(text)

    header = f"Source: {pdf_path}\nExtracted: {datetime.utcnow().isoformat()}Z\n---\n\n"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(cleaned)
    except Exception as e:
        raise RuntimeError(f"Failed to write extracted text to {out_path}: {e}")

    return out_path


def ingest_folder(folder_path: str, docs_dir: str = "docs", recursive: bool = True, overwrite: bool = False) -> list:
    """Ingest all PDFs from a folder into docs_dir. Returns list of created text file paths."""
    pattern = "**/*.pdf" if recursive else "*.pdf"
    paths = glob.glob(os.path.join(folder_path, pattern), recursive=recursive)
    out_files = []
    for p in paths:
        try:
            out = ingest_pdf_file(p, docs_dir=docs_dir, overwrite=overwrite)
            out_files.append(out)
        except Exception as e:
            # Skip problematic files but continue
            print(f"Failed to ingest {p}: {e}", file=sys.stderr)
            continue
    return out_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingest PDFs into docs/ as text for local LLM retrieval.")
    parser.add_argument('--input', '-i', required=True, help='PDF file or folder to ingest')
    parser.add_argument('--docs', '-d', default='docs', help='Output docs directory')
    parser.add_argument('--recursive', action='store_true', help='Recursively find PDFs in folders')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing txt files')

    args = parser.parse_args()

    inp = args.input
    if os.path.isdir(inp):
        created = ingest_folder(inp, docs_dir=args.docs, recursive=args.recursive, overwrite=args.overwrite)
        print(f"Created {len(created)} docs under {args.docs}")
    elif os.path.isfile(inp):
        out = ingest_pdf_file(inp, docs_dir=args.docs, overwrite=args.overwrite)
        print(f"Created {out}")
    else:
        print(f"Input not found: {inp}")
