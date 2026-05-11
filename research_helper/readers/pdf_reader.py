from pathlib import Path
import re


def extract_text(pdf_path: Path) -> str:
    """Extract full text from a PDF using pymupdf, falling back to pdfplumber."""
    try:
        return _extract_pymupdf(pdf_path)
    except Exception:
        return _extract_pdfplumber(pdf_path)


def _extract_pymupdf(pdf_path: Path) -> str:
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    text = "\n".join(pages)
    return _clean(text)


def _extract_pdfplumber(pdf_path: Path) -> str:
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    return _clean("\n".join(pages))


def _clean(text: str) -> str:
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove form feeds
    text = text.replace("\f", "\n")
    return text.strip()
