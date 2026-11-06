"""Turn an uploaded file into clean text pages. OCR runs only on PDF pages with no text layer."""

import csv
import io
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import PurePath

from apps.common.exceptions import InvalidDocument

logger = logging.getLogger(__name__)
CSV_ROWS_PER_PAGE = 50


@dataclass
class ParsedPage:
    page_number: int
    content: str


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # rejoin words hyphenated across a line break
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_page(page) -> str:
    if not shutil.which("tesseract"):
        logger.warning("Page %s has no text layer and tesseract is not installed; skipping it", page.number + 1)
        return ""
    import pytesseract
    from PIL import Image

    pixmap = page.get_pixmap(dpi=200)
    return pytesseract.image_to_string(Image.open(io.BytesIO(pixmap.tobytes("png"))))


def _parse_pdf(data: bytes) -> list[ParsedPage]:
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        pages = []
        for index, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip() or _ocr_page(page)
            pages.append(ParsedPage(index, clean_text(text)))
    return pages


def _parse_docx(data: bytes) -> list[ParsedPage]:
    from docx import Document as DocxDocument

    doc = DocxDocument(io.BytesIO(data))
    lines = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        # Keep headings as markdown so structure-based chunking can find them.
        lines.append(f"# {text}" if paragraph.style.name.startswith("Heading") else text)
    for table in doc.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return [ParsedPage(1, clean_text("\n\n".join(lines)))]


def _parse_csv(data: bytes) -> list[ParsedPage]:
    text = data.decode("utf-8-sig", errors="replace")
    rows = [f"{'; '.join(f'{k}: {v}' for k, v in row.items() if k)}" for row in csv.DictReader(io.StringIO(text))]
    return [
        ParsedPage(number, "\n".join(rows[start : start + CSV_ROWS_PER_PAGE]))
        for number, start in enumerate(range(0, len(rows), CSV_ROWS_PER_PAGE), start=1)
    ]


def _parse_text(data: bytes) -> list[ParsedPage]:
    return [ParsedPage(1, clean_text(data.decode("utf-8", errors="replace")))]


PARSERS = {".pdf": _parse_pdf, ".docx": _parse_docx, ".csv": _parse_csv, ".txt": _parse_text, ".md": _parse_text}


def parse_document(data: bytes, filename: str) -> list[ParsedPage]:
    extension = PurePath(filename).suffix.lower()
    parser = PARSERS.get(extension)
    if parser is None:
        raise InvalidDocument(f"Unsupported file type: {extension or filename}")
    try:
        pages = [page for page in parser(data) if page.content]
    except InvalidDocument:
        raise
    except Exception as exc:  # third-party parsers raise many types on corrupt files
        raise InvalidDocument(f"Could not parse {filename}: {exc}") from exc
    if not pages:
        raise InvalidDocument(f"No text could be extracted from {filename}")
    return pages
