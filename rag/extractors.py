"""Extract clean text and citation-ready metadata from supported documents.

Each extractor returns a list of dictionaries. A dictionary represents one source section:

* TXT files are split into paragraphs and retain their line range.
* PDF files are split into pages and retain their page number.
* DOCX files are split into paragraphs and retain their paragraph number.

The later chunking step will split these sections further while preserving this metadata.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterator

from docx import Document
from pypdf import PdfReader


Section = dict[str, Any]
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def clean_text(text: str) -> str:
    """Return readable text with repeated whitespace reduced to one space.

    This small cleanup removes common extraction artifacts, such as newlines in the
    middle of a sentence. It deliberately does not lowercase or otherwise alter
    the words because the displayed evidence should remain close to the source.
    """
    text = text.replace("\u00a0", " ")  # Replace non-breaking spaces with normal spaces.
    return re.sub(r"\s+", " ", text).strip()


def extract_txt(file_path: str | Path) -> list[Section]:
    """Extract clean paragraphs and line-range metadata from a TXT file."""
    path = _validate_file(file_path, ".txt")
    raw_text = path.read_text(encoding="utf-8-sig")
    sections: list[Section] = []

    # A blank line marks a paragraph boundary in a plain-text document.
    for paragraph_number, (line_start, line_end, paragraph) in enumerate(
        _txt_paragraphs(raw_text), start=1
    ):
        cleaned_paragraph = clean_text(paragraph)
        if cleaned_paragraph:
            sections.append(
                _make_section(
                    path=path,
                    text=cleaned_paragraph,
                    source_type="paragraph",
                    source_number=paragraph_number,
                    line_start=line_start,
                    line_end=line_end,
                )
            )

    return sections


def extract_pdf(file_path: str | Path) -> list[Section]:
    """Extract clean page-level text and metadata from a text-based PDF file."""
    path = _validate_file(file_path, ".pdf")
    reader = PdfReader(path)

    # Password-protected PDFs are outside this prototype's current scope.
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError(f"Cannot extract text from password-protected PDF: {path.name}")

    sections: list[Section] = []
    for page_number, page in enumerate(reader.pages, start=1):
        # pypdf returns None when a page has no extractable text.
        cleaned_page_text = clean_text(page.extract_text() or "")
        if cleaned_page_text:
            sections.append(
                _make_section(
                    path=path,
                    text=cleaned_page_text,
                    source_type="page",
                    source_number=page_number,
                )
            )

    return sections


def extract_docx(file_path: str | Path) -> list[Section]:
    """Extract clean paragraph-level text and metadata from a DOCX file."""
    path = _validate_file(file_path, ".docx")
    document = Document(path)
    sections: list[Section] = []

    for paragraph_number, paragraph in enumerate(document.paragraphs, start=1):
        cleaned_paragraph = clean_text(paragraph.text)
        # Empty paragraphs are layout spacing, not useful evidence for retrieval.
        if cleaned_paragraph:
            sections.append(
                _make_section(
                    path=path,
                    text=cleaned_paragraph,
                    source_type="paragraph",
                    source_number=paragraph_number,
                )
            )

    return sections


def extract_document(file_path: str | Path) -> list[Section]:
    """Extract sections from a supported TXT, PDF, or DOCX file.

    Raises:
        FileNotFoundError: If the provided path does not point to a file.
        ValueError: If the file extension is not supported.
    """
    path = _validate_file(file_path)
    extractors: dict[str, Callable[[Path], list[Section]]] = {
        ".txt": extract_txt,
        ".pdf": extract_pdf,
        ".docx": extract_docx,
    }

    try:
        return extractors[path.suffix.lower()](path)
    except KeyError as error:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Use one of: {supported}."
        ) from error


def _validate_file(file_path: str | Path, expected_suffix: str | None = None) -> Path:
    """Return an existing file path and optionally check its expected extension."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file was not found: {path}")

    if expected_suffix and path.suffix.lower() != expected_suffix:
        raise ValueError(f"Expected a {expected_suffix} file, received: {path.name}")

    return path


def _txt_paragraphs(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield each non-empty TXT paragraph with its inclusive line range."""
    paragraph_lines: list[str] = []
    line_start: int | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if line_start is None:
                line_start = line_number
            paragraph_lines.append(line)
            continue

        if paragraph_lines:
            yield line_start or line_number, line_number - 1, "\n".join(paragraph_lines)
            paragraph_lines = []
            line_start = None

    # Yield the final paragraph when the file does not end with a blank line.
    if paragraph_lines:
        yield line_start or 1, len(text.splitlines()), "\n".join(paragraph_lines)


def _make_section(
    *,
    path: Path,
    text: str,
    source_type: str,
    source_number: int,
    line_start: int | None = None,
    line_end: int | None = None,
) -> Section:
    """Create the consistent metadata shape shared by all extractors."""
    source_label = f"{path.name}, {source_type} {source_number}"
    section: Section = {
        "document_name": path.name,
        "document_path": str(path),
        "file_type": path.suffix.lower().lstrip("."),
        "source_id": f"{path.name}:{source_type}:{source_number}",
        "source_type": source_type,
        "source_number": source_number,
        "source_label": source_label,
        "text": text,
    }

    if source_type == "page":
        section["page_number"] = source_number
    if source_type == "paragraph":
        section["paragraph_number"] = source_number
    if line_start is not None and line_end is not None:
        section["line_start"] = line_start
        section["line_end"] = line_end

    return section
