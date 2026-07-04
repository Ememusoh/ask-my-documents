"""Simple, configurable chunking strategies for extracted document sections.

Chunk sizes in this module are measured in words. Word counts are easy to inspect
and explain during the project evaluation, without adding another tokenization
library. Every returned chunk keeps the source metadata from the extractor.
"""

from __future__ import annotations

import re
from typing import Any


Section = dict[str, Any]
Chunk = dict[str, Any]
SectionRange = dict[str, Any]


def clean_text(text: str) -> str:
    """Normalize whitespace without changing the words from the source document."""
    # A chunk may come from a manually created section, so clean it again here.
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def fixed_size_chunks(
    sections: list[Section], chunk_size: int, overlap: int = 0
) -> list[Chunk]:
    """Split each document into fixed-size word chunks.

    Args:
        sections: Cleaned source sections returned by ``rag.extractors``.
        chunk_size: Maximum number of words in each chunk.
        overlap: Number of words repeated from the previous chunk.

    Returns:
        Chunks with their text, word count, strategy settings, and source metadata.
    """
    _validate_chunk_settings(chunk_size, overlap)
    chunks: list[Chunk] = []
    step_size = chunk_size - overlap

    for document_sections in _group_sections_by_document(sections):
        words, section_ranges = _document_words_and_ranges(document_sections)
        if not words:
            continue

        for start in range(0, len(words), step_size):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            if not chunk_words:
                continue

            source_sections = _source_sections_for_range(section_ranges, start, end)
            source_chunk_number = (start // step_size) + 1
            chunks.append(
                _make_chunk(
                    source_sections=source_sections,
                    text=" ".join(chunk_words),
                    chunk_number=len(chunks) + 1,
                    source_chunk_number=source_chunk_number,
                    strategy="fixed_size",
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

            # The final chunk is allowed to be shorter because no more words remain.
            if end >= len(words):
                break

    return chunks


def paragraph_aware_chunks(sections: list[Section], chunk_size: int) -> list[Chunk]:
    """Create chunks that keep neighbouring source sections together when possible.

    Short paragraphs are combined until they reach ``chunk_size`` words. A section
    longer than ``chunk_size`` is split into smaller chunks on its own. This gives a
    simple alternative to fixed-size chunking for the four planned experiments.
    """
    _validate_chunk_settings(chunk_size, overlap=0)
    chunks: list[Chunk] = []
    current_sections: list[Section] = []
    current_texts: list[str] = []
    current_word_count = 0

    def save_current_chunk() -> None:
        """Save the current group of source sections, then start a new group."""
        nonlocal current_sections, current_texts, current_word_count
        if not current_sections:
            return

        chunks.append(
            _make_chunk(
                source_sections=current_sections,
                text="\n\n".join(current_texts),
                chunk_number=len(chunks) + 1,
                source_chunk_number=1,
                strategy="paragraph_aware",
                chunk_size=chunk_size,
                overlap=0,
            )
        )
        current_sections = []
        current_texts = []
        current_word_count = 0

    for section in sections:
        words = _section_words(section)
        if not words:
            continue

        section_text = " ".join(words)
        section_word_count = len(words)

        # Do not mix content from different documents in the same chunk.
        document_changed = current_sections and (
            section.get("document_name") != current_sections[0].get("document_name")
        )
        if document_changed:
            save_current_chunk()

        # A long paragraph cannot stay whole, so split it into word-sized pieces.
        if section_word_count > chunk_size:
            save_current_chunk()
            for start in range(0, section_word_count, chunk_size):
                chunks.append(
                    _make_chunk(
                        source_sections=[section],
                        text=" ".join(words[start : start + chunk_size]),
                        chunk_number=len(chunks) + 1,
                        source_chunk_number=(start // chunk_size) + 1,
                        strategy="paragraph_aware",
                        chunk_size=chunk_size,
                        overlap=0,
                    )
                )
            continue

        # Start a new chunk before adding a paragraph that would make it too large.
        if current_word_count and current_word_count + section_word_count > chunk_size:
            save_current_chunk()

        current_sections.append(section)
        current_texts.append(section_text)
        current_word_count += section_word_count

    save_current_chunk()
    return chunks


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    """Check settings early so invalid values do not cause confusing chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")


def _section_words(section: Section) -> list[str]:
    """Return the cleaned words from one extracted source section."""
    return clean_text(str(section.get("text", ""))).split()


def _group_sections_by_document(sections: list[Section]) -> list[list[Section]]:
    """Group sections so one chunk never mixes two different documents."""
    grouped_sections: dict[str, list[Section]] = {}

    for section in sections:
        document_name = str(section.get("document_name", "unknown_document"))
        grouped_sections.setdefault(document_name, []).append(section)

    return list(grouped_sections.values())


def _document_words_and_ranges(
    sections: list[Section],
) -> tuple[list[str], list[SectionRange]]:
    """Create one word list plus section word ranges for citation tracking."""
    document_words: list[str] = []
    section_ranges: list[SectionRange] = []

    for section in sections:
        section_words = _section_words(section)
        if not section_words:
            continue

        start = len(document_words)
        document_words.extend(section_words)
        end = len(document_words)
        section_ranges.append({"start": start, "end": end, "section": section})

    return document_words, section_ranges


def _source_sections_for_range(
    section_ranges: list[SectionRange], chunk_start: int, chunk_end: int
) -> list[Section]:
    """Find the original sections that overlap with one fixed-size chunk."""
    source_sections = []

    for section_range in section_ranges:
        section_start = int(section_range["start"])
        section_end = int(section_range["end"])

        # Two ranges overlap when each starts before the other one ends.
        if section_start < chunk_end and section_end > chunk_start:
            source_sections.append(section_range["section"])

    return source_sections


def _make_chunk(
    *,
    source_sections: list[Section],
    text: str,
    chunk_number: int,
    source_chunk_number: int,
    strategy: str,
    chunk_size: int,
    overlap: int,
) -> Chunk:
    """Build one chunk dictionary with clear source and experiment metadata."""
    first_section = source_sections[0]
    source_ids = [str(section["source_id"]) for section in source_sections]
    source_labels = [str(section["source_label"]) for section in source_sections]

    # Start with the first source's metadata, then add information specific to a chunk.
    chunk: Chunk = {
        **first_section,
        "chunk_id": f"{first_section['source_id']}:chunk:{source_chunk_number}",
        "chunk_number": chunk_number,
        "chunking_strategy": strategy,
        "chunk_size_words": chunk_size,
        "overlap_words": overlap,
        "word_count": len(text.split()),
        "source_ids": source_ids,
        "source_labels": source_labels,
        "source_count": len(source_sections),
        "source_label": " | ".join(source_labels),
        "text": text,
    }
    return chunk
