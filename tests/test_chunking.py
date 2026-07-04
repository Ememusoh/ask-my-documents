"""Tests for the two project-specific chunking strategies."""

from rag.chunking import fixed_size_chunks, paragraph_aware_chunks


def _section(number: int, text: str) -> dict:
    """Create a small fake extracted section for chunking tests."""
    return {
        "document_name": "demo.txt",
        "document_path": "demo.txt",
        "file_type": "txt",
        "source_id": f"demo.txt:paragraph:{number}",
        "source_type": "paragraph",
        "source_number": number,
        "source_label": f"demo.txt, paragraph {number}",
        "text": text,
    }


def test_fixed_size_chunking_uses_document_level_word_windows() -> None:
    """Fixed-size chunks should not create tiny chunks for every short paragraph."""
    sections = [
        _section(1, "one two"),
        _section(2, "three four five six"),
        _section(3, "seven eight nine"),
    ]

    chunks = fixed_size_chunks(sections, chunk_size=5)

    assert [chunk["word_count"] for chunk in chunks] == [5, 4]
    assert chunks[0]["text"] == "one two three four five"
    assert chunks[1]["text"] == "six seven eight nine"
    assert chunks[0]["source_count"] == 2
    assert chunks[1]["source_count"] == 2


def test_fixed_size_chunking_supports_overlap() -> None:
    """Overlap should repeat the last words from the previous fixed-size chunk."""
    sections = [_section(1, "one two three four five six seven")]

    chunks = fixed_size_chunks(sections, chunk_size=4, overlap=1)

    assert chunks[0]["text"] == "one two three four"
    assert chunks[1]["text"] == "four five six seven"


def test_paragraph_aware_chunking_respects_boundaries() -> None:
    """Paragraph-aware chunks should combine short paragraphs without cutting them."""
    sections = [
        _section(1, "one two"),
        _section(2, "three four"),
        _section(3, "five six seven"),
    ]

    chunks = paragraph_aware_chunks(sections, chunk_size=5)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "one two\n\nthree four"
    assert chunks[1]["text"] == "five six seven"
